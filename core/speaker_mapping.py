"""
Mapeamento de speakers para personagens usando embeddings ECAPA.

IMPORTANTE:

A diarização produz:

    SPEAKER_00
    SPEAKER_01

Esses IDs são preservados.

Nunca substituímos `speaker` pelo nome do personagem.

Em vez disso adicionamos:

    speaker_id
    voice_character
    voice_score
    voice_margin
    voice_status

Isso permite que o Identity Resolver combine voz + visão + histórico.
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio

from speechbrain.inference.speaker import EncoderClassifier

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


# ---------------------------------------------------------------------------
# Similaridade
# ---------------------------------------------------------------------------


def _cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def _load_embedder(
    device: str = "cpu",
):
    """
    Carrega ECAPA pelo SpeechBrain.
    """

    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts={"device": device},
    )


# ---------------------------------------------------------------------------
# Áudio
# ---------------------------------------------------------------------------


def _load_audio(
    audio_path: str,
):
    """
    Carrega áudio e converte para mono.
    """

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    return (
        waveform,
        sample_rate,
    )


def _resample(
    waveform,
    sample_rate,
    target_sr=16000,
):
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sr,
        )

        sample_rate = target_sr

    return (
        waveform,
        sample_rate,
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _embedding_from_waveform(
    embedder,
    waveform,
    sample_rate,
):
    waveform, sample_rate = _resample(
        waveform,
        sample_rate,
        target_sr=16000,
    )

    waveform = waveform.to(torch.float32)

    with torch.no_grad():
        embedding = embedder.encode_batch(waveform)

    return embedding.squeeze().cpu().numpy()


def _embedding_from_file(
    embedder,
    audio_path: str,
):
    waveform, sample_rate = _load_audio(audio_path)

    return _embedding_from_waveform(
        embedder,
        waveform,
        sample_rate,
    )


# ---------------------------------------------------------------------------
# Voice profiles
# ---------------------------------------------------------------------------


def build_voice_profiles(
    voice_profiles_dir: str,
    hf_token: str = "",
    device: str = "cpu",
) -> dict:
    """
    Gera um perfil médio por personagem.

    Exemplo:

        models/voice_profiles/
            goku/
                sample1.wav
                sample2.wav

            vegeta/
                sample1.wav
                sample2.wav
    """

    base = Path(voice_profiles_dir)

    if not base.exists():
        return {}

    embedder = _load_embedder(device)

    profiles = {}

    for character_dir in sorted(base.iterdir()):

        if not character_dir.is_dir():
            continue

        embeddings = []

        for audio_file in sorted(character_dir.glob("*.wav")):

            try:

                print(
                    "Gerando embedding: " f"{character_dir.name}/" f"{audio_file.name}"
                )

                embedding = _embedding_from_file(
                    embedder,
                    str(audio_file),
                )

                embeddings.append(embedding)

            except Exception as error:

                print("Aviso: não foi possível " f"processar {audio_file}: " f"{error}")

        if not embeddings:
            continue

        profile = np.mean(
            np.stack(embeddings),
            axis=0,
        )

        norm = np.linalg.norm(profile)

        if norm > 0:
            profile = profile / norm

        profiles[character_dir.name] = profile

    return profiles


# ---------------------------------------------------------------------------
# Speaker chunks
# ---------------------------------------------------------------------------


def build_speaker_embedding_samples(
    audio_path: str,
    segments: list[dict],
    hf_token: str = "",
    device: str = "cpu",
    max_samples_per_speaker: int = 8,
    min_segment_seconds: float = 1.0,
    max_segment_seconds: float = 8.0,
) -> dict:
    """
    Gera múltiplos embeddings por speaker.

    Ao contrário da versão anterior, não concatena cegamente todos
    os trechos e pega os primeiros 10 segundos.

    Isso reduz o impacto de:

    - música;
    - efeitos;
    - gritos;
    - ruído;
    - segmentos muito curtos;
    - uma única fala atípica.
    """

    waveform, sample_rate = _load_audio(audio_path)

    waveform, sample_rate = _resample(
        waveform,
        sample_rate,
        target_sr=16000,
    )

    audio = waveform.squeeze(0)

    speaker_segments = {}

    for segment in segments:

        speaker = segment.get("speaker_id") or segment.get("speaker")

        if not speaker:
            continue

        start = float(
            segment.get(
                "start",
                0,
            )
        )

        end = float(
            segment.get(
                "end",
                start,
            )
        )

        duration = end - start

        if duration < min_segment_seconds:
            continue

        duration = min(
            duration,
            max_segment_seconds,
        )

        end = start + duration

        start_sample = max(
            0,
            int(start * sample_rate),
        )

        end_sample = min(
            audio.shape[0],
            int(end * sample_rate),
        )

        if end_sample <= start_sample:
            continue

        chunk = audio[start_sample:end_sample]

        if chunk.numel() == 0:
            continue

        speaker_segments.setdefault(
            speaker,
            [],
        ).append(chunk)

    embedder = _load_embedder(device)

    speaker_embeddings = {}

    for speaker, chunks in speaker_segments.items():

        # Priorizamos trechos mais longos.
        chunks = sorted(
            chunks,
            key=lambda x: x.numel(),
            reverse=True,
        )

        chunks = chunks[:max_samples_per_speaker]

        embeddings = []

        for chunk in chunks:

            try:

                embedding = _embedding_from_waveform(
                    embedder,
                    chunk.unsqueeze(0),
                    sample_rate,
                )

                embeddings.append(embedding)

            except Exception as error:

                print("Aviso: erro gerando " f"embedding de {speaker}: " f"{error}")

        if embeddings:
            speaker_embeddings[speaker] = embeddings

    return speaker_embeddings


def aggregate_speaker_embeddings(
    speaker_embeddings: dict,
) -> dict:
    """
    Agrega múltiplos embeddings de cada speaker.

    Usa a média dos embeddings normalizados.
    """

    result = {}

    for speaker, embeddings in speaker_embeddings.items():

        if not embeddings:
            continue

        normalized = []

        for embedding in embeddings:

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            norm = np.linalg.norm(embedding)

            if norm > 0:
                embedding = embedding / norm

            normalized.append(embedding)

        profile = np.mean(
            np.stack(normalized),
            axis=0,
        )

        norm = np.linalg.norm(profile)

        if norm > 0:
            profile = profile / norm

        result[speaker] = profile

    return result


def build_speaker_embeddings(
    audio_path: str,
    segments: list[dict],
    hf_token: str = "",
    device: str = "cpu",
) -> dict:
    """
    Compatibilidade com a API anterior.

    Agora usa múltiplos segmentos e agrega os embeddings.
    """

    samples = build_speaker_embedding_samples(
        audio_path,
        segments,
        hf_token=hf_token,
        device=device,
    )

    return aggregate_speaker_embeddings(samples)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _rank_voice_profiles(
    embedding: np.ndarray,
    profiles: dict,
) -> list[tuple[str, float]]:
    results = []

    for character, profile in profiles.items():

        score = _cosine_similarity(
            embedding,
            profile,
        )

        results.append(
            (
                character,
                score,
            )
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results


def map_speakers_to_characters(
    segments: list[dict],
    audio_path: str,
    voice_profiles_dir: str,
    hf_token: str = "",
    similarity_threshold: float = 0.75,
    margin_threshold: float = 0.08,
    device: str = "cpu",
) -> list[dict]:
    """
    Associa SPEAKER_XX a um personagem por voz.

    IMPORTANTE:

    Nunca substitui:

        seg["speaker"]

    pelo nome do personagem.

    Mantemos:

        speaker_id
        voice_character
        voice_score
        voice_margin
        voice_status
    """

    profiles = build_voice_profiles(
        voice_profiles_dir,
        hf_token,
        device=device,
    )

    # Garante speaker_id nos segmentos antigos.
    for segment in segments:

        if not segment.get("speaker_id"):

            segment["speaker_id"] = segment.get(
                "speaker",
                "SPEAKER_UNKNOWN",
            )

    if not profiles:

        print("Nenhum perfil de voz encontrado. " "Mantendo somente SPEAKER_XX.")

        for segment in segments:
            segment["voice_character"] = None

            segment["voice_score"] = 0.0

            segment["voice_margin"] = 0.0

            segment["voice_status"] = "SEM_PERFIL"

        return segments

    print(f"{len(profiles)} perfil(is) " "de personagem encontrado(s).")

    speaker_embeddings = build_speaker_embeddings(
        audio_path,
        segments,
        hf_token,
        device=device,
    )

    speaker_to_result = {}

    for speaker, embedding in speaker_embeddings.items():

        ranked = _rank_voice_profiles(
            embedding,
            profiles,
        )

        if not ranked:
            continue

        best_character, best_score = ranked[0]

        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        margin = best_score - second_score

        if best_score < similarity_threshold:
            status = "SEM_MATCH"

        elif margin < margin_threshold:
            status = "AMBIGUO"

        else:
            status = "CONFIRMADO"

        speaker_to_result[speaker] = {
            "character": best_character,
            "score": float(best_score),
            "margin": float(margin),
            "status": status,
        }

        print(
            f"{speaker} -> "
            f"{best_character} "
            f"(similaridade: "
            f"{best_score:.3f}, "
            f"margem: {margin:.3f}, "
            f"status: {status})"
        )

    for segment in segments:

        speaker = segment.get("speaker_id") or segment.get("speaker")

        result = speaker_to_result.get(speaker)

        if result is None:

            segment["voice_character"] = None

            segment["voice_score"] = 0.0

            segment["voice_margin"] = 0.0

            segment["voice_status"] = "SEM_EMBEDDING"

            continue

        segment["voice_character"] = result["character"]

        segment["voice_score"] = round(
            result["score"],
            4,
        )

        segment["voice_margin"] = round(
            result["margin"],
            4,
        )

        segment["voice_status"] = result["status"]

        # speaker continua intacto.

    return segments


# ---------------------------------------------------------------------------
# Mapeamento manual
# ---------------------------------------------------------------------------


def mapear_manual(
    segments: list[dict],
    mapeamento: dict,
) -> list[dict]:
    """
    Mapeamento manual sem destruir o speaker original.

    Exemplo:

        {
            "SPEAKER_00": "Goku",
            "SPEAKER_01": "Vegeta",
        }
    """

    for segment in segments:

        speaker = segment.get("speaker_id") or segment.get("speaker")

        if not speaker:
            continue

        character = mapeamento.get(speaker)

        if character:

            segment["voice_character"] = character

            segment["voice_status"] = "MANUAL"

            segment["voice_score"] = 1.0

            segment["voice_margin"] = 1.0

    return segments
