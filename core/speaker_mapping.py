"""
Mapeamento de speakers para personagens usando embeddings ECAPA do SpeechBrain.

Estrutura esperada:

models/voice_profiles/
    joao/
        amostra1.wav
        amostra2.wav
    maria/
        amostra1.wav
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def _load_embedder(device: str = "cpu"):
    """
    Carrega o ECAPA diretamente pelo SpeechBrain.

    Isso evita a incompatibilidade entre:
        pyannote.audio 4.x
        SpeechBrain 1.x
    """

    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts={"device": device},
    )


def _load_audio(audio_path: str):
    """
    Carrega áudio e converte para mono float32.
    """

    waveform, sample_rate = torchaudio.load(audio_path)

    # Stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sample_rate


def _resample(waveform, sample_rate, target_sr=16000):
    """
    ECAPA normalmente trabalha em 16 kHz.
    """

    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sr,
        )
        sample_rate = target_sr

    return waveform, sample_rate


def _embedding_from_waveform(embedder, waveform, sample_rate):
    """
    Gera embedding ECAPA a partir de um tensor [1, samples].
    """

    waveform, sample_rate = _resample(
        waveform,
        sample_rate,
        target_sr=16000,
    )

    waveform = waveform.to(torch.float32)

    with torch.no_grad():
        embedding = embedder.encode_batch(waveform)

    return embedding.squeeze().cpu().numpy()


def _embedding_from_file(embedder, audio_path: str):
    """
    Gera embedding a partir de um arquivo de áudio.
    """

    waveform, sample_rate = _load_audio(audio_path)

    return _embedding_from_waveform(
        embedder,
        waveform,
        sample_rate,
    )


def build_voice_profiles(
    voice_profiles_dir: str,
    hf_token: str = "",
    device: str = "cpu",
) -> dict:
    """
    Gera um embedding médio para cada personagem.

    Exemplo:

        models/voice_profiles/joao/*.wav
        models/voice_profiles/maria/*.wav
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
                    f"Gerando embedding: "
                    f"{character_dir.name}/{audio_file.name}"
                )

                emb = _embedding_from_file(
                    embedder,
                    str(audio_file),
                )

                embeddings.append(emb)

            except Exception as e:
                print(
                    f"Aviso: não foi possível processar "
                    f"{audio_file}: {e}"
                )

        if embeddings:

            profile = np.mean(
                np.stack(embeddings),
                axis=0,
            )

            # Normaliza o perfil
            norm = np.linalg.norm(profile)

            if norm > 0:
                profile = profile / norm

            profiles[character_dir.name] = profile

    return profiles


def build_speaker_embeddings(
    audio_path: str,
    segments: list[dict],
    hf_token: str = "",
    device: str = "cpu",
) -> dict:
    """
    Gera um embedding para cada SPEAKER_XX detectado pelo WhisperX.

    Os segmentos daquele speaker são concatenados e limitados
    a aproximadamente 10 segundos.
    """

    waveform, sample_rate = _load_audio(audio_path)

    waveform, sample_rate = _resample(
        waveform,
        sample_rate,
        target_sr=16000,
    )

    audio = waveform.squeeze(0)

    speaker_chunks = {}

    for seg in segments:

        speaker = seg.get("speaker")

        if not speaker:
            continue

        start = float(seg["start"])
        end = float(seg["end"])

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

        if chunk.numel() > 0:
            speaker_chunks.setdefault(
                speaker,
                [],
            ).append(chunk)

    embedder = _load_embedder(device)

    speaker_embeddings = {}

    max_samples = 10 * sample_rate

    for speaker, chunks in speaker_chunks.items():

        concatenated = torch.cat(chunks)

        # Limita a aproximadamente 10 segundos
        concatenated = concatenated[:max_samples]

        # Muito pouco áudio pode gerar embedding ruim
        if concatenated.numel() < sample_rate:
            print(
                f"Aviso: {speaker} possui menos de 1s de áudio."
            )

        concatenated = concatenated.unsqueeze(0)

        try:

            emb = _embedding_from_waveform(
                embedder,
                concatenated,
                sample_rate,
            )

            speaker_embeddings[speaker] = emb

        except Exception as e:

            print(
                f"Aviso: não foi possível gerar embedding "
                f"para {speaker}: {e}"
            )

    return speaker_embeddings


def map_speakers_to_characters(
    segments: list[dict],
    audio_path: str,
    voice_profiles_dir: str,
    hf_token: str = "",
    similarity_threshold: float = 0.75,
    device: str = "cpu",
) -> list[dict]:
    """
    Substitui SPEAKER_XX pelo personagem correspondente.

    Se nenhum perfil atingir o threshold, mantém SPEAKER_XX.
    """

    profiles = build_voice_profiles(
        voice_profiles_dir,
        hf_token,
        device=device,
    )

    if not profiles:
        print(
            "Nenhum perfil de voz encontrado. "
            "Mantendo SPEAKER_XX."
        )

        return segments

    print(
        f"{len(profiles)} perfil(is) de personagem encontrado(s)."
    )

    speaker_embeddings = build_speaker_embeddings(
        audio_path,
        segments,
        hf_token,
        device=device,
    )

    speaker_to_character = {}

    for speaker, emb in speaker_embeddings.items():

        best_match = None
        best_score = -1.0

        for character, profile_emb in profiles.items():

            score = _cosine_similarity(
                emb,
                profile_emb,
            )

            if score > best_score:
                best_match = character
                best_score = score

        print(
            f"{speaker} -> {best_match} "
            f"(similaridade: {best_score:.3f})"
        )

        if (
            best_match is not None
            and best_score >= similarity_threshold
        ):
            speaker_to_character[speaker] = best_match

        else:
            speaker_to_character[speaker] = speaker

    for seg in segments:

        speaker = seg.get("speaker")

        if speaker:
            seg["speaker"] = speaker_to_character.get(
                speaker,
                speaker,
            )

    return segments


def mapear_manual(
    segments: list[dict],
    mapeamento: dict,
) -> list[dict]:
    """
    Mapeamento manual.

    Exemplo:

        {
            "SPEAKER_00": "João",
            "SPEAKER_01": "Maria",
        }
    """

    for seg in segments:

        speaker = seg.get("speaker")

        if speaker:
            seg["speaker"] = mapeamento.get(
                speaker,
                speaker,
            )

    return segments
