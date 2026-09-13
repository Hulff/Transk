"""
Transcrição + diarização de falantes usando WhisperX.

Retorna:

[
    {
        "start": 12.3,
        "end": 15.8,
        "speaker_id": "SPEAKER_00",
        "speaker": "SPEAKER_00",
        "text": "Não vou fazer isso."
    }
]

`speaker_id` é a identidade original produzida pela diarização.

Ele NÃO deve ser substituído pelo nome de personagem.
"""

import gc

import torch
import whisperx


def transcribe_and_diarize(
    audio_path: str,
    model_name: str = "large-v2",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    hf_token: str = "",
    min_speakers: int = None,
    max_speakers: int = None,
) -> list[dict]:
    """
    Executa:

        WhisperX
        ↓
        alinhamento
        ↓
        diarização
        ↓
        associação palavra -> speaker

    O resultado mantém o speaker original.
    """

    # ------------------------------------------------------------------
    # 1. Transcrição
    # ------------------------------------------------------------------

    print(f"[transcribe] Carregando WhisperX: " f"{model_name}")

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )

    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(
        audio,
        language=language,
        batch_size=16,
    )

    del model

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Alinhamento
    # ------------------------------------------------------------------

    print("[transcribe] Alinhando timestamps...")

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )

    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
    )

    del align_model

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Diarização
    # ------------------------------------------------------------------

    print("[transcribe] Executando diarização...")

    try:

        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )

    except TypeError:

        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=hf_token,
            device=device,
        )

    diarize_segments = diarize_model(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    result = whisperx.assign_word_speakers(
        diarize_segments,
        result,
    )

    # ------------------------------------------------------------------
    # 4. Formatação
    # ------------------------------------------------------------------

    segments = []

    for segment in result["segments"]:

        speaker_id = segment.get(
            "speaker",
            "SPEAKER_UNKNOWN",
        )

        text = segment.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        segments.append(
            {
                "start": round(
                    float(
                        segment.get(
                            "start",
                            0,
                        )
                    ),
                    2,
                ),
                "end": round(
                    float(
                        segment.get(
                            "end",
                            0,
                        )
                    ),
                    2,
                ),
                "speaker_id": speaker_id,
                "speaker": speaker_id,
                "text": text,
            }
        )

    print("[transcribe] " f"{len(segments)} segmentos produzidos.")

    return segments


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Uso: python transcribe.py " "<caminho_do_audio.wav>")
        sys.exit(1)

    segments = transcribe_and_diarize(sys.argv[1])

    print(
        json.dumps(
            segments,
            ensure_ascii=False,
            indent=2,
        )
    )
