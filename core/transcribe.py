"""
Transcrição + diarização de falantes usando WhisperX.

Retorna uma lista de segmentos no formato:
[
    {
        "start": 12.3,
        "end": 15.8,
        "speaker": "SPEAKER_00",
        "text": "Não vou fazer isso."
    },
    ...
]
"""
import whisperx
import gc
import torch


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
    Executa o pipeline completo do WhisperX: transcrição, alinhamento
    de timestamps por palavra e diarização de falantes.
    """
    # 1. Transcrição
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language=language, batch_size=16)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. Alinhamento (melhora precisão dos timestamps)
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device
    )

    del align_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Diarização (quem falou o quê)
    # Versões diferentes do WhisperX usam nomes diferentes para o parâmetro
    # do token (use_auth_token nas mais antigas, token nas mais novas).
    try:
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
    except TypeError:
        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=hf_token, device=device
        )
    diarize_segments = diarize_model(
        audio_path, min_speakers=min_speakers, max_speakers=max_speakers
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 4. Formata saída simplificada
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "speaker": seg.get("speaker", "SPEAKER_UNKNOWN"),
            "text": seg.get("text", "").strip(),
        })

    return segments


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Uso: python transcribe.py <caminho_do_audio.wav>")
        sys.exit(1)

    segs = transcribe_and_diarize(sys.argv[1])
    print(json.dumps(segs, ensure_ascii=False, indent=2))