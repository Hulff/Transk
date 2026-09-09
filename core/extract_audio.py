"""
Extração de áudio de arquivos de vídeo usando ffmpeg.
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg


def extract_audio(video_path: str, output_dir: str = "temp") -> str:
    """
    Extrai o áudio de um vídeo e salva como .wav mono 16kHz
    (formato ideal para o Whisper/WhisperX).

    Args:
        video_path: caminho do arquivo de vídeo (mp4, mkv, etc).
        output_dir: pasta onde o .wav será salvo.

    Returns:
        Caminho do arquivo .wav gerado.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_path = Path(output_dir) / f"{video_path.stem}.wav"

    # Usa o FFmpeg fornecido pelo pacote imageio-ffmpeg.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Erro ao extrair áudio:\n{result.stderr}"
        )

    return str(audio_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python extract_audio.py <caminho_do_video>")
        sys.exit(1)

    caminho = extract_audio(sys.argv[1])
    print(f"Áudio extraído em: {caminho}")
