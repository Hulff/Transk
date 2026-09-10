"""
Resolve o caminho do binário do ffmpeg usando o imageio-ffmpeg, que
baixa e empacota um binário próprio na primeira execução — assim não
é preciso instalar o ffmpeg no sistema operacional separadamente.

Algumas libs (como o WhisperX) chamam o comando "ffmpeg" diretamente
via subprocess, esperando encontrá-lo no PATH do sistema — sem passar
pelo imageio-ffmpeg. Pra cobrir esses casos, `ensure_ffmpeg_in_path()`
cria um atalho chamado "ffmpeg" dentro do venv atual (pasta bin/),
apontando pro binário baixado pelo imageio-ffmpeg.
"""

import os
import shutil
import sys
from pathlib import Path

import imageio_ffmpeg


def get_ffmpeg_path() -> str:
    """Retorna o caminho do binário do ffmpeg gerenciado pelo imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def ensure_ffmpeg_in_path() -> None:
    """
    Garante que o comando "ffmpeg" (nome exato) existe no PATH,
    criando um symlink (ou cópia, se symlink não for suportado) dentro
    da pasta bin/ do venv atual. Chame isso uma vez, cedo, antes de
    qualquer lib que dependa de "ffmpeg" estar no PATH (ex: WhisperX).
    """
    ffmpeg_real_path = Path(get_ffmpeg_path())
    bin_dir = Path(sys.prefix) / "bin"  # pasta bin/ do venv ativo
    bin_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_link = bin_dir / "ffmpeg"

    if ffmpeg_link.exists():
        return  # já configurado

    try:
        ffmpeg_link.symlink_to(ffmpeg_real_path)
    except OSError:
        # sistema de arquivos sem suporte a symlink (raro) — copia o binário
        shutil.copy(ffmpeg_real_path, ffmpeg_link)
        os.chmod(ffmpeg_link, 0o755)
