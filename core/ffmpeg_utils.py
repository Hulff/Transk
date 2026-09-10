"""
Resolve o caminho do binário do ffmpeg usando o imageio-ffmpeg, que
baixa e empacota um binário próprio na primeira execução — assim não
é preciso instalar o ffmpeg no sistema operacional separadamente.
"""

import imageio_ffmpeg


def get_ffmpeg_path() -> str:
    """Retorna o caminho do binário do ffmpeg gerenciado pelo imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()
