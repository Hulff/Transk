"""
Transcritor de Episódios — CLI

Comandos disponíveis:

    # Só a parte de áudio (transcrição + diarização + personagens por voz)
    python app.py audio caminho/do/episodio.mp4

    # Só a parte de cena (requer que "audio" já tenha rodado antes, se
    # o modo for "fala" — veja config.yaml)
    python app.py cena caminho/do/episodio.mp4

    # As duas juntas (reaproveita cache se "audio" ou "cena" já rodaram antes)
    python app.py processar caminho/do/episodio.mp4

    # Forçar recomputar tudo, ignorando cache
    python app.py processar caminho/do/episodio.mp4 --forcar

    # Processar tudo, mas sem cena mesmo que esteja habilitada no config
    python app.py processar caminho/do/episodio.mp4 --sem-cena
"""

import typer
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

from core.ffmpeg_utils import ensure_ffmpeg_in_path

ensure_ffmpeg_in_path()  # precisa rodar antes de importar/usar o WhisperX

from core.extract_audio import extract_audio
from core.transcribe import transcribe_and_diarize
from core.speaker_mapping import map_speakers_to_characters
from core.scene_analysis import (
    describe_dialogue_scenes,
    group_into_scenes,
    describe_scenes,
    DEFAULT_QUESTION,
)
from core.merge import merge_timeline
from core.cache import save_cache, load_cache, has_cache
from output.formatter import save_outputs

load_dotenv()  # lê o arquivo .env (se existir) e popula os.environ

app = typer.Typer(help="Transcritor de episódios: falas por personagem + ações de cena")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Token do Hugging Face: prioriza a variável de ambiente HF_TOKEN (via .env),
    # com fallback pro valor em config.yaml (que deve ficar vazio no repo).
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        cfg["diarization"]["hf_token"] = env_token
    elif not cfg["diarization"].get("hf_token"):
        raise RuntimeError(
            "Token do Hugging Face não encontrado. Crie um arquivo .env "
            "(copie de .env.example) com HF_TOKEN=seu_token, ou preencha "
            "diarization.hf_token no config.yaml."
        )

    return cfg


def _rodar_audio(video_path: str, cfg: dict, base_name: str) -> list[dict]:
    """Extrai áudio, transcreve, diariza e mapeia personagens por voz."""
    typer.echo("Extraindo áudio...")
    audio_path = extract_audio(video_path)

    typer.echo("Transcrevendo e identificando falantes (pode demorar)...")
    segments = transcribe_and_diarize(
        audio_path,
        model_name=cfg["whisper"]["model"],
        language=cfg["whisper"]["language"],
        device=cfg["whisper"]["device"],
        compute_type=cfg["whisper"]["compute_type"],
        hf_token=cfg["diarization"]["hf_token"],
        min_speakers=cfg["diarization"].get("min_speakers"),
        max_speakers=cfg["diarization"].get("max_speakers"),
    )

    typer.echo("Reconhecendo personagens por voz (se houver perfis cadastrados)...")
    segments = map_speakers_to_characters(
        segments,
        audio_path=audio_path,
        voice_profiles_dir=cfg["speaker_mapping"]["voice_profiles_dir"],
        hf_token=cfg["diarization"]["hf_token"],
        similarity_threshold=cfg["speaker_mapping"]["similarity_threshold"],
    )

    save_cache(base_name, "falas", segments)
    return segments


def _rodar_cena(
    video_path: str, cfg: dict, base_name: str, segments: list[dict] = None
) -> list[dict]:
    """
    Extrai e descreve cenas usando o Qwen2.5-VL. Precisa das falas já
    processadas (passadas em `segments` ou carregadas do cache).
    """
    modo = cfg["scene_analysis"].get("mode", "fala")
    sa = cfg["scene_analysis"]

    if modo not in ("fala", "cena"):
        raise RuntimeError(
            "Só os modos 'fala' e 'cena' são suportados com o Qwen2.5-VL no momento."
        )

    if segments is None:
        segments = load_cache(base_name, "falas")
    if not segments:
        raise RuntimeError(
            f"O modo '{modo}' precisa das falas já processadas. "
            "Rode 'python app.py audio <video>' primeiro."
        )

    common_kwargs = dict(
        model_name=sa.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
        padding_seconds=sa.get("padding_seconds", 0.5),
        question=sa.get("question") or DEFAULT_QUESTION,
        language=sa.get("language", "pt-BR"),
        max_new_tokens=sa.get("max_new_tokens", 80),
        repetition_penalty=sa.get("repetition_penalty", 1.15),
        no_repeat_ngram_size=sa.get("no_repeat_ngram_size", 3),
        do_sample=sa.get("do_sample", False),
        load_in_4bit=sa.get("load_in_4bit", False),
    )

    if modo == "fala":
        typer.echo(f"Analisando cena ancorada em {len(segments)} falas...")
        scenes = describe_dialogue_scenes(
            segments,
            video_path,
            frames_per_fala=sa.get("frames_per_fala", 5),
            **common_kwargs,
        )
    else:  # modo == "cena"
        gap = sa.get("gap_threshold_seconds", 3.0)
        grouped = group_into_scenes(segments, gap_threshold_seconds=gap)
        typer.echo(
            f"{len(segments)} falas agrupadas em {len(grouped)} cenas (gap > {gap}s = nova cena)..."
        )
        common_kwargs.pop("padding_seconds", None)
        scenes = describe_scenes(
            grouped,
            video_path,
            frames_per_scene=sa.get("frames_per_scene", 8),
            padding_seconds=sa.get("padding_seconds", 0.5),
            **common_kwargs,
        )

    save_cache(base_name, "cenas", scenes)
    return scenes


@app.command()
def audio(
    video_path: str = typer.Argument(..., help="Caminho do arquivo de vídeo"),
    config_path: str = typer.Option(
        "config.yaml", help="Caminho do arquivo de configuração"
    ),
):
    """Roda só a análise de áudio (falas por personagem) e salva em cache."""
    cfg = load_config(config_path)
    base_name = Path(video_path).stem

    segments = _rodar_audio(video_path, cfg, base_name)

    typer.echo(f"\nConcluído! {len(segments)} falas identificadas e salvas em cache.")
    typer.echo("Rode 'python app.py processar' quando quiser gerar o roteiro final.")


@app.command()
def cena(
    video_path: str = typer.Argument(..., help="Caminho do arquivo de vídeo"),
    config_path: str = typer.Option(
        "config.yaml", help="Caminho do arquivo de configuração"
    ),
):
    """
    Roda só a análise de cena/ações e salva em cache. No modo "fala"
    (padrão), requer que 'python app.py audio' já tenha rodado antes.
    """
    cfg = load_config(config_path)
    base_name = Path(video_path).stem

    scenes = _rodar_cena(video_path, cfg, base_name)

    typer.echo(f"\nConcluído! {len(scenes)} cenas descritas e salvas em cache.")
    typer.echo("Rode 'python app.py processar' quando quiser gerar o roteiro final.")


@app.command()
def processar(
    video_path: str = typer.Argument(..., help="Caminho do arquivo de vídeo"),
    config_path: str = typer.Option(
        "config.yaml", help="Caminho do arquivo de configuração"
    ),
    sem_cena: bool = typer.Option(
        False,
        "--sem-cena",
        help="Pula a análise de ações/cena mesmo se habilitada no config",
    ),
    forcar: bool = typer.Option(
        False, "--forcar", help="Ignora qualquer cache existente e recomputa tudo"
    ),
):
    """
    Roda o pipeline completo (áudio + cena, se habilitada) e gera o
    roteiro final. Reaproveita resultados já salvos em cache pelos
    comandos 'audio' e 'cena', a menos que --forcar seja usado.
    """
    cfg = load_config(config_path)
    base_name = Path(video_path).stem

    # --- Falas ---
    if not forcar and has_cache(base_name, "falas"):
        typer.echo("Falas já processadas anteriormente — reaproveitando cache...")
        segments = load_cache(base_name, "falas")
    else:
        segments = _rodar_audio(video_path, cfg, base_name)

    # --- Cenas ---
    scenes = None
    quer_cena = cfg["scene_analysis"]["enabled"] and not sem_cena
    if quer_cena:
        if not forcar and has_cache(base_name, "cenas"):
            typer.echo("Cenas já processadas anteriormente — reaproveitando cache...")
            scenes = load_cache(base_name, "cenas")
        else:
            scenes = _rodar_cena(video_path, cfg, base_name, segments=segments)
    else:
        typer.echo("Análise de cena desabilitada, pulando...")

    # --- Junta e salva ---
    typer.echo("Montando roteiro final e salvando...")
    timeline = merge_timeline(segments, scenes)
    caminhos = save_outputs(
        timeline,
        output_dir=cfg["output"]["output_dir"],
        base_name=base_name,
        formats=cfg["output"]["formats"],
    )

    typer.echo("\nConcluído! Arquivos gerados:")
    for formato, caminho in caminhos.items():
        typer.echo(f"  - {formato}: {caminho}")


if __name__ == "__main__":
    app()
