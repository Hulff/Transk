# app.py

import os
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from core.ffmpeg_utils import ensure_ffmpeg_in_path

ensure_ffmpeg_in_path()

from core.extract_audio import extract_audio
from core.transcribe import transcribe_and_diarize
from core.speaker_mapping import map_speakers_to_characters
from core.scene_analysis import (
    group_into_scenes,
    describe_scenes,
    DEFAULT_QUESTION,
)

from core.context_analysis import (
    analyze_full_context,
    validate_analysis,
    save_analysis,
)
from core.merge import merge_timeline
from core.cache import (
    save_cache,
    load_cache,
    has_cache,
)
from output.formatter import save_outputs

load_dotenv()

app = typer.Typer(help="Transcritor de episódios: falas + ações + ambiente")


def load_config(
    config_path: str = "config.yaml",
) -> dict:

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        cfg = yaml.safe_load(file)

    env_token = os.getenv("HF_TOKEN")

    if env_token:
        cfg["diarization"]["hf_token"] = env_token

    elif not cfg["diarization"].get("hf_token"):

        raise RuntimeError(
            "Token do Hugging Face não encontrado. "
            "Crie um arquivo .env com HF_TOKEN=seu_token."
        )

    return cfg


def _rodar_audio(
    video_path: str,
    cfg: dict,
    base_name: str,
) -> list[dict]:

    typer.echo("Extraindo áudio...")

    audio_path = extract_audio(video_path)

    typer.echo("Transcrevendo e identificando falantes...")

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

    typer.echo("Reconhecendo personagens por voz...")

    segments = map_speakers_to_characters(
        segments,
        audio_path=audio_path,
        voice_profiles_dir=cfg["speaker_mapping"]["voice_profiles_dir"],
        hf_token=cfg["diarization"]["hf_token"],
        similarity_threshold=cfg["speaker_mapping"]["similarity_threshold"],
    )

    save_cache(
        base_name,
        "falas",
        segments,
    )

    return segments


def _rodar_cena(
    video_path: str,
    cfg: dict,
    base_name: str,
    segments: list[dict] | None = None,
) -> list[dict]:

    scene_cfg = cfg["scene_analysis"]

    if segments is None:
        segments = load_cache(
            base_name,
            "falas",
        )

    if not segments:
        raise RuntimeError(
            "A análise visual precisa das falas. "
            "Rode 'python app.py audio <video>' primeiro."
        )

    gap = scene_cfg.get(
        "gap_threshold_seconds",
        3.0,
    )

    grouped = group_into_scenes(
        segments,
        gap_threshold_seconds=gap,
    )

    typer.echo(f"{len(segments)} falas agrupadas em " f"{len(grouped)} cenas.")

    scenes = describe_scenes(
        grouped,
        video_path,
        model_name=scene_cfg.get(
            "model",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ),
        frames_per_scene=scene_cfg.get(
            "frames_per_scene",
            12,
        ),
        padding_seconds=scene_cfg.get(
            "padding_seconds",
            0.5,
        ),
        question=scene_cfg.get(
            "question",
        )
        or DEFAULT_QUESTION,
        language=scene_cfg.get(
            "language",
            "pt-BR",
        ),
        max_new_tokens=scene_cfg.get(
            "max_new_tokens",
            400,
        ),
        repetition_penalty=scene_cfg.get(
            "repetition_penalty",
            1.15,
        ),
        no_repeat_ngram_size=scene_cfg.get(
            "no_repeat_ngram_size",
            3,
        ),
        do_sample=scene_cfg.get(
            "do_sample",
            False,
        ),
        load_in_4bit=scene_cfg.get(
            "load_in_4bit",
            False,
        ),
    )

    save_cache(
        base_name,
        "cenas",
        scenes,
    )

    return scenes


@app.command()
def audio(
    video_path: str = typer.Argument(
        ...,
        help="Caminho do vídeo",
    ),
    config_path: str = typer.Option(
        "config.yaml",
    ),
):

    cfg = load_config(config_path)

    base_name = Path(video_path).stem

    segments = _rodar_audio(
        video_path,
        cfg,
        base_name,
    )

    typer.echo(f"\nConcluído! " f"{len(segments)} falas identificadas.")


@app.command()
def cena(
    video_path: str = typer.Argument(
        ...,
        help="Caminho do vídeo",
    ),
    config_path: str = typer.Option(
        "config.yaml",
    ),
):

    cfg = load_config(config_path)

    base_name = Path(video_path).stem

    scenes = _rodar_cena(
        video_path,
        cfg,
        base_name,
    )

    typer.echo(f"\nConcluído! " f"{len(scenes)} cenas analisadas.")


@app.command()
def processar(
    video_path: str = typer.Argument(
        ...,
        help="Caminho do vídeo",
    ),
    config_path: str = typer.Option(
        "config.yaml",
    ),
    sem_cena: bool = typer.Option(
        False,
        "--sem-cena",
    ),
    forcar: bool = typer.Option(
        False,
        "--forcar",
    ),
):

    cfg = load_config(config_path)

    base_name = Path(video_path).stem

    # ---------------------------------------------------------
    # ÁUDIO
    # ---------------------------------------------------------

    if not forcar and has_cache(base_name, "falas"):

        typer.echo("Falas encontradas no cache.")

        segments = load_cache(
            base_name,
            "falas",
        )

    else:

        segments = _rodar_audio(
            video_path,
            cfg,
            base_name,
        )

    # ---------------------------------------------------------
    # CENAS
    # ---------------------------------------------------------

    scenes = None

    if cfg["scene_analysis"]["enabled"] and not sem_cena:

        if not forcar and has_cache(
            base_name,
            "cenas",
        ):

            typer.echo("Cenas encontradas no cache.")

            scenes = load_cache(
                base_name,
                "cenas",
            )

        else:

            scenes = _rodar_cena(
                video_path,
                cfg,
                base_name,
                segments,
            )

    else:

        typer.echo("Análise visual desabilitada.")

    # ---------------------------------------------------------
    # TIMELINE FINAL
    # ---------------------------------------------------------

    typer.echo("Montando roteiro audiovisual...")

    timeline = merge_timeline(
        segments,
        scenes,
    )

    # ---------------------------------------------------------
    # ANÁLISE FINAL DO VÍDEO
    # ---------------------------------------------------------

    analysis_path = None

    if scenes:
        if not forcar and has_cache(
            base_name,
            "analise_validada",
        ):
            typer.echo("Análise validada encontrada no cache.")

            analysis = load_cache(
                base_name,
                "analise_validada",
            )

        else:
            if not forcar and has_cache(base_name, "analise_rascunho"):
                typer.echo("Rascunho da análise encontrado no cache.")
                draft = load_cache(base_name, "analise_rascunho")
            else:
                typer.echo("Gerando análise consolidada do vídeo...")

                contexts = [
                    {
                        "start": scene.get("timestamp"),
                        "end": scene.get("end"),
                        "context": scene.get("description", ""),
                        "dialogue": scene.get("dialogue", ""),
                    }
                    for scene in scenes
                ]

                draft = analyze_full_context(
                    contexts=contexts,
                    config=cfg,
                )

                save_cache(base_name, "analise_rascunho", draft)

            typer.echo("Validando e corrigindo a análise...")

            contexts = [
                {
                    "start": scene.get("timestamp"),
                    "end": scene.get("end"),
                    "context": scene.get("description", ""),
                    "dialogue": scene.get("dialogue", ""),
                }
                for scene in scenes
            ]

            analysis = validate_analysis(
                draft_analysis=draft,
                contexts=contexts,
                config=cfg,
            )

            save_cache(
                base_name,
                "analise_validada",
                analysis,
            )

        analysis_path = Path(cfg["output"]["output_dir"]) / f"{base_name}_analise.txt"

        save_analysis(
            analysis=analysis,
            path=str(analysis_path),
        )

        typer.echo(f"Análise consolidada exportada: {analysis_path}")

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

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
