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
from core.character_consistency import (
    build_character_sheets,
    apply_character_consistency,
)
from core.merge import merge_timeline
from core.cache import (
    save_cache,
    load_cache,
    has_cache,
)
from core.dataset_builder import (
    list_scenes,
    add_example,
    save_run_to_dataset,
    list_pending_review,
    update_correction,
    export_fewshot_snippet,
    export_finetune_jsonl,
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
    save: bool = typer.Option(
        False,
        "--save",
        help="Ao final, grava todas as cenas processadas no dataset (dataset/dataset.jsonl), pendentes de revisão manual.",
    ),
    fichas: str = typer.Option(
        None,
        "--fichas",
        help="Caminho de um arquivo de fichas de personagens já existente (ex: de outro episódio da mesma série), pra reaproveitar em vez de gerar do zero.",
    ),
    revisar_fichas: bool = typer.Option(
        False,
        "--revisar-fichas",
        help="Abre a ficha de personagens no editor de texto pra você corrigir antes dela ser usada na reescrita das cenas.",
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
    # CONSISTÊNCIA DE PERSONAGENS ENTRE CENAS
    # ---------------------------------------------------------

    if scenes and cfg["scene_analysis"].get("enable_character_consistency", True):

        pular_cache_consistencia = forcar or fichas or revisar_fichas

        if not pular_cache_consistencia and has_cache(base_name, "cenas_consistentes"):

            typer.echo("Cenas com consistência de personagens encontradas no cache.")

            scenes = load_cache(base_name, "cenas_consistentes")

        else:

            contexts = [
                {
                    "start": scene.get("timestamp"),
                    "end": scene.get("end"),
                    "context": scene.get("description", ""),
                    "dialogue": scene.get("dialogue", ""),
                }
                for scene in scenes
            ]

            if fichas:
                typer.echo(f"Carregando ficha de personagens de: {fichas}")
                sheets = Path(fichas).read_text(encoding="utf-8")
            elif not forcar and has_cache(base_name, "fichas_personagens"):
                typer.echo("Fichas de personagens encontradas no cache.")
                sheets = load_cache(base_name, "fichas_personagens")
            else:
                typer.echo(
                    "Montando fichas de personagens (juntando todas as cenas)..."
                )
                sheets = build_character_sheets(contexts, cfg)

            if revisar_fichas:
                typer.echo("Abrindo ficha de personagens pra revisão...")
                edited = typer.edit(sheets)
                if edited is not None and edited.strip():
                    sheets = edited

            save_cache(base_name, "fichas_personagens", sheets)

            fichas_path = (
                Path(cfg["output"]["output_dir"])
                / f"{base_name}_fichas_personagens.txt"
            )
            fichas_path.parent.mkdir(parents=True, exist_ok=True)
            fichas_path.write_text(sheets.strip() + "\n", encoding="utf-8")
            typer.echo(f"Ficha de personagens salva em: {fichas_path}")
            typer.echo(
                "(reaproveite em outro episódio da mesma série com "
                f"--fichas {fichas_path})"
            )

            typer.echo("Reescrevendo cenas com referências consistentes...")
            consistency_results = apply_character_consistency(contexts, sheets, cfg)

            for scene, result in zip(scenes, consistency_results):
                scene["description_original"] = scene.get("description", "")
                scene["description"] = result["enriched_description"]
                scene["needs_visual_review"] = result["needs_visual_review"]
                scene["review_reason"] = result["review_reason"]

            save_cache(base_name, "cenas_consistentes", scenes)

        pendentes = [s for s in scenes if s.get("needs_visual_review")]
        if pendentes:
            review_path = (
                Path(cfg["output"]["output_dir"])
                / f"{base_name}_revisao_visual_pendente.txt"
            )
            review_path.parent.mkdir(parents=True, exist_ok=True)
            with open(review_path, "w", encoding="utf-8") as f:
                for scene in pendentes:
                    f.write(
                        f"[{scene.get('timestamp'):.1f}s - {scene.get('end'):.1f}s] "
                        f"{scene.get('review_reason', '')}\n"
                    )
            typer.echo(
                f"\n{len(pendentes)} cena(s) sinalizada(s) como possivelmente precisando "
                f"de reanálise visual — veja {review_path}"
            )

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

    # ---------------------------------------------------------
    # DATASET (--save)
    # ---------------------------------------------------------

    if save:

        if not scenes:
            typer.echo("\n--save ignorado: não há cenas processadas nesta execução.")
        else:
            added_ids = save_run_to_dataset(base_name, scenes)

            if added_ids:
                typer.echo(
                    f"\n{len(added_ids)} cena(s) gravada(s) no dataset (pendentes de revisão):"
                )
                for example_id in added_ids:
                    typer.echo(f"  - {example_id}")
                typer.echo("Revise com: python app.py dataset-revisar")
            else:
                typer.echo(
                    "\n--save: todas as cenas já estavam no dataset, nada novo gravado."
                )


@app.command("dataset-listar")
def dataset_listar(
    video_path: str = typer.Argument(
        ..., help="Caminho do vídeo (mesmo usado antes, só pra achar o cache)"
    ),
):
    """Lista as cenas já processadas (cache) com índice, diálogo e a descrição que o modelo gerou."""
    base_name = Path(video_path).stem
    scenes = list_scenes(base_name)

    for i, scene in enumerate(scenes):
        typer.echo(
            f"\n--- Cena {i} ({scene.get('timestamp'):.1f}s - {scene.get('end'):.1f}s) ---"
        )
        typer.echo(f"Diálogo:\n{scene.get('dialogue', '')}")
        typer.echo(f"\nModelo gerou:\n{scene.get('description', '')}")


@app.command("dataset-adicionar")
def dataset_adicionar(
    video_path: str = typer.Argument(..., help="Caminho do vídeo"),
    indice: int = typer.Argument(..., help="Índice da cena (veja com dataset-listar)"),
    arquivo_correcao: str = typer.Option(
        None,
        "--arquivo",
        help="Caminho de um .txt com a versão corrigida. Se omitido, abre um editor de texto no terminal.",
    ),
):
    """
    Adiciona um exemplo ao dataset: mostra o que o modelo gerou pra
    essa cena e pede a versão corrigida (via editor ou arquivo).
    """
    base_name = Path(video_path).stem
    scenes = list_scenes(base_name)

    if indice < 0 or indice >= len(scenes):
        typer.echo(f"Índice inválido. Há {len(scenes)} cenas (0 a {len(scenes) - 1}).")
        raise typer.Exit(1)

    scene = scenes[indice]

    typer.echo(f"Diálogo:\n{scene.get('dialogue', '')}\n")
    typer.echo(f"Modelo gerou:\n{scene.get('description', '')}\n")

    if arquivo_correcao:
        corrected = Path(arquivo_correcao).read_text(encoding="utf-8")
    else:
        corrected = typer.edit(scene.get("description", "")) or ""

    if not corrected.strip():
        typer.echo("Correção vazia, nada foi salvo.")
        raise typer.Exit(1)

    example_id = add_example(base_name, indice, corrected)
    typer.echo(f"\nExemplo salvo: {example_id}")


@app.command("dataset-revisar")
def dataset_revisar():
    """
    Percorre os exemplos gravados automaticamente (via --save) que
    ainda não têm correção humana, mostrando o que o modelo gerou e
    pedindo a versão corrigida (abre o editor de texto do terminal).
    """
    pendentes = list_pending_review()

    if not pendentes:
        typer.echo("Nenhum exemplo pendente de revisão.")
        raise typer.Exit()

    typer.echo(f"{len(pendentes)} exemplo(s) pendente(s).\n")

    for ex in pendentes:
        typer.echo(f"--- {ex['id']} ---")
        typer.echo(f"Diálogo:\n{ex.get('dialogue', '')}\n")
        typer.echo(f"Modelo gerou:\n{ex.get('model_output', '')}\n")

        corrected = typer.edit(ex.get("model_output", "") or "")

        if corrected is None:
            typer.echo("Pulado (nada foi salvo pra este exemplo).\n")
            continue

        if not corrected.strip():
            typer.echo("Correção vazia, pulado.\n")
            continue

        update_correction(ex["id"], corrected)
        typer.echo(f"Salvo: {ex['id']}\n")


@app.command("dataset-exportar")
def dataset_exportar(
    formato: str = typer.Option(
        "fewshot",
        help="'fewshot' (trecho pra colar no prompt) ou 'finetune' (jsonl pra treino)",
    ),
    n: int = typer.Option(
        3, help="Quantos exemplos incluir (só usado no formato fewshot)"
    ),
):
    """Exporta o dataset acumulado no formato pedido."""
    if formato == "fewshot":
        snippet = export_fewshot_snippet(n=n)
        if not snippet:
            typer.echo(
                "Dataset vazio — adicione exemplos com 'dataset-adicionar' primeiro."
            )
            raise typer.Exit(1)
        typer.echo(snippet)
    elif formato == "finetune":
        path = export_finetune_jsonl()
        typer.echo(f"Exportado para: {path}")
    else:
        typer.echo("Formato inválido. Use 'fewshot' ou 'finetune'.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
