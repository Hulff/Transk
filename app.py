import os
from pathlib import Path
from typing import Any

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

from core.identity_resolver import (
    resolve_scene_characters,
    format_character_registry,
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


app = typer.Typer(
    help="Transcritor de episódios: falas + ações + ambiente"
)


# ============================================================
# CONFIG
# ============================================================


def load_config(
    config_path: str = "config.yaml",
) -> dict:

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        cfg = yaml.safe_load(file)

    if cfg is None:
        cfg = {}

    cfg.setdefault(
        "whisper",
        {},
    )

    cfg.setdefault(
        "diarization",
        {},
    )

    cfg.setdefault(
        "speaker_mapping",
        {},
    )

    cfg.setdefault(
        "identity_resolution",
        {},
    )

    cfg.setdefault(
        "scene_analysis",
        {},
    )

    cfg.setdefault(
        "output",
        {},
    )

    env_token = os.getenv(
        "HF_TOKEN"
    )

    if env_token:
        cfg["diarization"][
            "hf_token"
        ] = env_token

    elif not cfg["diarization"].get(
        "hf_token"
    ):
        raise RuntimeError(
            "Token do Hugging Face não encontrado. "
            "Crie um arquivo .env com HF_TOKEN=seu_token."
        )

    # Defaults para compatibilidade
    cfg["speaker_mapping"].setdefault(
        "similarity_threshold",
        0.75,
    )

    cfg["speaker_mapping"].setdefault(
        "margin_threshold",
        0.08,
    )

    cfg["speaker_mapping"].setdefault(
        "voice_profiles_dir",
        "models/voice_profiles",
    )

    identity_cfg = cfg[
        "identity_resolution"
    ]

    identity_cfg.setdefault(
        "enabled",
        True,
    )

    identity_cfg.setdefault(
        "visual_weight",
        0.40,
    )

    identity_cfg.setdefault(
        "voice_weight",
        0.35,
    )

    identity_cfg.setdefault(
        "temporal_weight",
        0.15,
    )

    identity_cfg.setdefault(
        "context_weight",
        0.10,
    )

    identity_cfg.setdefault(
        "confirmed_threshold",
        0.80,
    )

    identity_cfg.setdefault(
        "probable_threshold",
        0.65,
    )

    cfg["scene_analysis"].setdefault(
        "enabled",
        True,
    )

    cfg["scene_analysis"].setdefault(
        "enable_character_consistency",
        True,
    )

    cfg["output"].setdefault(
        "output_dir",
        "resultados",
    )

    cfg["output"].setdefault(
        "formats",
        [
            "txt",
            "json",
            "srt",
        ],
    )

    return cfg


# ============================================================
# HELPERS
# ============================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def _get_speaker_id(
    segment: dict,
) -> str | None:

    return (
        segment.get(
            "speaker_id"
        )
        or segment.get(
            "speaker"
        )
    )


def _build_scene_contexts(
    scenes: list[dict],
    registry: dict | None = None,
) -> list[dict]:
    """
    Constrói o contexto usado pelos módulos de consistência
    e análise final.

    Além do texto antigo, agora inclui:

        visual_descriptors
        scene_events
        resolved_characters
        speaker_character_mapping
        character_registry
    """

    contexts = []

    for index, scene in enumerate(
        scenes,
        start=1,
    ):

        context = {
            "scene_id": scene.get(
                "scene_id",
                index,
            ),
            "start": scene.get(
                "timestamp",
                scene.get(
                    "start",
                    0,
                ),
            ),
            "end": scene.get(
                "end",
                0,
            ),
            "context": scene.get(
                "description",
                "",
            ),
            "dialogue": scene.get(
                "dialogue",
                "",
            ),
            "visual_descriptors": scene.get(
                "visual_descriptors",
                [],
            ),
            "scene_events": scene.get(
                "scene_events",
                {},
            ),
            "resolved_characters": scene.get(
                "resolved_characters",
                [],
            ),
            "speaker_character_mapping": scene.get(
                "speaker_character_mapping",
                {},
            ),
        }

        if registry is not None:
            context[
                "character_registry"
            ] = registry

        contexts.append(
            context
        )

    return contexts


def _apply_identity_to_segments(
    segments: list[dict],
    scenes: list[dict],
) -> list[dict]:
    """
    Propaga a resolução de identidade para os segmentos de fala.

    O speaker original permanece:

        speaker_id = SPEAKER_00

    E adicionamos:

        character_id
        character_name

    Só propagamos identidades que não estejam AMBÍGUAS.
    """

    speaker_to_character = {}

    for scene in scenes:

        mapping = scene.get(
            "speaker_character_mapping",
            {},
        )

        if not isinstance(
            mapping,
            dict,
        ):
            continue

        resolved_characters = scene.get(
            "resolved_characters",
            [],
        )

        status_by_character = {}

        for resolved in resolved_characters:

            character_id = resolved.get(
                "character_id"
            )

            status = resolved.get(
                "status",
                "AMBÍGUO",
            )

            if character_id:
                status_by_character[
                    character_id
                ] = status

        for speaker_id, character_id in (
            mapping.items()
        ):

            if not speaker_id:
                continue

            if not character_id:
                continue

            status = status_by_character.get(
                character_id,
                "AMBÍGUO",
            )

            # Não força identidade ambígua para as falas.
            if status == "AMBÍGUO":
                continue

            speaker_to_character[
                speaker_id
            ] = character_id

    # ---------------------------------------------------------
    # Nome global por character_id
    # ---------------------------------------------------------

    character_names = {}

    for scene in scenes:

        for resolved in scene.get(
            "resolved_characters",
            [],
        ):

            character_id = resolved.get(
                "character_id"
            )

            name = resolved.get(
                "name"
            )

            if (
                character_id
                and name
            ):
                character_names[
                    character_id
                ] = name

    # ---------------------------------------------------------
    # Aplicação
    # ---------------------------------------------------------

    for segment in segments:

        speaker_id = _get_speaker_id(
            segment
        )

        if not speaker_id:
            continue

        character_id = speaker_to_character.get(
            speaker_id
        )

        if not character_id:
            continue

        segment[
            "character_id"
        ] = character_id

        segment[
            "character_name"
        ] = character_names.get(
            character_id
        )

    return segments


def _print_identity_summary(
    registry: dict,
) -> None:
    """
    Exibe no terminal o resultado da resolução.
    """

    if not registry:
        typer.echo(
            "Nenhum personagem global foi resolvido."
        )
        return

    typer.echo(
        "\nIdentidades resolvidas:"
    )

    for character_id, character in (
        registry.items()
    ):

        name = (
            character.get(
                "name"
            )
            or "SEM_NOME"
        )

        visual = character.get(
            "visual_confidence"
        )

        voice = character.get(
            "voice_confidence"
        )

        identity = character.get(
            "identity_confidence"
        )

        speakers = character.get(
            "speaker_ids",
            [],
        )

        typer.echo(
            f"  {character_id} -> {name}"
        )

        typer.echo(
            f"      speakers: {speakers}"
        )

        typer.echo(
            f"      visual: "
            f"{visual if visual is not None else '-'}"
        )

        typer.echo(
            f"      voice: "
            f"{voice if voice is not None else '-'}"
        )

        typer.echo(
            f"      identity: "
            f"{identity if identity is not None else '-'}"
        )


def _resolver_identidades(
    scenes: list[dict],
    cfg: dict,
    base_name: str,
    forcar: bool = False,
) -> tuple[
    list[dict],
    dict,
]:
    """
    Resolve a identidade global dos personagens.

    Fluxo:

        Qwen
          ↓
        visual_descriptors
          ↓
        identity_resolver
          ↓
        character_id
          ↓
        character_registry
    """

    identity_cfg = cfg.get(
        "identity_resolution",
        {},
    )

    if not identity_cfg.get(
        "enabled",
        True,
    ):
        typer.echo(
            "Resolução global de identidade desabilitada."
        )

        return scenes, {}

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    if (
        not forcar
        and has_cache(
            base_name,
            "identity_resolution",
        )
        and has_cache(
            base_name,
            "character_registry",
        )
    ):

        typer.echo(
            "Resolução de identidade encontrada no cache."
        )

        cached_scenes = load_cache(
            base_name,
            "identity_resolution",
        )

        registry = load_cache(
            base_name,
            "character_registry",
        )

        if cached_scenes is not None:

            return (
                cached_scenes,
                registry or {},
            )

    # ---------------------------------------------------------
    # Registry anterior
    # ---------------------------------------------------------

    previous_registry = None

    if (
        not forcar
        and has_cache(
            base_name,
            "character_registry",
        )
    ):
        previous_registry = load_cache(
            base_name,
            "character_registry",
        )

    # ---------------------------------------------------------
    # Resolver
    # ---------------------------------------------------------

    typer.echo(
        "Resolvendo identidade global dos personagens..."
    )

    resolved_scenes, registry = (
        resolve_scene_characters(
            scenes,
            previous_registry=previous_registry,
            visual_weight=identity_cfg.get(
                "visual_weight",
                0.40,
            ),
            voice_weight=identity_cfg.get(
                "voice_weight",
                0.35,
            ),
            temporal_weight=identity_cfg.get(
                "temporal_weight",
                0.15,
            ),
            context_weight=identity_cfg.get(
                "context_weight",
                0.10,
            ),
            confirmed_threshold=identity_cfg.get(
                "confirmed_threshold",
                0.80,
            ),
            probable_threshold=identity_cfg.get(
                "probable_threshold",
                0.65,
            ),
        )
    )

    # ---------------------------------------------------------
    # Salva
    # ---------------------------------------------------------

    save_cache(
        base_name,
        "identity_resolution",
        resolved_scenes,
    )

    save_cache(
        base_name,
        "character_registry",
        registry,
    )

    # ---------------------------------------------------------
    # Ficha legível
    # ---------------------------------------------------------

    registry_text = (
        format_character_registry(
            registry
        )
    )

    save_cache(
        base_name,
        "fichas_personagens",
        registry_text,
    )

    _print_identity_summary(
        registry
    )

    return (
        resolved_scenes,
        registry,
    )


# ============================================================
# ÁUDIO
# ============================================================


def _rodar_audio(
    video_path: str,
    cfg: dict,
    base_name: str,
) -> list[dict]:

    typer.echo(
        "Extraindo áudio..."
    )

    audio_path = extract_audio(
        video_path
    )

    typer.echo(
        "Transcrevendo e identificando falantes..."
    )

    segments = transcribe_and_diarize(
        audio_path,
        model_name=cfg[
            "whisper"
        ]["model"],
        language=cfg[
            "whisper"
        ]["language"],
        device=cfg[
            "whisper"
        ]["device"],
        compute_type=cfg[
            "whisper"
        ]["compute_type"],
        hf_token=cfg[
            "diarization"
        ]["hf_token"],
        min_speakers=cfg[
            "diarization"
        ].get(
            "min_speakers"
        ),
        max_speakers=cfg[
            "diarization"
        ].get(
            "max_speakers"
        ),
    )

    typer.echo(
        "Reconhecendo personagens por voz..."
    )

    speaker_cfg = cfg[
        "speaker_mapping"
    ]

    segments = map_speakers_to_characters(
        segments,
        audio_path=audio_path,
        voice_profiles_dir=speaker_cfg[
            "voice_profiles_dir"
        ],
        hf_token=cfg[
            "diarization"
        ]["hf_token"],
        similarity_threshold=speaker_cfg.get(
            "similarity_threshold",
            0.75,
        ),
        margin_threshold=speaker_cfg.get(
            "margin_threshold",
            0.08,
        ),
        device=cfg[
            "whisper"
        ].get(
            "device",
            "cpu",
        ),
    )

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    save_cache(
        base_name,
        "falas",
        segments,
    )

    save_cache(
        base_name,
        "speaker_mapping",
        segments,
    )

    return segments


# ============================================================
# CENAS
# ============================================================


def _rodar_cena(
    video_path: str,
    cfg: dict,
    base_name: str,
    segments: list[dict] | None = None,
) -> list[dict]:

    scene_cfg = cfg[
        "scene_analysis"
    ]

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

    typer.echo(
        f"{len(segments)} falas agrupadas em "
        f"{len(grouped)} cenas."
    )

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
            None,
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

    # ---------------------------------------------------------
    # Adiciona IDs estáveis das cenas
    # ---------------------------------------------------------

    for index, scene in enumerate(
        scenes,
        start=1,
    ):

        scene[
            "scene_id"
        ] = index

    save_cache(
        base_name,
        "cenas",
        scenes,
    )

    return scenes


# ============================================================
# COMANDO: AUDIO
# ============================================================


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

    cfg = load_config(
        config_path
    )

    base_name = Path(
        video_path
    ).stem

    segments = _rodar_audio(
        video_path,
        cfg,
        base_name,
    )

    typer.echo(
        "\nConcluído! "
        f"{len(segments)} falas identificadas."
    )


# ============================================================
# COMANDO: CENA
# ============================================================


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

    cfg = load_config(
        config_path
    )

    base_name = Path(
        video_path
    ).stem

    scenes = _rodar_cena(
        video_path,
        cfg,
        base_name,
    )

    typer.echo(
        "\nConcluído! "
        f"{len(scenes)} cenas analisadas."
    )


# ============================================================
# COMANDO: PROCESSAR
# ============================================================


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
        help=(
            "Ao final, grava todas as cenas processadas "
            "no dataset (dataset/dataset.jsonl), "
            "pendentes de revisão manual."
        ),
    ),
    fichas: str = typer.Option(
        None,
        "--fichas",
        help=(
            "Caminho de um arquivo de fichas de personagens "
            "já existente."
        ),
    ),
    revisar_fichas: bool = typer.Option(
        False,
        "--revisar-fichas",
        help=(
            "Abre a ficha de personagens no editor de texto "
            "pra você corrigir antes dela ser usada."
        ),
    ),
):

    cfg = load_config(
        config_path
    )

    base_name = Path(
        video_path
    ).stem

    # ========================================================
    # ÁUDIO
    # ========================================================

    if (
        not forcar
        and has_cache(
            base_name,
            "falas",
        )
    ):

        typer.echo(
            "Falas encontradas no cache."
        )

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

    # ========================================================
    # CENAS
    # ========================================================

    scenes = None

    if (
        cfg[
            "scene_analysis"
        ]["enabled"]
        and not sem_cena
    ):

        if (
            not forcar
            and has_cache(
                base_name,
                "cenas",
            )
        ):

            typer.echo(
                "Cenas encontradas no cache."
            )

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

        typer.echo(
            "Análise visual desabilitada."
        )

    # ========================================================
    # RESOLUÇÃO DE IDENTIDADE
    # ========================================================

    registry = {}

    if scenes:

        (
            scenes,
            registry,
        ) = _resolver_identidades(
            scenes,
            cfg,
            base_name,
            forcar=forcar,
        )

        # ----------------------------------------------------
        # Propaga character_id para as falas
        # ----------------------------------------------------

        segments = (
            _apply_identity_to_segments(
                segments,
                scenes,
            )
        )

        save_cache(
            base_name,
            "falas",
            segments,
        )

    # ========================================================
    # CONSISTÊNCIA DE PERSONAGENS ENTRE CENAS
    # ========================================================

    if (
        scenes
        and cfg[
            "scene_analysis"
        ].get(
            "enable_character_consistency",
            True,
        )
    ):

        pular_cache_consistencia = (
            forcar
            or fichas
            or revisar_fichas
        )

        if (
            not pular_cache_consistencia
            and has_cache(
                base_name,
                "cenas_consistentes",
            )
        ):

            typer.echo(
                "Cenas com consistência de personagens "
                "encontradas no cache."
            )

            scenes = load_cache(
                base_name,
                "cenas_consistentes",
            )

        else:

            contexts = _build_scene_contexts(
                scenes,
                registry,
            )

            # ------------------------------------------------
            # FICHAS
            # ------------------------------------------------

            if fichas:

                typer.echo(
                    "Carregando ficha de personagens de: "
                    f"{fichas}"
                )

                sheets = Path(
                    fichas
                ).read_text(
                    encoding="utf-8"
                )

            elif (
                not forcar
                and has_cache(
                    base_name,
                    "fichas_personagens",
                )
            ):

                typer.echo(
                    "Fichas de personagens encontradas no cache."
                )

                sheets = load_cache(
                    base_name,
                    "fichas_personagens",
                )

            else:

                typer.echo(
                    "Montando fichas de personagens "
                    "(juntando todas as cenas)..."
                )

                sheets = build_character_sheets(
                    contexts,
                    cfg,
                )

            # ------------------------------------------------
            # REVISÃO MANUAL
            # ------------------------------------------------

            if revisar_fichas:

                typer.echo(
                    "Abrindo ficha de personagens pra revisão..."
                )

                edited = typer.edit(
                    sheets
                )

                if (
                    edited is not None
                    and edited.strip()
                ):
                    sheets = edited

            # ------------------------------------------------
            # CACHE DA FICHA
            # ------------------------------------------------

            save_cache(
                base_name,
                "fichas_personagens",
                sheets,
            )

            fichas_path = (
                Path(
                    cfg[
                        "output"
                    ][
                        "output_dir"
                    ]
                )
                / (
                    f"{base_name}"
                    "_fichas_personagens.txt"
                )
            )

            fichas_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fichas_path.write_text(
                sheets.strip()
                + "\n",
                encoding="utf-8",
            )

            typer.echo(
                "Ficha de personagens salva em: "
                f"{fichas_path}"
            )

            typer.echo(
                "(reaproveite em outro episódio da mesma "
                "série com "
                f"--fichas {fichas_path})"
            )

            # ------------------------------------------------
            # REESCRITA
            # ------------------------------------------------

            typer.echo(
                "Reescrevendo cenas com referências consistentes..."
            )

            consistency_results = (
                apply_character_consistency(
                    contexts,
                    sheets,
                    cfg,
                )
            )

            for (
                scene,
                result,
            ) in zip(
                scenes,
                consistency_results,
            ):

                scene[
                    "description_original"
                ] = scene.get(
                    "description",
                    "",
                )

                scene[
                    "description"
                ] = result[
                    "enriched_description"
                ]

                scene[
                    "needs_visual_review"
                ] = result[
                    "needs_visual_review"
                ]

                scene[
                    "review_reason"
                ] = result[
                    "review_reason"
                ]

            save_cache(
                base_name,
                "cenas_consistentes",
                scenes,
            )

        # ----------------------------------------------------
        # REVISÕES VISUAIS PENDENTES
        # ----------------------------------------------------

        pendentes = [
            scene
            for scene in scenes
            if scene.get(
                "needs_visual_review"
            )
        ]

        if pendentes:

            review_path = (
                Path(
                    cfg[
                        "output"
                    ][
                        "output_dir"
                    ]
                )
                / (
                    f"{base_name}"
                    "_revisao_visual_pendente.txt"
                )
            )

            review_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                review_path,
                "w",
                encoding="utf-8",
            ) as file:

                for scene in pendentes:

                    start = _safe_float(
                        scene.get(
                            "timestamp",
                            0,
                        )
                    )

                    end = _safe_float(
                        scene.get(
                            "end",
                            0,
                        )
                    )

                    file.write(
                        f"[{start:.1f}s - "
                        f"{end:.1f}s] "
                        f"{scene.get('review_reason', '')}\n"
                    )

            typer.echo(
                f"\n{len(pendentes)} cena(s) "
                "sinalizada(s) como possivelmente "
                "precisando de reanálise visual — "
                f"veja {review_path}"
            )

    # ========================================================
    # TIMELINE FINAL
    # ========================================================

    typer.echo(
        "Montando roteiro audiovisual..."
    )

    timeline = merge_timeline(
        segments,
        scenes,
    )

    # ========================================================
    # ANÁLISE FINAL DO VÍDEO
    # ========================================================

    analysis_path = None

    if scenes:

        if (
            not forcar
            and has_cache(
                base_name,
                "analise_validada",
            )
        ):

            typer.echo(
                "Análise validada encontrada no cache."
            )

            analysis = load_cache(
                base_name,
                "analise_validada",
            )

        else:

            if (
                not forcar
                and has_cache(
                    base_name,
                    "analise_rascunho",
                )
            ):

                typer.echo(
                    "Rascunho da análise encontrado no cache."
                )

                draft = load_cache(
                    base_name,
                    "analise_rascunho",
                )

            else:

                typer.echo(
                    "Gerando análise consolidada do vídeo..."
                )

                contexts = _build_scene_contexts(
                    scenes,
                    registry,
                )

                draft = analyze_full_context(
                    contexts=contexts,
                    config=cfg,
                )

                save_cache(
                    base_name,
                    "analise_rascunho",
                    draft,
                )

            typer.echo(
                "Validando e corrigindo a análise..."
            )

            contexts = _build_scene_contexts(
                scenes,
                registry,
            )

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

        # ----------------------------------------------------
        # Salva análise
        # ----------------------------------------------------

        analysis_path = (
            Path(
                cfg[
                    "output"
                ][
                    "output_dir"
                ]
            )
            / f"{base_name}_analise.txt"
        )

        save_analysis(
            analysis=analysis,
            path=str(
                analysis_path
            ),
        )

        typer.echo(
            "Análise consolidada exportada: "
            f"{analysis_path}"
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    caminhos = save_outputs(
        timeline,
        output_dir=cfg[
            "output"
        ][
            "output_dir"
        ],
        base_name=base_name,
        formats=cfg[
            "output"
        ][
            "formats"
        ],
    )

    typer.echo(
        "\nConcluído! Arquivos gerados:"
    )

    for (
        formato,
        caminho,
    ) in caminhos.items():

        typer.echo(
            f"  - {formato}: {caminho}"
        )

    # ========================================================
    # DATASET (--save)
    # ========================================================

    if save:

        if not scenes:

            typer.echo(
                "\n--save ignorado: "
                "não há cenas processadas nesta execução."
            )

        else:

            added_ids = (
                save_run_to_dataset(
                    base_name,
                    scenes,
                )
            )

            if added_ids:

                typer.echo(
                    f"\n{len(added_ids)} cena(s) "
                    "gravada(s) no dataset "
                    "(pendentes de revisão):"
                )

                for example_id in added_ids:

                    typer.echo(
                        f"  - {example_id}"
                    )

                typer.echo(
                    "Revise com: "
                    "python app.py dataset-revisar"
                )

            else:

                typer.echo(
                    "\n--save: todas as cenas já estavam "
                    "no dataset, nada novo gravado."
                )


# ============================================================
# DATASET - LISTAR
# ============================================================


@app.command(
    "dataset-listar"
)
def dataset_listar(
    video_path: str = typer.Argument(
        ...,
        help=(
            "Caminho do vídeo "
            "(mesmo usado antes, só pra achar o cache)"
        ),
    ),
):
    """
    Lista as cenas já processadas no cache.
    """

    base_name = Path(
        video_path
    ).stem

    scenes = list_scenes(
        base_name
    )

    for index, scene in enumerate(
        scenes
    ):

        start = _safe_float(
            scene.get(
                "timestamp",
                0,
            )
        )

        end = _safe_float(
            scene.get(
                "end",
                0,
            )
        )

        typer.echo(
            f"\n--- Cena {index} "
            f"({start:.1f}s - {end:.1f}s) ---"
        )

        typer.echo(
            "Diálogo:"
        )

        typer.echo(
            scene.get(
                "dialogue",
                "",
            )
        )

        typer.echo(
            "\nModelo gerou:"
        )

        typer.echo(
            scene.get(
                "description",
                "",
            )
        )

        resolved = scene.get(
            "resolved_characters",
            [],
        )

        if resolved:

            typer.echo(
                "\nIdentidades:"
            )

            for character in resolved:

                typer.echo(
                    "  "
                    f"{character.get('character_id')} "
                    "-> "
                    f"{character.get('name') or 'SEM_NOME'} "
                    f"[{character.get('status', 'AMBÍGUO')}] "
                    f"score="
                    f"{character.get('identity_confidence', 0):.3f}"
                )


# ============================================================
# DATASET - ADICIONAR
# ============================================================


@app.command(
    "dataset-adicionar"
)
def dataset_adicionar(
    video_path: str = typer.Argument(
        ...,
        help="Caminho do vídeo",
    ),
    indice: int = typer.Argument(
        ...,
        help=(
            "Índice da cena "
            "(veja com dataset-listar)"
        ),
    ),
    arquivo_correcao: str = typer.Option(
        None,
        "--arquivo",
        help=(
            "Caminho de um .txt com a versão corrigida. "
            "Se omitido, abre um editor."
        ),
    ),
):
    """
    Adiciona uma correção manual ao dataset.
    """

    base_name = Path(
        video_path
    ).stem

    scenes = list_scenes(
        base_name
    )

    if (
        indice < 0
        or indice >= len(scenes)
    ):

        typer.echo(
            "Índice inválido. "
            f"Há {len(scenes)} cenas "
            f"(0 a {len(scenes) - 1})."
        )

        raise typer.Exit(
            1
        )

    scene = scenes[
        indice
    ]

    typer.echo(
        "Diálogo:"
    )

    typer.echo(
        scene.get(
            "dialogue",
            "",
        )
    )

    typer.echo(
        "\nModelo gerou:"
    )

    typer.echo(
        scene.get(
            "description",
            "",
        )
    )

    if arquivo_correcao:

        corrected = Path(
            arquivo_correcao
        ).read_text(
            encoding="utf-8"
        )

    else:

        corrected = (
            typer.edit(
                scene.get(
                    "description",
                    "",
                )
            )
            or ""
        )

    if not corrected.strip():

        typer.echo(
            "Correção vazia, nada foi salvo."
        )

        raise typer.Exit(
            1
        )

    example_id = add_example(
        base_name,
        indice,
        corrected,
    )

    typer.echo(
        f"\nExemplo salvo: "
        f"{example_id}"
    )


# ============================================================
# DATASET - REVISAR
# ============================================================


@app.command(
    "dataset-revisar"
)
def dataset_revisar():
    """
    Percorre exemplos que ainda não têm correção humana.
    """

    pendentes = (
        list_pending_review()
    )

    if not pendentes:

        typer.echo(
            "Nenhum exemplo pendente de revisão."
        )

        raise typer.Exit()

    typer.echo(
        f"{len(pendentes)} exemplo(s) "
        "pendente(s).\n"
    )

    for ex in pendentes:

        typer.echo(
            f"--- {ex['id']} ---"
        )

        typer.echo(
            "Diálogo:"
        )

        typer.echo(
            ex.get(
                "dialogue",
                "",
            )
        )

        typer.echo(
            "\nModelo gerou:"
        )

        typer.echo(
            ex.get(
                "model_output",
                "",
            )
        )

        corrected = typer.edit(
            ex.get(
                "model_output",
                ""
            )
            or ""
        )

        if corrected is None:

            typer.echo(
                "Pulado "
                "(nada foi salvo pra este exemplo).\n"
            )

            continue

        if not corrected.strip():

            typer.echo(
                "Correção vazia, pulado.\n"
            )

            continue

        update_correction(
            ex["id"],
            corrected,
        )

        typer.echo(
            f"Salvo: {ex['id']}\n"
        )


# ============================================================
# DATASET - EXPORTAR
# ============================================================


@app.command(
    "dataset-exportar"
)
def dataset_exportar(
    formato: str = typer.Option(
        "fewshot",
        help=(
            "'fewshot' (trecho pra colar no prompt) "
            "ou 'finetune' (jsonl pra treino)"
        ),
    ),
    n: int = typer.Option(
        3,
        help=(
            "Quantos exemplos incluir "
            "(só usado no formato fewshot)"
        ),
    ),
):
    """
    Exporta o dataset acumulado.
    """

    if formato == "fewshot":

        snippet = (
            export_fewshot_snippet(
                n=n
            )
        )

        if not snippet:

            typer.echo(
                "Dataset vazio — adicione exemplos "
                "com 'dataset-adicionar' primeiro."
            )

            raise typer.Exit(
                1
            )

        typer.echo(
            snippet
        )

    elif formato == "finetune":

        path = (
            export_finetune_jsonl()
        )

        typer.echo(
            f"Exportado para: {path}"
        )

    else:

        typer.echo(
            "Formato inválido. "
            "Use 'fewshot' ou 'finetune'."
        )

        raise typer.Exit(
            1
        )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    app()

