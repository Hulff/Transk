# core/scene_analysis.py

"""
Análise audiovisual usando Qwen2.5-VL.

Responsabilidades:

- extrair frames representativos das cenas;
- analisar visualmente os frames;
- identificar personagens visualmente presentes;
- extrair características estáveis e temporárias;
- registrar eventos e ações;
- associar personagens visuais aos speaker_id da diarização
  quando houver evidência suficiente;
- preservar os speaker_id originais;
- gerar um resumo textual compatível com o restante do pipeline.

IMPORTANTE:

O Qwen é utilizado como EXTRATOR DE EVIDÊNCIAS VISUAIS.

Ele NÃO é responsável por decidir definitivamente quem é um personagem
globalmente. A resolução definitiva de identidade é feita posteriormente
pelo core.identity_resolver.

Portanto:

    SPEAKER_00
        ↓
    evidência de voz
        ↓
    personagem visual da cena
        ↓
    evidência visual
        ↓
    identity_resolver
        ↓
    character_001 / Goku / etc.

Nunca sobrescrevemos o speaker_id original.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from core.ffmpeg_utils import get_ffmpeg_path


# ---------------------------------------------------------------------
# EXTRAÇÃO DE FRAMES
# ---------------------------------------------------------------------


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: int = 5,
) -> list[dict]:
    """
    Extrai frames periódicos do vídeo.

    Retorna:

    [
        {
            "timestamp": 0,
            "frame_path": ".../frame_00001.jpg"
        },
        ...
    ]
    """

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    pattern = str(
        Path(output_dir) / "frame_%05d.jpg"
    )

    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps=1/{interval_seconds}",
        pattern,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Erro ao extrair frames:\n{result.stderr}"
        )

    frames = sorted(
        Path(output_dir).glob("frame_*.jpg")
    )

    return [
        {
            "timestamp": i * interval_seconds,
            "frame_path": str(frame),
        }
        for i, frame in enumerate(frames)
    ]


def _extract_frame_at(
    video_path: str,
    timestamp: float,
    output_path: str,
):
    """
    Extrai um único frame em um timestamp específico.
    """

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-ss",
        str(max(timestamp, 0)),
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        output_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Erro ao extrair frame em {timestamp}s:\n"
            f"{result.stderr}"
        )


# ---------------------------------------------------------------------
# PROMPT LEGADO
# ---------------------------------------------------------------------

LEGACY_QUESTION = """
You are analyzing a scene from a movie, anime, TV episode, or other video.

Your task is to produce an objective audiovisual description of the entire
scene by combining the transcribed dialogue with what can be visually observed
across all provided frames.

The frames represent different moments of the same scene. Do NOT describe
each frame as an isolated image. Instead, identify actions, interactions,
events, movements, and relevant changes that occur throughout the scene.

The dialogue is contextual information. Do not treat dialogue alone as proof
that a physical event happened.

Describe ONLY information supported by the dialogue or visible in the frames.

Do not invent events.

Do not invent character names.

Only use a character's name if that exact name appears in the provided
dialogue transcript.

Do not use outside knowledge about the movie, show, or franchise to identify
characters.

If a character's name is not given in the dialogue, describe the character
using a neutral, consistent label.

Return an objective description.
""".strip()


# ---------------------------------------------------------------------
# PROMPT V2
# ---------------------------------------------------------------------

DEFAULT_QUESTION = """
You are a visual analyst of audiovisual scenes.

Analyze ALL provided frames TOGETHER as one continuous scene.

Do not treat each frame as an independent image.

Your task is to extract structured visual evidence that can later be used
by another system to determine whether the same character appears in
different scenes.

The system already has speaker identities produced by audio diarization.

Your job is NOT to make the final global character identification.

Instead:

1. identify visually distinguishable characters in the scene;
2. describe their observable physical characteristics;
3. associate a visual character with a speaker_id ONLY when the provided
   frames and dialogue give enough evidence to support the association;
4. explicitly mark uncertain associations as "indeterminado";
5. record actions and events;
6. produce an objective summary.

IMPORTANT IDENTITY RULES
========================

1. Describe ONLY visually observable characteristics.

2. Never invent visual characteristics.

3. If a characteristic cannot be determined, use:
   "indeterminado"

4. Do not use outside knowledge about the movie, anime, series, game,
   franchise, or character.

5. Do not identify a character because you recognize the fictional
   character from their appearance.

6. A character name may only be used when that exact name appears in the
   provided dialogue.

7. Do not assume that the person speaking is visually visible.

8. Do not assume that the closest person to the camera is the speaker.

9. Do not associate a speaker with a visual character merely because the
   character is present in the same scene.

10. A speaker association is STRONG only when the visual evidence and
    dialogue timing provide a reasonable basis for the association.

11. If a speaker is heard while the corresponding person is not visible,
    leave the visual speaker association as indeterminado.

12. If multiple visual characters could correspond to a speaker, mark the
    association as indeterminado instead of guessing.

13. Never overwrite, rename, or modify the original speaker_id.

14. A character can appear without speaking.

15. A speaker can speak while their face is not visible.

16. Background characters should not automatically receive speaker IDs.

17. Clothing is temporary evidence and should receive less importance than
    stable physical characteristics.

18. Stable characteristics include:
    - hair;
    - face;
    - skin tone;
    - facial hair;
    - body/build;
    - distinctive physical features.

19. Temporary characteristics include:
    - clothing;
    - accessories;
    - injuries;
    - dirt;
    - objects being carried.

20. A change of clothes does NOT automatically indicate a different person.

21. When two characters look similar, explicitly describe the features
    that can distinguish them.

22. Do not infer race, ethnicity, personality, intentions, or other
    non-observable attributes.

23. If visual evidence is weak, prefer "indeterminado".

DIALOGUE AND SPEAKERS
=====================

The following dialogue comes from audio transcription and diarization.

The speaker_id values are the ORIGINAL diarization identifiers.

{dialogue}

Available speaker IDs in this scene:

{speaker_ids}

Speaker timing information:

{speaker_timing}

When possible, use the timing information together with the frames.

For example, if SPEAKER_00 is speaking during a moment where exactly one
clearly visible character is speaking on screen, that may be evidence for
associating that visual character with SPEAKER_00.

However, do NOT make the association if the evidence is ambiguous.

CHARACTER REFERENCES
====================

Create stable visual references:

personagem_A
personagem_B
personagem_C
...

The same visual character must keep the same character_ref throughout
this scene.

Do not use a global character name as character_ref.

STRUCTURED OUTPUT
=================

Return EXACTLY these three blocks:

CHARACTERS_JSON:
[
  {{
    "character_ref": "personagem_A",

    "name": null,

    "speaker_associations": [
      {{
        "speaker_id": "SPEAKER_00",
        "status": "confirmado",
        "evidence": "A pessoa está visivelmente falando durante o trecho associado ao speaker.",
        "confidence": "alta"
      }}
    ],

    "stable_appearance": {{
      "skin_tone": "indeterminado",
      "hair_color": "indeterminado",
      "hair_length": "indeterminado",
      "hair_style": "indeterminado",
      "face_shape": "indeterminado",
      "facial_hair": "indeterminado",
      "build": "indeterminado",
      "distinctive_features": []
    }},

    "temporary_appearance": {{
      "clothing": [],
      "accessories": [],
      "temporary_features": []
    }},

    "visual_evidence": {{
      "visible_in_frames": [],
      "visibility_quality": "alta"
    }}
  }}
]

RULES FOR speaker_associations:

- Use [] when no speaker can be associated with sufficient evidence.
- "status" must be one of:
    "confirmado"
    "provavel"
    "indeterminado"

- "confidence" must be one of:
    "alta"
    "media"
    "baixa"

- Do NOT use "confirmado" when the character is merely visible while
  another person is speaking off-screen.

- If there is uncertainty, use "indeterminado".

- "visible_in_frames" should contain the frame indexes where the character
  is clearly visible.

SCENE_EVENTS_JSON:
{{
  "characters_present": [
    "personagem_A",
    "personagem_B"
  ],

  "enters_or_exits": [],

  "actions": [],

  "objects_used": [],

  "visual_changes": [],

  "speaker_events": [
    {{
      "speaker_id": "SPEAKER_00",
      "character_ref": "personagem_A",
      "status": "confirmado",
      "evidence": "O personagem aparece falando no momento correspondente."
    }}
  ]
}}

Rules:

- speaker_events must use only speaker IDs from the provided list.
- character_ref must use only character references from CHARACTERS_JSON.
- Use an empty list when no reliable association exists.
- Do not invent associations.

SUMMARY:
Write a short objective paragraph in {language}.

The summary must:

- describe what is visibly happening;
- mention relevant characters using their names only when names are
  explicitly established by the dialogue;
- otherwise use neutral descriptions;
- avoid invented events;
- avoid invented identities;
- make sense without the JSON blocks.

Do not output anything outside the three blocks.
""".strip()


# ---------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------


def _load_qwen_model(
    model_name: str,
    load_in_4bit: bool = False,
    attn_implementation: str = "sdpa",
):
    import torch

    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    print(
        f"[scene_analysis] Carregando {model_name} "
        f"(4bit={load_in_4bit})..."
    )

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        kwargs.pop("torch_dtype", None)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        **kwargs,
    )

    processor = AutoProcessor.from_pretrained(
        model_name
    )

    return model, processor


# ---------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------


def _extract_json_block(
    raw: str,
    start_marker: str,
    end_marker: str | None,
) -> str | None:
    """
    Extrai o conteúdo entre dois marcadores.
    """

    start_idx = raw.find(start_marker)

    if start_idx == -1:
        return None

    start_idx += len(start_marker)

    if end_marker:
        end_idx = raw.find(
            end_marker,
            start_idx,
        )

        if end_idx == -1:
            return raw[start_idx:].strip()

        return raw[
            start_idx:end_idx
        ].strip()

    return raw[start_idx:].strip()


def _strip_code_fence(
    text: str,
) -> str:
    """
    Remove fences Markdown como:

    ```json
    [...]
    ```
    """

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def _safe_list(value: Any) -> list:
    """
    Garante que um campo seja uma lista.
    """

    if isinstance(value, list):
        return value

    return []


def _safe_dict(value: Any) -> dict:
    """
    Garante que um campo seja um dict.
    """

    if isinstance(value, dict):
        return value

    return {}


def _normalize_character(
    character: dict,
    index: int,
) -> dict:
    """
    Normaliza um personagem retornado pelo Qwen.

    Isso evita que pequenas variações na saída do modelo quebrem
    o identity_resolver posteriormente.
    """

    character = dict(character)

    character_ref = character.get(
        "character_ref"
    )

    if not character_ref:
        character_ref = (
            f"personagem_{chr(65 + index)}"
        )

    stable = _safe_dict(
        character.get("stable_appearance")
    )

    temporary = _safe_dict(
        character.get("temporary_appearance")
    )

    stable_defaults = {
        "skin_tone": "indeterminado",
        "hair_color": "indeterminado",
        "hair_length": "indeterminado",
        "hair_style": "indeterminado",
        "face_shape": "indeterminado",
        "facial_hair": "indeterminado",
        "build": "indeterminado",
        "distinctive_features": [],
    }

    temporary_defaults = {
        "clothing": [],
        "accessories": [],
        "temporary_features": [],
    }

    for key, default in stable_defaults.items():
        if key not in stable:
            stable[key] = default

    for key, default in temporary_defaults.items():
        if key not in temporary:
            temporary[key] = default

    associations = _safe_list(
        character.get(
            "speaker_associations"
        )
    )

    normalized_associations = []

    for association in associations:

        if not isinstance(
            association,
            dict,
        ):
            continue

        speaker_id = association.get(
            "speaker_id"
        )

        if not speaker_id:
            continue

        status = str(
            association.get(
                "status",
                "indeterminado",
            )
        ).lower()

        if status not in {
            "confirmado",
            "provavel",
            "indeterminado",
        }:
            status = "indeterminado"

        confidence = str(
            association.get(
                "confidence",
                "baixa",
            )
        ).lower()

        if confidence not in {
            "alta",
            "media",
            "baixa",
        }:
            confidence = "baixa"

        normalized_associations.append(
            {
                "speaker_id": speaker_id,
                "status": status,
                "confidence": confidence,
                "evidence": str(
                    association.get(
                        "evidence",
                        "",
                    )
                ),
            }
        )

    visual_evidence = _safe_dict(
        character.get(
            "visual_evidence"
        )
    )

    visible_in_frames = _safe_list(
        visual_evidence.get(
            "visible_in_frames"
        )
    )

    visibility_quality = visual_evidence.get(
        "visibility_quality",
        "media",
    )

    return {
        "character_ref": character_ref,
        "name": character.get("name"),
        "speaker_associations": normalized_associations,
        "stable_appearance": stable,
        "temporary_appearance": temporary,
        "visual_evidence": {
            "visible_in_frames": visible_in_frames,
            "visibility_quality": visibility_quality,
        },
    }


def _normalize_scene_events(
    events: dict,
) -> dict:
    """
    Normaliza SCENE_EVENTS_JSON.
    """

    events = _safe_dict(events)

    return {
        "characters_present": _safe_list(
            events.get(
                "characters_present"
            )
        ),
        "enters_or_exits": _safe_list(
            events.get(
                "enters_or_exits"
            )
        ),
        "actions": _safe_list(
            events.get(
                "actions"
            )
        ),
        "objects_used": _safe_list(
            events.get(
                "objects_used"
            )
        ),
        "visual_changes": _safe_list(
            events.get(
                "visual_changes"
            )
        ),
        "speaker_events": _safe_list(
            events.get(
                "speaker_events"
            )
        ),
    }


def parse_structured_response(
    raw: str,
) -> tuple[list[dict], dict, str]:
    """
    Faz o parsing da resposta estruturada.

    Retorna:

        characters
        scene_events
        summary

    O parser é tolerante a falhas para não derrubar todo o pipeline
    caso o Qwen produza JSON inválido.
    """

    characters: list[dict] = []
    scene_events: dict = {}
    summary = raw.strip()

    chars_block = _extract_json_block(
        raw,
        "CHARACTERS_JSON:",
        "SCENE_EVENTS_JSON:",
    )

    events_block = _extract_json_block(
        raw,
        "SCENE_EVENTS_JSON:",
        "SUMMARY:",
    )

    summary_block = _extract_json_block(
        raw,
        "SUMMARY:",
        None,
    )

    if chars_block:

        try:

            parsed = json.loads(
                _strip_code_fence(
                    chars_block
                )
            )

            if isinstance(
                parsed,
                list,
            ):
                characters = [
                    _normalize_character(
                        character,
                        index,
                    )
                    for index, character in enumerate(
                        parsed
                    )
                    if isinstance(
                        character,
                        dict,
                    )
                ]

        except (
            json.JSONDecodeError,
            ValueError,
        ):

            print(
                "[scene_analysis] "
                "aviso: CHARACTERS_JSON "
                "malformado, ignorando."
            )

    if events_block:

        try:

            parsed = json.loads(
                _strip_code_fence(
                    events_block
                )
            )

            if isinstance(
                parsed,
                dict,
            ):
                scene_events = _normalize_scene_events(
                    parsed
                )

        except (
            json.JSONDecodeError,
            ValueError,
        ):

            print(
                "[scene_analysis] "
                "aviso: SCENE_EVENTS_JSON "
                "malformado, ignorando."
            )

    if summary_block:

        summary = summary_block.strip()

    return (
        characters,
        scene_events,
        summary,
    )


# ---------------------------------------------------------------------
# SPEAKERS
# ---------------------------------------------------------------------


def _get_segment_speaker(
    segment: dict,
) -> str:
    """
    Retorna o speaker_id sem destruir compatibilidade com o campo
    antigo 'speaker'.
    """

    return str(
        segment.get(
            "speaker_id",
            segment.get(
                "speaker",
                "?",
            ),
        )
    )


def _build_dialogue(
    segments: list[dict],
) -> str:
    """
    Formata o diálogo mantendo speaker_id explícito.

    Também inclui timestamp para permitir que o Qwen relacione
    visualmente momentos da cena com a fala.
    """

    lines = []

    for segment in segments:

        speaker_id = _get_segment_speaker(
            segment
        )

        start = float(
            segment.get(
                "start",
                0,
            )
        )

        end = float(
            segment.get(
                "end",
                start,
            )
        )

        text = str(
            segment.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        lines.append(
            f"[{start:.2f}s - {end:.2f}s] "
            f"{speaker_id}: {text}"
        )

    return "\n".join(lines)


def _build_speaker_ids(
    segments: list[dict],
) -> list[str]:
    """
    Retorna speaker IDs únicos na ordem em que aparecem.
    """

    result = []

    seen = set()

    for segment in segments:

        speaker_id = _get_segment_speaker(
            segment
        )

        if speaker_id == "?":
            continue

        if speaker_id not in seen:

            seen.add(
                speaker_id
            )

            result.append(
                speaker_id
            )

    return result


def _build_speaker_timing(
    segments: list[dict],
) -> str:
    """
    Cria um resumo temporal dos speakers.
    """

    grouped: dict[str, list[tuple[float, float]]] = {}

    for segment in segments:

        speaker_id = _get_segment_speaker(
            segment
        )

        if speaker_id == "?":
            continue

        start = float(
            segment.get(
                "start",
                0,
            )
        )

        end = float(
            segment.get(
                "end",
                start,
            )
        )

        grouped.setdefault(
            speaker_id,
            [],
        ).append(
            (
                start,
                end,
            )
        )

    lines = []

    for speaker_id, intervals in grouped.items():

        formatted = ", ".join(
            f"{start:.2f}-{end:.2f}s"
            for start, end in intervals
        )

        lines.append(
            f"{speaker_id}: {formatted}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# ASSOCIAÇÃO SPEAKER ↔ PERSONAGEM VISUAL
# ---------------------------------------------------------------------


def _validate_speaker_associations(
    characters: list[dict],
    scene_events: dict,
    valid_speaker_ids: set[str],
) -> tuple[list[dict], dict]:
    """
    Remove associações inválidas produzidas pelo modelo.

    O Qwen não pode inventar SPEAKER_99, por exemplo.

    Também garante que character_ref usado nos eventos realmente exista.
    """

    valid_character_refs = {
        character["character_ref"]
        for character in characters
    }

    for character in characters:

        associations = []

        for association in character.get(
            "speaker_associations",
            [],
        ):

            speaker_id = association.get(
                "speaker_id"
            )

            if speaker_id not in valid_speaker_ids:
                continue

            associations.append(
                association
            )

        character[
            "speaker_associations"
        ] = associations

    validated_events = dict(
        scene_events
    )

    speaker_events = []

    for event in scene_events.get(
        "speaker_events",
        [],
    ):

        if not isinstance(
            event,
            dict,
        ):
            continue

        speaker_id = event.get(
            "speaker_id"
        )

        character_ref = event.get(
            "character_ref"
        )

        if speaker_id not in valid_speaker_ids:
            continue

        if character_ref not in valid_character_refs:
            continue

        status = str(
            event.get(
                "status",
                "indeterminado",
            )
        ).lower()

        if status not in {
            "confirmado",
            "provavel",
            "indeterminado",
        }:
            status = "indeterminado"

        speaker_events.append(
            {
                "speaker_id": speaker_id,
                "character_ref": character_ref,
                "status": status,
                "evidence": str(
                    event.get(
                        "evidence",
                        "",
                    )
                ),
            }
        )

    validated_events[
        "speaker_events"
    ] = speaker_events

    return (
        characters,
        validated_events,
    )


def _build_speaker_character_index(
    characters: list[dict],
) -> dict[str, list[dict]]:
    """
    Cria um índice:

        SPEAKER_00 ->
            [
                {
                    character_ref: personagem_A,
                    status: confirmado,
                    ...
                }
            ]

    Isso facilita o identity_resolver.
    """

    index: dict[str, list[dict]] = {}

    for character in characters:

        character_ref = character.get(
            "character_ref"
        )

        for association in character.get(
            "speaker_associations",
            [],
        ):

            speaker_id = association.get(
                "speaker_id"
            )

            if not speaker_id:
                continue

            index.setdefault(
                speaker_id,
                [],
            ).append(
                {
                    "character_ref": character_ref,
                    "status": association.get(
                        "status",
                        "indeterminado",
                    ),
                    "confidence": association.get(
                        "confidence",
                        "baixa",
                    ),
                    "evidence": association.get(
                        "evidence",
                        "",
                    ),
                }
            )

    return index


# ---------------------------------------------------------------------
# QWEN
# ---------------------------------------------------------------------


def _describe_scene(
    model,
    processor,
    frame_paths: list[str],
    dialogue: str,
    speaker_ids: list[str],
    speaker_timing: str,
    question_template: str,
    language: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int | None,
    do_sample: bool,
) -> str:
    """
    Executa o Qwen2.5-VL sobre todos os frames da cena.
    """

    from qwen_vl_utils import process_vision_info

    question = question_template.format(
        dialogue=dialogue,
        speaker_ids=(
            ", ".join(speaker_ids)
            if speaker_ids
            else "nenhum speaker identificado"
        ),
        speaker_timing=(
            speaker_timing
            if speaker_timing
            else "nenhum intervalo disponível"
        ),
        language=language,
    )

    content = [
        {
            "type": "image",
            "image": path,
        }
        for path in frame_paths
    ]

    content.append(
        {
            "type": "text",
            "text": question,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(
        model.device
    )

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
    }

    if no_repeat_ngram_size:
        generate_kwargs[
            "no_repeat_ngram_size"
        ] = no_repeat_ngram_size

    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output.strip()


# ---------------------------------------------------------------------
# AGRUPAMENTO DE CENAS
# ---------------------------------------------------------------------


def group_into_scenes(
    segments: list[dict],
    gap_threshold_seconds: float = 3.0,
) -> list[dict]:
    """
    Agrupa segmentos consecutivos em cenas.

    A estrutura original dos segmentos é preservada.
    """

    if not segments:
        return []

    scenes = []

    current = [
        segments[0]
    ]

    for segment in segments[1:]:

        gap = (
            segment["start"]
            - current[-1]["end"]
        )

        if gap > gap_threshold_seconds:

            scenes.append(
                current
            )

            current = [
                segment
            ]

        else:

            current.append(
                segment
            )

    scenes.append(
        current
    )

    return [
        {
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "segments": group,
        }
        for group in scenes
    ]


# ---------------------------------------------------------------------
# DESCRIÇÃO DAS CENAS
# ---------------------------------------------------------------------


def describe_scenes(
    scenes: list[dict],
    video_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    frames_per_scene: int = 12,
    padding_seconds: float = 0.5,
    output_dir: str = "temp/frames_cenas",
    question: str = DEFAULT_QUESTION,
    language: str = "pt-BR",
    max_new_tokens: int = 600,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int | None = None,
    do_sample: bool = False,
    load_in_4bit: bool = False,
) -> list[dict]:
    """
    Analisa todas as cenas com Qwen2.5-VL.

    Cada resultado possui:

    {
        "scene_id": 0,
        "timestamp": ...,
        "end": ...,
        "frames": [...],

        "dialogue": "...",

        "speaker_ids": [
            "SPEAKER_00"
        ],

        "speaker_timing": "...",

        "description": "...",

        "visual_descriptors": [...],

        "scene_events": {...},

        "speaker_character_index": {
            "SPEAKER_00": [...]
        }
    }

    O campo "frames" é importante porque o identity_resolver pode
    posteriormente utilizar as mesmas evidências visuais.
    """

    model, processor = _load_qwen_model(
        model_name,
        load_in_4bit=load_in_4bit,
    )

    print(
        "[scene_analysis] Modelo carregado. "
        f"Analisando {len(scenes)} cenas..."
    )

    results = []

    try:

        for idx, scene in enumerate(
            scenes
        ):

            start = float(
                scene["start"]
            )

            end = float(
                scene["end"]
            )

            duration = max(
                end - start,
                0.1,
            )

            # ---------------------------------------------------------
            # TIMESTAMPS
            # ---------------------------------------------------------

            if frames_per_scene <= 1:

                timestamps = [
                    start + duration / 2
                ]

            else:

                timestamps = [
                    start
                    - padding_seconds
                    + (
                        duration
                        + 2 * padding_seconds
                    )
                    * i
                    / (
                        frames_per_scene - 1
                    )
                    for i in range(
                        frames_per_scene
                    )
                ]

            timestamps = [
                max(
                    timestamp,
                    0,
                )
                for timestamp in timestamps
            ]

            # ---------------------------------------------------------
            # FRAMES
            # ---------------------------------------------------------

            frame_paths = []

            frame_metadata = []

            for frame_index, timestamp in enumerate(
                timestamps
            ):

                frame_path = str(
                    Path(output_dir)
                    / f"cena{idx:04d}_{frame_index}.jpg"
                )

                try:

                    _extract_frame_at(
                        video_path,
                        timestamp,
                        frame_path,
                    )

                    frame_paths.append(
                        frame_path
                    )

                    frame_metadata.append(
                        {
                            "frame_index": frame_index,
                            "timestamp": timestamp,
                            "frame_path": frame_path,
                        }
                    )

                except RuntimeError as error:

                    print(
                        "[scene_analysis] "
                        f"Erro ao extrair frame "
                        f"{timestamp:.1f}s: {error}"
                    )

            # ---------------------------------------------------------
            # DIÁLOGO
            # ---------------------------------------------------------

            scene_segments = scene.get(
                "segments",
                [],
            )

            dialogue = _build_dialogue(
                scene_segments
            )

            speaker_ids = _build_speaker_ids(
                scene_segments
            )

            speaker_timing = _build_speaker_timing(
                scene_segments
            )

            # ---------------------------------------------------------
            # ANÁLISE
            # ---------------------------------------------------------

            description = ""

            visual_descriptors: list[
                dict
            ] = []

            scene_events: dict = {}

            if frame_paths:

                try:

                    raw_response = _describe_scene(
                        model=model,
                        processor=processor,
                        frame_paths=frame_paths,
                        dialogue=dialogue,
                        speaker_ids=speaker_ids,
                        speaker_timing=speaker_timing,
                        question_template=question,
                        language=language,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        do_sample=do_sample,
                    )

                    (
                        visual_descriptors,
                        scene_events,
                        description,
                    ) = parse_structured_response(
                        raw_response
                    )

                    (
                        visual_descriptors,
                        scene_events,
                    ) = _validate_speaker_associations(
                        visual_descriptors,
                        scene_events,
                        set(speaker_ids),
                    )

                except Exception as error:

                    print(
                        "[scene_analysis] "
                        f"Erro ao analisar cena: {error}"
                    )

            # ---------------------------------------------------------
            # ÍNDICE SPEAKER -> PERSONAGEM
            # ---------------------------------------------------------

            speaker_character_index = (
                _build_speaker_character_index(
                    visual_descriptors
                )
            )

            # ---------------------------------------------------------
            # RESULTADO
            # ---------------------------------------------------------

            result = {
                "scene_id": idx,

                "timestamp": start,

                "start": start,

                "end": end,

                "frames": frame_metadata,

                "frame_paths": frame_paths,

                "dialogue": dialogue,

                "speaker_ids": speaker_ids,

                "speaker_timing": speaker_timing,

                "description": description,

                "visual_descriptors": visual_descriptors,

                "scene_events": scene_events,

                "speaker_character_index": (
                    speaker_character_index
                ),
            }

            results.append(
                result
            )

            # ---------------------------------------------------------
            # LOG
            # ---------------------------------------------------------

            print(
                "[scene_analysis] Cena "
                f"{idx + 1}/{len(scenes)} "
                f"({start:.1f}s - {end:.1f}s)"
            )

            print(
                "[scene_analysis] "
                f"Speakers: "
                f"{', '.join(speaker_ids) if speaker_ids else 'nenhum'}"
            )

            if visual_descriptors:

                print(
                    "[scene_analysis] "
                    f"Personagens visuais: "
                    f"{len(visual_descriptors)}"
                )

                for character in visual_descriptors:

                    associations = character.get(
                        "speaker_associations",
                        [],
                    )

                    if associations:

                        mapping_text = ", ".join(
                            (
                                f"{item['speaker_id']}="
                                f"{item['status']}"
                            )
                            for item in associations
                        )

                    else:

                        mapping_text = (
                            "sem associação de speaker"
                        )

                    print(
                        "[scene_analysis]   "
                        f"{character['character_ref']}: "
                        f"{mapping_text}"
                    )

            if description:

                print(
                    description
                )

    finally:

        del model
        del processor

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------
# COMPATIBILIDADE
# ---------------------------------------------------------------------


def describe_dialogue_scenes(
    segments: list[dict],
    video_path: str,
    **kwargs,
) -> list[dict]:
    """
    Mantém a API antiga.

    Cada segmento vira uma cena independente.
    """

    scenes = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "segments": [segment],
        }
        for segment in segments
    ]

    return describe_scenes(
        scenes,
        video_path,
        **kwargs,
    )

