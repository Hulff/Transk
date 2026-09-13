"""
Character Consistency

Responsável por manter consistência textual entre as cenas depois que
a identidade global já foi resolvida pelo identity_resolver.

IMPORTANTE:

Este módulo NÃO decide quem é quem.

A decisão de identidade pertence a:

    core.identity_resolver

Aqui apenas consumimos:

    character_id
    character_name
    speaker_id
    character_registry
    resolved_characters
    speaker_character_mapping

e usamos essas informações para produzir uma descrição textual
consistente.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _load_model(config: dict[str, Any]):
    scene_cfg = config.get(
        "scene_analysis",
        {},
    )

    model_name = scene_cfg.get(
        "model",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )

    load_in_4bit = scene_cfg.get(
        "load_in_4bit",
        False,
    )

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": scene_cfg.get(
            "attn_implementation",
            "sdpa",
        ),
    }

    if load_in_4bit:
        kwargs["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        **kwargs,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def _generate_text(
    model,
    processor,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=False,
        )

    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    result = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result.strip()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def normalize_registry(
    registry: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:

    if not isinstance(registry, dict):
        return {}

    result = {}

    for character_id, character in registry.items():

        if not isinstance(character, dict):
            continue

        item = dict(character)

        item["character_id"] = item.get("character_id") or character_id

        item.setdefault(
            "name",
            None,
        )

        item.setdefault(
            "aliases",
            [],
        )

        item.setdefault(
            "stable_appearance",
            {},
        )

        item.setdefault(
            "temporary_appearance",
            {},
        )

        item.setdefault(
            "distinctive_features",
            [],
        )

        item.setdefault(
            "speaker_ids",
            [],
        )

        item.setdefault(
            "speaker_associations",
            [],
        )

        item.setdefault(
            "scenes",
            [],
        )

        item.setdefault(
            "identity_status",
            "AMBÍGUO",
        )

        result[item["character_id"]] = item

    return result


def get_registry_character(
    registry: dict[str, dict[str, Any]],
    character_id: str | None,
) -> dict[str, Any] | None:

    if not character_id:
        return None

    return registry.get(character_id)


def get_character_display_name(
    character: dict[str, Any] | None,
) -> str | None:

    if not character:
        return None

    name = character.get("name")

    if name and str(name).strip():
        return str(name).strip()

    return None


# ---------------------------------------------------------------------------
# Descrições
# ---------------------------------------------------------------------------


def _clean_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item not in (None, ""))

    if isinstance(value, dict):
        parts = []

        for key, item in value.items():
            if item in (None, "", []):
                continue

            parts.append(f"{key}: {item}")

        return "; ".join(parts)

    return str(value)


def describe_stable_appearance(
    character: dict[str, Any],
) -> str:

    appearance = character.get(
        "stable_appearance",
        {},
    )

    if not isinstance(appearance, dict):
        return ""

    fields = [
        "skin_tone",
        "hair_color",
        "hair_length",
        "hair_style",
        "face_shape",
        "facial_hair",
        "build",
        "body_proportions",
        "silhouette",
        "eyes",
        "eyebrows",
        "ears",
        "nose",
        "mouth",
        "visible_marks",
    ]

    parts = []

    for field in fields:
        value = appearance.get(field)

        if value in (
            None,
            "",
            [],
        ):
            continue

        parts.append(f"{field}={_clean_value(value)}")

    distinctive = character.get(
        "distinctive_features",
        [],
    )

    if distinctive:
        parts.append("características distintivas=" + _clean_value(distinctive))

    return "; ".join(parts)


def describe_temporary_appearance(
    character: dict[str, Any],
) -> str:

    appearance = character.get(
        "temporary_appearance",
        {},
    )

    if not isinstance(appearance, dict):
        return ""

    fields = [
        "clothing",
        "clothing_colors",
        "clothing_patterns",
        "footwear",
        "headwear",
        "accessories",
        "weapons",
    ]

    parts = []

    for field in fields:
        value = appearance.get(field)

        if value in (
            None,
            "",
            [],
        ):
            continue

        parts.append(f"{field}={_clean_value(value)}")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Contexto de personagem
# ---------------------------------------------------------------------------


def build_character_context(
    character: dict[str, Any],
) -> str:

    character_id = character.get(
        "character_id",
        "unknown",
    )

    name = character.get("name") or "NÃO IDENTIFICADO"

    status = character.get(
        "identity_status",
        "AMBÍGUO",
    )

    stable = describe_stable_appearance(character)

    temporary = describe_temporary_appearance(character)

    speakers = ", ".join(
        character.get(
            "speaker_ids",
            [],
        )
    )

    scenes = ", ".join(
        str(item)
        for item in character.get(
            "scenes",
            [],
        )
    )

    lines = [
        f"CHARACTER_ID: {character_id}",
        f"NAME: {name}",
        f"IDENTITY_STATUS: {status}",
        f"SPEAKERS: {speakers or 'não identificado'}",
        f"SCENES: {scenes or 'não identificado'}",
    ]

    if stable:
        lines.append(f"STABLE_APPEARANCE: {stable}")

    if temporary:
        lines.append(f"TEMPORARY_APPEARANCE: {temporary}")

    return "\n".join(lines)


def build_scene_character_registry(
    context: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> str:

    resolved = context.get(
        "resolved_characters",
        [],
    )

    if not isinstance(resolved, list):
        return "Nenhum personagem resolvido."

    blocks = []

    for item in resolved:

        if not isinstance(item, dict):
            continue

        character_id = item.get("character_id")

        if not character_id:
            continue

        character = registry.get(character_id)

        if not character:
            continue

        blocks.append(build_character_context(character))

    if not blocks:
        return "Nenhum personagem resolvido."

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Speaker mapping
# ---------------------------------------------------------------------------


def build_speaker_character_map(
    context: dict[str, Any],
) -> dict[str, dict[str, Any]]:

    mapping = context.get(
        "speaker_character_mapping",
        {},
    )

    if isinstance(mapping, dict):
        return mapping

    return {}


def resolve_segment_character(
    segment: dict[str, Any],
    context: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    result = dict(segment)

    speaker_id = segment.get("speaker_id") or segment.get("speaker")

    result["speaker_id"] = speaker_id

    mapping = build_speaker_character_map(context)

    speaker_info = mapping.get(speaker_id)

    if isinstance(speaker_info, dict):

        character_id = speaker_info.get("character_id")

        result["character_id"] = character_id

        character = registry.get(character_id)

        if character:
            result["character_name"] = character.get("name")

    return result


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------


def format_dialogue_with_characters(
    dialogue: Any,
    context: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> str:

    if isinstance(dialogue, str):
        return dialogue.strip()

    if not isinstance(dialogue, list):
        return str(dialogue or "").strip()

    lines = []

    for segment in dialogue:

        if not isinstance(segment, dict):
            continue

        enriched = resolve_segment_character(
            segment,
            context,
            registry,
        )

        speaker_id = enriched.get("speaker_id")

        character_name = enriched.get("character_name")

        label = character_name or speaker_id or "PERSONAGEM"

        text = enriched.get("text") or enriched.get("content") or ""

        start = enriched.get("start")

        end = enriched.get("end")

        if start is not None and end is not None:
            lines.append(f"[{start} - {end}] " f"{label}: {text}")
        else:
            lines.append(f"{label}: {text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eventos
# ---------------------------------------------------------------------------


def normalize_scene_events(
    context: dict[str, Any],
) -> dict[str, Any]:

    events = context.get(
        "scene_events",
        {},
    )

    if isinstance(events, dict):
        return events

    return {}


# ---------------------------------------------------------------------------
# Cena consistente
# ---------------------------------------------------------------------------


def build_consistent_scene(
    context: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    result = dict(context)

    result["character_registry"] = registry

    result["character_registry_context"] = build_scene_character_registry(
        context,
        registry,
    )

    result["speaker_character_mapping"] = build_speaker_character_map(context)

    result["scene_events"] = normalize_scene_events(context)

    result["dialogue_with_characters"] = format_dialogue_with_characters(
        context.get("dialogue", ""),
        context,
        registry,
    )

    return result


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_consistency_prompt(
    context: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> str:

    consistent_scene = build_consistent_scene(
        context,
        registry,
    )

    return f"""
Você está realizando uma etapa de consistência textual de uma cena.

IMPORTANTE:

A identidade dos personagens JÁ FOI RESOLVIDA por um sistema separado.

Você NÃO deve tentar descobrir novamente quem é cada personagem.

Você deve respeitar os seguintes identificadores:

- speaker_id = identidade original da diarização;
- character_id = identidade global resolvida;
- character_name = nome associado à identidade global.

Não altere esses identificadores.

Não crie novos personagens.

Não combine personagens diferentes.

Não separe um personagem em outro personagem apenas porque a roupa
mudou.

ROUPA E ACESSÓRIOS NÃO SÃO PROVA SUFICIENTE DE IDENTIDADE.

Priorize características estáveis:

- rosto;
- cabelo;
- características físicas;
- proporções corporais;
- marcas distintivas.

Se a identidade estiver marcada como AMBÍGUO, preserve a ambiguidade.
Não tente resolver por conta própria.

REGISTRO DE PERSONAGENS:

{consistent_scene["character_registry_context"]}

MAPEAMENTO SPEAKER -> CHARACTER:

{consistent_scene["speaker_character_mapping"]}

DIÁLOGO:

{consistent_scene["dialogue_with_characters"]}

DESCRIÇÃO ORIGINAL:

{context.get("context", "")}

EVENTOS:

{consistent_scene["scene_events"]}

Sua tarefa é apenas melhorar a descrição textual da cena.

Não invente:

- eventos;
- falas;
- personagens;
- nomes;
- objetos;
- ações;
- intenções;
- acontecimentos.

Quando houver um personagem resolvido, use seu nome apenas se
character_name estiver disponível.

Quando não houver nome confiável, utilize um rótulo neutro consistente.

Não transforme SPEAKER_00 em outro personagem.

Não transforme character_001 em character_002.

Retorne apenas a descrição final da cena.
""".strip()


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------


def apply_character_consistency(
    contexts: list[dict[str, Any]],
    character_sheets: str | None,
    config: dict[str, Any],
    *,
    character_registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:

    registry = normalize_registry(character_registry)

    if not contexts:
        return []

    model, processor = _load_model(config)

    scene_cfg = config.get(
        "scene_analysis",
        {},
    )

    results = []

    try:

        for index, context in enumerate(contexts):

            prompt = build_consistency_prompt(
                context,
                registry,
            )

            raw = _generate_text(
                model,
                processor,
                prompt,
                max_new_tokens=scene_cfg.get(
                    "consistency_max_new_tokens",
                    300,
                ),
                repetition_penalty=scene_cfg.get(
                    "repetition_penalty",
                    1.15,
                ),
            )

            enriched = raw.strip()

            if not enriched:
                enriched = context.get(
                    "context",
                    "",
                )

            results.append(
                {
                    **context,
                    "enriched_description": enriched,
                    "character_registry": registry,
                    "character_sheets": (character_sheets or ""),
                    "needs_visual_review": any(
                        item.get("status") == "AMBÍGUO"
                        for item in context.get(
                            "resolved_characters",
                            [],
                        )
                        if isinstance(item, dict)
                    ),
                }
            )

            print(
                "[character_consistency] " f"Cena {index + 1}/" f"{len(contexts)}: ok"
            )

    finally:

        del model
        del processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Compatibilidade com app.py
# ---------------------------------------------------------------------------


def resolve_character_consistency(
    contexts: list[dict[str, Any]],
    character_sheets: str | None,
    config: dict[str, Any],
    character_registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:

    return apply_character_consistency(
        contexts,
        character_sheets,
        config,
        character_registry=character_registry,
    )
