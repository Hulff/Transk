"""
Consistência de personagens entre cenas.

Depois que todas as cenas já foram analisadas visualmente, monta uma
ficha consolidada por personagem (nome exato do diálogo, ou rótulo
neutro consistente + descrição física), juntando informação espalhada
por todas as cenas. Em seguida, reescreve — só em texto, sem
reprocessar nenhum frame — a descrição de cada cena usando essa ficha,
corrigindo referências vagas de cenas antigas quando o personagem só
foi identificado/nomeado em cenas posteriores 

Casos em que o modelo não consegue resolver a identidade com
confiança são sinalizados (`needs_visual_review`) em vez de chutados —
ficam candidatos a reprocessamento visual posterior, só esses casos
específicos, não a cena inteira de novo.
"""

from typing import Any

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


def _load_model(config: dict[str, Any]):
    scene_cfg = config.get("scene_analysis", {})

    model_name = scene_cfg.get(
        "model",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )

    load_in_4bit = scene_cfg.get("load_in_4bit", False)

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": scene_cfg.get("attn_implementation", "sdpa"),
    }

    if load_in_4bit:
        kwargs["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def _generate_text(
    model,
    processor,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(text=[text], padding=True, return_tensors="pt")
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
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    result = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result.strip()


def _format_scenes_text(contexts: list[dict[str, Any]]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"SCENE {i}\n"
            f"DIALOGUE:\n{ctx.get('dialogue', '').strip()}\n\n"
            f"DESCRIPTION:\n{ctx.get('context', '').strip()}"
        )
    return "\n\n---\n\n".join(blocks)


CHARACTER_SHEET_PROMPT = """
Below are the individual scene analyses of a video (dialogue + visual
description for each scene).

{scenes_text}

Build a character reference sheet for each character that can be
identified — either by an exact name that appears in the dialogue, or
by a distinctive, consistent visual description repeated across
multiple scenes.

For each character, list:
- NAME: the exact name if it appears in the dialogue anywhere;
  otherwise write "UNNAMED" followed by the most distinctive neutral
  label used for them (e.g. "UNNAMED - character with green skin and
  purple clothing")
- DESCRIPTION: consistent physical traits (hair, clothing, build,
  etc.) as described across the scenes
- TRAITS: role, personality, or behavior - only what is explicitly
  supported by the scene analyses or dialogue
- SCENES: comma-separated scene numbers where this character appears

Only include a character if they appear in more than one scene, or
are clearly central to a scene. Do NOT invent names, traits, or
physical details that are not supported by the scene analyses above.

Return ONLY the character sheets, one per character, in this exact
format:

CHARACTER: <name or UNNAMED - label>
DESCRIPTION: <physical traits>
TRAITS: <role/personality>
SCENES: <scene numbers>
""".strip()


def build_character_sheets(
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    """
    Consolida uma ficha por personagem identificável, a partir de
    TODAS as cenas já analisadas. Não reprocessa nenhum frame - é uma
    chamada só de texto.
    """
    model, processor = _load_model(config)

    try:
        scene_cfg = config.get("scene_analysis", {})

        prompt = CHARACTER_SHEET_PROMPT.format(
            scenes_text=_format_scenes_text(contexts),
        )

        return _generate_text(
            model,
            processor,
            prompt,
            max_new_tokens=scene_cfg.get("character_sheet_max_new_tokens", 800),
            repetition_penalty=scene_cfg.get("repetition_penalty", 1.15),
        )
    finally:
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


SCENE_REWRITE_PROMPT = """
Here is a character reference sheet built from the entire video:

{character_sheets}

Here is one scene from that same video:

DIALOGUE:
{dialogue}

ORIGINAL DESCRIPTION:
{description}

Rewrite the ORIGINAL DESCRIPTION above, replacing vague or neutral
character references with the correct name from the character sheet,
but ONLY when the physical description in this scene clearly and
confidently matches a character sheet entry. Do not change anything
else - do not add new events, actions, or details that were not in
the original description.

If you cannot confidently match a character mentioned in this scene
to any entry in the character sheet, do NOT guess - keep the original
neutral reference for that character, and add a final line starting
with "NEEDS_VISUAL_REVIEW:" followed by a short reason.

If everything in the scene can be confidently resolved (or there was
nothing ambiguous to begin with), do not add a NEEDS_VISUAL_REVIEW
line at all.

Return ONLY the rewritten description (and the NEEDS_VISUAL_REVIEW
line if applicable) - no other commentary, no headers.
""".strip()


def apply_character_consistency(
    contexts: list[dict[str, Any]],
    character_sheets: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Reescreve, cena por cena, a descrição visual usando a ficha de
    personagens consolidada - só texto, sem reprocessar frames.

    Retorna uma lista paralela a `contexts`, cada item com:
      - "enriched_description": a descrição reescrita (ou a original,
        se não houve nada pra ajustar, ou se a geração veio vazia)
      - "needs_visual_review": True se o modelo não conseguiu resolver
        com confiança a identidade de algum personagem nessa cena
      - "review_reason": o motivo apontado pelo modelo, se houver
    """
    model, processor = _load_model(config)
    scene_cfg = config.get("scene_analysis", {})

    results = []

    try:
        for i, ctx in enumerate(contexts):
            prompt = SCENE_REWRITE_PROMPT.format(
                character_sheets=character_sheets,
                dialogue=ctx.get("dialogue", "").strip(),
                description=ctx.get("context", "").strip(),
            )

            raw = _generate_text(
                model,
                processor,
                prompt,
                max_new_tokens=scene_cfg.get("consistency_max_new_tokens", 300),
                repetition_penalty=scene_cfg.get("repetition_penalty", 1.15),
            )

            needs_review = False
            review_reason = None
            enriched = raw

            marker = "NEEDS_VISUAL_REVIEW:"
            if marker in raw:
                before, _, after = raw.partition(marker)
                enriched = before.strip()
                review_reason = after.strip()
                needs_review = True

            if not enriched.strip():
                enriched = ctx.get("context", "")

            results.append(
                {
                    "enriched_description": enriched,
                    "needs_visual_review": needs_review,
                    "review_reason": review_reason,
                }
            )

            status = (
                f"PRECISA REVISÃO VISUAL ({review_reason})" if needs_review else "ok"
            )
            print(f"[character_consistency] Cena {i + 1}/{len(contexts)}: {status}")

    finally:
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
