# core/context_analysis.py

from __future__ import annotations

from typing import Any
import json
import re

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


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


def _build_prompt(scene: dict[str, Any]) -> str:
    dialogue = scene.get("dialogue", "").strip()
    visual = scene.get("description", "").strip()

    return f"""
Você está analisando uma cena de um filme ou anime.

Sua tarefa é produzir uma descrição audiovisual objetiva da cena,
combinando o diálogo transcrito e o que pode ser observado visualmente.

DIÁLOGO:
{dialogue}

DESCRIÇÃO VISUAL PRÉVIA:
{visual}

Descreva somente informações sustentadas pelo diálogo ou pelas imagens.

A descrição deve registrar:

- contexto da cena;
- personagens presentes;
- ambiente/local quando for relevante;
- ações físicas realizadas pelos personagens;
- objetos importantes e interações com eles;
- acontecimentos relevantes;
- falas importantes;
- mudanças relevantes durante a cena;
- estado ou emoção APARENTE dos personagens quando puder ser inferido
  visualmente ou pelo diálogo.

Dê prioridade a acontecimentos concretos.

Exemplos:
- personagem pega um objeto;
- personagem abre uma porta;
- personagem corre;
- personagem cai;
- personagem ataca outro personagem;
- personagem pisa sobre outro personagem;
- personagem observa alguém;
- personagem entra ou sai de um local;
- ocorre uma explosão;
- um objeto é destruído.

Não invente acontecimentos.
Não invente nomes de personagens.
Não deduza acontecimentos que não possam ser sustentados pelo vídeo
ou pelo diálogo.
Não repita informações sem necessidade.

Retorne exatamente neste formato:

CONTEXTO
<descrição breve da situação da cena>

PERSONAGENS
- <personagem>: <ação/estado relevante>

AMBIENTE
<descrição do ambiente, somente se relevante>

ACONTECIMENTOS
- <acontecimento>
- <acontecimento>

AÇÕES
- <ação>
- <ação>

FALAS IMPORTANTES
- <personagem>: <fala ou resumo da fala>

ESTADO / EMOÇÕES OBSERVÁVEIS
- <personagem>: <estado ou emoção aparente>

Se uma seção não tiver informação suficiente, escreva:
Não identificado.
""".strip()


def _analyze_scene(
    model,
    processor,
    scene: dict[str, Any],
    frames: list[str],
    config: dict[str, Any],
) -> str:

    prompt = _build_prompt(scene)

    content = []

    for frame in frames:
        content.append(
            {
                "type": "image",
                "image": frame,
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
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

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    scene_cfg = config.get("scene_analysis", {})

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=scene_cfg.get("context_max_new_tokens", 500),
            repetition_penalty=scene_cfg.get(
                "repetition_penalty",
                1.15,
            ),
            no_repeat_ngram_size=scene_cfg.get(
                "no_repeat_ngram_size",
                3,
            ),
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return _clean_result(result)


def _clean_result(text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"^(assistant|Assistant)\s*:?\s*",
        "",
        text,
    )

    return text.strip()


def analyze_scenes(
    scenes: list[dict[str, Any]],
    frames_by_scene: dict[int, list[str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:

    if not scenes:
        return []

    model, processor = _load_model(config)

    results = []

    try:
        for index, scene in enumerate(scenes):

            frames = frames_by_scene.get(index, [])

            if not frames:
                results.append(
                    {
                        "start": scene.get("start"),
                        "end": scene.get("end"),
                        "context": "Não foi possível analisar visualmente esta cena.",
                    }
                )
                continue

            context = _analyze_scene(
                model=model,
                processor=processor,
                scene=scene,
                frames=frames,
                config=config,
            )

            results.append(
                {
                    "start": scene.get("start"),
                    "end": scene.get("end"),
                    "context": context,
                }
            )

    finally:
        del model
        del processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def save_context(
    contexts: list[dict[str, Any]],
    path: str,
) -> None:

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            contexts,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _format_scenes_text(
    contexts: list[dict[str, Any]],
) -> str:

    scenes_text = []

    for index, context in enumerate(contexts, start=1):

        dialogue = context.get(
            "dialogue",
            "",
        ).strip()

        visual = context.get(
            "context",
            "",
        ).strip()

        scenes_text.append(f"""
SCENE {index}
START: {context.get("start")}
END: {context.get("end")}

DIALOGUE:
{dialogue}

SCENE ANALYSIS:
{visual}
""".strip())

    return "\n\n".join(scenes_text)


def _build_full_analysis_prompt(
    contexts: list[dict[str, Any]],
) -> str:

    scenes_text = _format_scenes_text(contexts)

    return f"""
You are analyzing a complete movie, anime episode, TV episode, or other
long-form video.

Below are the individual analyses of the scenes that were already processed.

Your task is to consolidate these scene analyses into one objective,
chronological audiovisual analysis of the entire video.

Use ONLY information supported by the provided scene analyses and dialogue.

The goal is to describe what happens throughout the video, not to invent
a deeper interpretation of the story.

Prioritize:

- the main characters and their participation;
- relevant locations and environments;
- important physical actions;
- interactions between characters;
- interactions with objects;
- important events;
- changes in the situation;
- attacks, fights, falls, explosions, destruction, entrances, exits,
  movements, and other concrete events;
- important dialogue;
- observable or explicitly supported emotional states;
- the chronological sequence of events.

Preserve the order in which events occur.

Do NOT invent events.

Do NOT invent character names.

Only use a character's name if that exact name appears verbatim in the
DIALOGUE sections of the scene analyses above. Do not use outside
knowledge of any movie, show, or franchise to identify or correct a
character's name, even if you recognize who they might be.

If a character is not named in the dialogue, refer to them using a
neutral, consistent label instead (e.g. "the man", "the character with
spiky hair", "the taller character"). Use the SAME label for the same
character across the whole analysis — do not switch labels or spellings
partway through.

Do NOT invent dialogue.

Do NOT attribute motivations, intentions, symbolism, themes, or psychological
states unless they are explicitly supported by the provided information.

Prefer concrete descriptions.

For example:

"Cell steps on Android 17's head."

is preferable to:

"Cell steps on Android 17's head to demonstrate his superiority."

Only report the second interpretation if it is explicitly supported.

Do not merge unrelated events simply because they involve the same character.

Do not repeat the same event unnecessarily.

SCENE ANALYSES:

{scenes_text}

Return the final analysis exactly in this structure:

GENERAL CONTEXT

<brief objective description of the overall situation>

CHARACTERS

- <character>: <role, participation, and relevant actions>

ENVIRONMENTS

- <location/environment>: <relevant events or changes>

SEQUENCE OF EVENTS

- <event in chronological order>
- <event in chronological order>
- <event in chronological order>

IMPORTANT ACTIONS

- <important physical action>
- <important physical action>

IMPORTANT DIALOGUE

- <character>: <important line or concise summary>
- <character>: <important line or concise summary>

OBSERVABLE STATES / EMOTIONS

- <character>: <observable or clearly supported state>

SUMMARY

<objective summary of the entire video>

If a section does not contain enough information, write:

Not identified.
""".strip()


def analyze_full_context(
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:

    if not contexts:
        return "Não foi possível gerar a análise: nenhuma cena foi processada."

    model, processor = _load_model(config)

    try:
        prompt = _build_full_analysis_prompt(contexts)

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

        scene_cfg = config.get("scene_analysis", {})

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=scene_cfg.get(
                    "context_max_new_tokens",
                    1000,
                ),
                repetition_penalty=scene_cfg.get(
                    "repetition_penalty",
                    1.15,
                ),
                no_repeat_ngram_size=scene_cfg.get(
                    "no_repeat_ngram_size",
                    3,
                ),
                do_sample=False,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs["input_ids"],
                generated_ids,
            )
        ]

        result = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return _clean_result(result)

    finally:
        del model
        del processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_analysis(
    analysis: str,
    path: str,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(analysis.strip() + "\n")


def _build_validation_prompt(
    draft_analysis: str,
    contexts: list[dict[str, Any]],
) -> str:

    scenes_text = _format_scenes_text(contexts)

    return f"""
You previously wrote the following consolidated analysis of a video,
based on individual scene analyses and dialogue.

DRAFT ANALYSIS:

{draft_analysis}

Below are the ORIGINAL scene-by-scene analyses and dialogue that this
draft was supposed to be based on. Treat this as the only source of
truth — anything in the draft that isn't supported here is an error.

{scenes_text}

Carefully review the draft against the original scene analyses and
dialogue above. Check specifically for:

- events, objects, or details in the draft that are NOT supported by
  the scene analyses or dialogue;
- character NAME SPELLING errors or inconsistent spelling of the same
  character across the document (e.g. the same character being called
  by two different or misspelled names in different sections) — pick
  ONE correct, consistent spelling for each character and use it
  throughout the entire document;
- specific numeric or identifying labels (e.g. "Android 7", "Android
  8") that do NOT appear verbatim in the scene analyses or dialogue
  above — if the exact label isn't present in the source, replace it
  with a neutral description instead of guessing a number;
- actions or dialogue attributed to the wrong character;
- events placed in the wrong chronological order;
- grammar, spelling, punctuation, or phrasing errors — including
  broken/garbled words (e.g. "terraines", "saiyn", "Goham's");
- interpretations of motivation, symbolism, or psychological state
  that go beyond what is explicitly supported;
- unnecessary repetition of the same event or information.

Correct every issue you find. Do not introduce any new information
that isn't supported by the scene analyses or dialogue above — when in
doubt, remove the unsupported claim or use a neutral description
rather than guessing.

If the draft is already fully accurate, return it unchanged.

Return ONLY the corrected analysis, using EXACTLY the same section
structure as the draft (same headers, same order, plain text — no
markdown symbols like ### or **). Do not explain what you changed or
add any commentary outside the analysis itself.
""".strip()


def validate_analysis(
    draft_analysis: str,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    """
    Segunda passada sobre a análise consolidada: confere o rascunho
    contra as análises de cena originais (a fonte confiável, ancorada
    em frames + diálogo) e corrige inconsistências — nomes trocados,
    eventos mal atribuídos, alucinação, erros de escrita.
    """
    if not draft_analysis or not contexts:
        return draft_analysis

    model, processor = _load_model(config)

    try:
        prompt = _build_validation_prompt(draft_analysis, contexts)

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

        scene_cfg = config.get("scene_analysis", {})

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=scene_cfg.get(
                    "validation_max_new_tokens",
                    1200,
                ),
                repetition_penalty=scene_cfg.get(
                    "repetition_penalty",
                    1.15,
                ),
                no_repeat_ngram_size=scene_cfg.get(
                    "no_repeat_ngram_size",
                    3,
                ),
                do_sample=False,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs["input_ids"],
                generated_ids,
            )
        ]

        result = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        revised = _clean_result(result)

        # segurança: se a revisão vier vazia por algum motivo, mantém o rascunho
        return revised if revised.strip() else draft_analysis

    finally:
        del model
        del processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
