# core/context_analysis.py

from __future__ import annotations

from typing import Any

import json
import re

import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

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


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_prompt(
    scene: dict[str, Any],
) -> str:

    dialogue = scene.get(
        "dialogue",
        "",
    )

    if isinstance(dialogue, list):
        dialogue = json.dumps(
            dialogue,
            ensure_ascii=False,
            indent=2,
        )

    dialogue = str(dialogue).strip()

    visual = str(
        scene.get(
            "description",
            scene.get(
                "context",
                "",
            ),
        )
        or ""
    ).strip()

    resolved_characters = scene.get(
        "resolved_characters",
        [],
    )

    speaker_mapping = scene.get(
        "speaker_character_mapping",
        {},
    )

    character_registry = scene.get(
        "character_registry",
        {},
    )

    scene_events = scene.get(
        "scene_events",
        {},
    )

    return f"""
Você está analisando uma cena de um filme, anime ou episódio.

Esta etapa NÃO é responsável por identificar personagens.

A identidade global dos personagens já foi resolvida por um sistema
externo.

Você deve PRESERVAR as identidades fornecidas.

Não altere:

- speaker_id;
- character_id;
- character_name.

Não crie novos personagens.

Não reidentifique personagens.

Não use conhecimento externo sobre filmes, animes ou franquias.

Não tente adivinhar nomes.

Não associe um personagem a outro apenas porque a aparência temporária
é semelhante.

ROUPA E ACESSÓRIOS NÃO SÃO PROVA SUFICIENTE DE IDENTIDADE.

Quando um personagem estiver como AMBÍGUO, mantenha a ambiguidade.

==================================================
IDENTIDADES RESOLVIDAS
==================================================

{json.dumps(
    resolved_characters,
    ensure_ascii=False,
    indent=2,
)}

==================================================
SPEAKER -> CHARACTER
==================================================

{json.dumps(
    speaker_mapping,
    ensure_ascii=False,
    indent=2,
)}

==================================================
REGISTRO GLOBAL
==================================================

{json.dumps(
    character_registry,
    ensure_ascii=False,
    indent=2,
)}

==================================================
EVENTOS ESTRUTURADOS
==================================================

{json.dumps(
    scene_events,
    ensure_ascii=False,
    indent=2,
)}

==================================================
DIÁLOGO
==================================================

{dialogue}

==================================================
DESCRIÇÃO VISUAL PRÉVIA
==================================================

{visual}

==================================================
TAREFA
==================================================

Produza uma descrição audiovisual objetiva da cena.

Descreva somente informações sustentadas pelo diálogo ou pelas imagens.

Registre:

- contexto da cena;
- personagens presentes;
- ambiente/local;
- ações físicas;
- objetos importantes;
- interações;
- acontecimentos;
- falas importantes;
- mudanças relevantes;
- estado ou emoção aparentemente observável.

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

Não invente nomes.

Não invente diálogos.

Não atribua ações a outro personagem.

Use as identidades fornecidas pelo sistema externo.

Se o registro disser:

character_id = character_001
character_name = Cell

então não transforme esse personagem em outro personagem.

Se o registro disser:

character_id = character_001
character_name = null

não invente um nome.

Se o registro disser:

status = AMBÍGUO

não tente resolver a identidade.

Retorne exatamente:

CONTEXTO
<descrição breve da situação>

PERSONAGENS
- <character_name ou identificador neutro>: <ação/estado>

AMBIENTE
<ambiente relevante>

ACONTECIMENTOS
- <acontecimento>
- <acontecimento>

AÇÕES
- <ação>
- <ação>

FALAS IMPORTANTES
- <personagem>: <fala ou resumo>

ESTADO / EMOÇÕES OBSERVÁVEIS
- <personagem>: <estado>

Se uma seção não tiver informação suficiente:

Não identificado.
""".strip()


# ---------------------------------------------------------------------------
# Scene analysis
# ---------------------------------------------------------------------------


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

    scene_cfg = config.get(
        "scene_analysis",
        {},
    )

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=scene_cfg.get(
                "context_max_new_tokens",
                500,
            ),
            repetition_penalty=scene_cfg.get(
                "repetition_penalty",
                1.15,
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


def _clean_result(
    text: str,
) -> str:

    text = text.strip()

    text = re.sub(
        r"^(assistant|Assistant)\s*:?\s*",
        "",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

            frames = frames_by_scene.get(
                index,
                [],
            )

            if not frames:

                results.append(
                    {
                        **scene,
                        "scene_id": scene.get(
                            "scene_id",
                            index,
                        ),
                        "start": scene.get("start"),
                        "end": scene.get("end"),
                        "context": (
                            "Não foi possível " "analisar visualmente " "esta cena."
                        ),
                        "resolved_characters": (
                            scene.get(
                                "resolved_characters",
                                [],
                            )
                        ),
                        "speaker_character_mapping": (
                            scene.get(
                                "speaker_character_mapping",
                                {},
                            )
                        ),
                        "character_registry": (
                            scene.get(
                                "character_registry",
                                {},
                            )
                        ),
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
                    **scene,
                    "scene_id": scene.get(
                        "scene_id",
                        index,
                    ),
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


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_context(
    contexts: list[dict[str, Any]],
    path: str,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            contexts,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def _format_scenes_text(
    contexts: list[dict[str, Any]],
) -> str:

    scenes_text = []

    for index, context in enumerate(
        contexts,
        start=1,
    ):

        dialogue = context.get(
            "dialogue",
            "",
        )

        if isinstance(dialogue, list):
            dialogue = json.dumps(
                dialogue,
                ensure_ascii=False,
            )

        visual = context.get(
            "context",
            "",
        )

        resolved = context.get(
            "resolved_characters",
            [],
        )

        mapping = context.get(
            "speaker_character_mapping",
            {},
        )

        scenes_text.append(f"""
SCENE {index}
START: {context.get("start")}
END: {context.get("end")}

RESOLVED CHARACTERS:
{json.dumps(
    resolved,
    ensure_ascii=False,
    indent=2,
)}

SPEAKER -> CHARACTER:
{json.dumps(
    mapping,
    ensure_ascii=False,
    indent=2,
)}

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
Você está consolidando a análise completa de um vídeo.

As identidades dos personagens já foram resolvidas anteriormente.

NÃO tente resolver novamente as identidades.

Use:

- character_id;
- character_name;
- speaker_id;

exatamente como fornecidos.

Não troque personagens.

Não crie personagens.

Não invente nomes.

Não use conhecimento externo.

Não corrija nomes com base em conhecimento da franquia.

Preserve ambiguidades.

Use somente informações sustentadas pelas cenas.

Priorize:

- personagens;
- locais;
- ações;
- interações;
- eventos;
- objetos;
- falas;
- mudanças;
- sequência cronológica.

Não invente:

- eventos;
- motivações;
- intenções;
- diálogos;
- nomes;
- relações entre personagens.

CENAS:

{scenes_text}

Retorne exatamente:

GENERAL CONTEXT

<descrição objetiva>

CHARACTERS

- <personagem>: <participação>

ENVIRONMENTS

- <local>: <eventos relevantes>

SEQUENCE OF EVENTS

- <evento>
- <evento>
- <evento>

IMPORTANT ACTIONS

- <ação>
- <ação>

IMPORTANT DIALOGUE

- <personagem>: <fala ou resumo>
- <personagem>: <fala ou resumo>

OBSERVABLE STATES / EMOTIONS

- <personagem>: <estado>

SUMMARY

<resumo objetivo>
""".strip()


def analyze_full_context(
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:

    if not contexts:
        return "Não foi possível gerar a análise: " "nenhuma cena foi processada."

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

        scene_cfg = config.get(
            "scene_analysis",
            {},
        )

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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _build_validation_prompt(
    draft_analysis: str,
    contexts: list[dict[str, Any]],
) -> str:

    scenes_text = _format_scenes_text(contexts)

    return f"""
Você deve validar uma análise consolidada de vídeo.

ANÁLISE:

{draft_analysis}

FONTE ORIGINAL:

{scenes_text}

Corrija somente erros verificáveis.

Verifique:

- eventos inventados;
- personagens trocados;
- nomes incorretos;
- nomes inconsistentes;
- speaker_id atribuído incorretamente;
- character_id atribuído incorretamente;
- ações atribuídas ao personagem errado;
- ordem cronológica;
- falas inventadas;
- detalhes inexistentes;
- repetições;
- erros gramaticais.

IMPORTANTE:

Não tente reidentificar personagens.

Respeite os character_id e character_name fornecidos.

Não use conhecimento externo.

Não invente nomes.

Se uma identidade estiver ambígua, preserve a ambiguidade.

Retorne somente a análise corrigida.
""".strip()


def validate_analysis(
    draft_analysis: str,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:

    if not draft_analysis or not contexts:
        return draft_analysis

    model, processor = _load_model(config)

    try:

        prompt = _build_validation_prompt(
            draft_analysis,
            contexts,
        )

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

        scene_cfg = config.get(
            "scene_analysis",
            {},
        )

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=scene_cfg.get(
                    "validation_max_new_tokens",
                    1000,
                ),
                repetition_penalty=scene_cfg.get(
                    "repetition_penalty",
                    1.15,
                ),
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

        result = _clean_result(result)

        return result or draft_analysis

    finally:

        del model
        del processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
