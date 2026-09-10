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
        output_ids[len(input_ids):]
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


def _build_full_analysis_prompt(
    contexts: list[dict[str, Any]],
) -> str:

    scenes_text = []

    for index, context in enumerate(contexts, start=1):
        scenes_text.append(f"""
CENA {index}
INÍCIO: {context.get("start")}
FIM: {context.get("end")}

ANÁLISE:
{context.get("context", "").strip()}
""".strip())

    return f"""
Você está analisando um filme, anime ou episódio completo.

Abaixo estão as análises individuais das cenas já processadas.

Sua tarefa é consolidar essas informações em uma única análise
cronológica e objetiva do vídeo inteiro.

Use SOMENTE as informações presentes nas análises das cenas.

Não invente acontecimentos.
Não invente nomes.
Não atribua intenções psicológicas que não estejam sustentadas.
Priorize acontecimentos concretos e observáveis.

É importante preservar a sequência dos acontecimentos.

ANÁLISES DAS CENAS:

{"\n\n".join(scenes_text)}

Retorne exatamente neste formato:

CONTEXTO GERAL

<descrição objetiva do contexto geral do vídeo>

PERSONAGENS

- <personagem>: <participação e acontecimentos relevantes>

AMBIENTES

- <ambiente>: <acontecimentos relevantes>

SEQUÊNCIA DOS ACONTECIMENTOS

- <acontecimento em ordem cronológica>
- <acontecimento em ordem cronológica>
- <acontecimento em ordem cronológica>

AÇÕES IMPORTANTES

- <ação>
- <ação>

FALAS IMPORTANTES

- <personagem>: <fala ou resumo>
- <personagem>: <fala ou resumo>

ESTADOS / EMOÇÕES OBSERVÁVEIS

- <personagem>: <estado ou emoção aparente>

SÍNTESE

<resumo objetivo do vídeo inteiro>

Se alguma seção não possuir informação suficiente, escreva:

Não identificado.
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
