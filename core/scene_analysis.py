# core/scene_analysis.py


"""
Análise audiovisual usando Qwen2.5-VL.

O objetivo é descrever o que realmente acontece no vídeo:
- ambiente;
- personagens;
- ações;
- objetos;
- acontecimentos;
- expressões/estado observável.

O diálogo é usado como contexto, mas não deve ser tratado como
evidência de que uma ação aconteceu visualmente.
"""

import subprocess
from pathlib import Path

from core.ffmpeg_utils import get_ffmpeg_path


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: int = 5,
) -> list[dict]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pattern = str(Path(output_dir) / "frame_%05d.jpg")

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
        raise RuntimeError(f"Erro ao extrair frames:\n{result.stderr}")

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))

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
            f"Erro ao extrair frame em {timestamp}s:\n" f"{result.stderr}"
        )


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

Prioritize concrete and observable events, including:

- characters who appear in the scene;
- relevant environment or location;
- physical actions performed by characters;
- interactions between characters;
- interactions with objects;
- important objects;
- movements and changes during the scene;
- attacks, fights, falls, explosions, destruction, or other physical events;
- characters entering or leaving a location;
- characters looking at, approaching, touching, carrying, or using something;
- events that happen without dialogue;
- facial expressions or apparent emotional states when visually observable
  or clearly supported by the dialogue.

Examples of concrete events:

- a character picks up a sword;
- a character opens a door;
- a character starts running;
- a character falls;
- one character attacks another;
- one character steps on another character;
- a character looks at another character;
- a character enters or leaves a room;
- an explosion occurs;
- an object is destroyed;
- a character picks up an object and carries it away.

Do NOT primarily interpret themes, symbolism, hidden motivations, or
psychological intentions.

For example, prefer:

"Cell steps on Android 17's head."

instead of:

"Cell steps on Android 17's head to demonstrate his superiority."

Only describe the first statement unless the second is explicitly supported
by the video or dialogue.

Do not invent events.

Do not invent character names.

Only use a character's name if that exact name appears in the provided
dialogue transcript. Do not use outside knowledge of any movie, show,
or franchise to identify or name a character, even if you recognize
who they might be — the dialogue transcript is the ONLY source for
names.

If a character's name is not given in the dialogue, describe the
character using a neutral, consistent label instead, such as "the man",
"the woman", "the character with spiky hair", "the taller character",
or "the character in dark clothing". Reuse the SAME label for the same
character throughout your answer — do not switch labels mid-description.

Do not infer events that cannot be supported by the video or dialogue.

Do not repeat information unnecessarily.

Return the result exactly in the following structure:

CONTEXT

<brief description of the situation>

CHARACTERS

- <character>: <relevant action or observable state>

ENVIRONMENT

<relevant description of the environment or location>

EVENTS

- <important event>
- <important event>

ACTIONS

- <physical action>
- <physical action>

IMPORTANT DIALOGUE

- <character>: <important line or concise summary of the dialogue>

OBSERVABLE STATES / EMOTIONS

- <character>: <observable or clearly supported emotional state>

If a section does not contain enough information, write:

Not identified.
""".strip()


# ---------------------------------------------------------------------
# Fase 1 (V2) — descritores visuais estruturados
#
# Em vez de só um texto livre, o modelo retorna três blocos marcados:
#   CHARACTERS_JSON     - lista de personagens com aparência estruturada
#                          (estável vs temporária), pra permitir matching
#                          entre cenas mais pra frente (Fase 3)
#   SCENE_EVENTS_JSON   - quem aparece/entra/sai, ações, objetos, mudanças
#   SUMMARY             - um resumo em texto corrido (mantém compatibilidade
#                          com o roteiro final e a síntese, que continuam
#                          consumindo texto simples)
#
# "indeterminado" é usado no lugar de qualquer característica que não
# possa ser confirmada visualmente — nunca se preenche por chute.
# ---------------------------------------------------------------------

DEFAULT_QUESTION = """
You are a visual analyst of audiovisual scenes.

Analyze ALL the provided frames TOGETHER as a single scene. Do not treat
each frame as an independent image.

Your goal is to identify and describe the characters visually present, and
record characteristics that could later be used to recognize the same
character in other scenes.

IMPORTANT RULES:

1. Describe only characteristics that are visually observable in the frames.
2. Never invent characteristics that are not visible.
3. If a characteristic cannot be determined, use "indeterminado".
4. Do not use outside knowledge about the movie, anime, series, or characters
   to figure out names.
5. Do not assign a name to a character just because their appearance
   resembles a known character.
6. A name should only be used if it is explicitly established by the
   dialogue below.
7. Distinguish stable characteristics (hair, face shape, skin tone, build,
   distinctive features) from temporary characteristics (clothing,
   accessories, injuries, dirt, items carried).
8. A change of clothing does not necessarily mean a change of character.
9. Consider all frames together to identify persistent characteristics.
10. If two characters look visually similar, carefully record the
    characteristics that allow telling them apart.
11. Do not turn visual characteristics into inferences about race,
    ethnicity, personality, or other non-observable traits.
12. Do not confuse a temporary characteristic with a permanent physical one.
13. When there is little visual evidence, prefer "indeterminado" over a
    speculative statement.

The dialogue spoken during this scene is (context only — do not assume
something happened visually just because the dialogue says so):

{dialogue}

Return your answer in EXACTLY this format, with these three marked blocks
and nothing else outside them:

CHARACTERS_JSON:
<a JSON array, one object per visually identifiable character, in this shape:
[
  {{
    "character_ref": "personagem_A",
    "name": "<exact name ONLY if it appears in the dialogue above, else null>",
    "stable_appearance": {{
      "skin_tone": "<or indeterminado>",
      "hair_color": "<or indeterminado>",
      "hair_length": "<or indeterminado>",
      "hair_style": "<or indeterminado>",
      "face_shape": "<or indeterminado>",
      "facial_hair": "<or indeterminado>",
      "build": "<or indeterminado>",
      "distinctive_features": ["<visible distinctive trait>", "..."]
    }},
    "temporary_appearance": {{
      "clothing": ["<visible clothing item>", "..."],
      "accessories": ["<visible accessory>", "..."],
      "temporary_features": ["<e.g. injury, dirt, item carried>", "..."]
    }}
  }}
]
Use an empty array [] if no character is clearly identifiable.>

SCENE_EVENTS_JSON:
<a JSON object in this shape:
{{
  "characters_present": ["personagem_A", "personagem_B"],
  "enters_or_exits": ["<e.g. personagem_A enters the room>"],
  "actions": ["<physical action performed by a character>"],
  "objects_used": ["<object and who uses it>"],
  "visual_changes": ["<relevant change during the scene>"]
}}
Use empty arrays if nothing applicable.>

SUMMARY:
<a short, objective paragraph in {language}, describing what happens in
the scene in plain language for a human-readable script. Follow the same
grounding rules above: no invented names, no invented events, use neutral
labels when names are unknown. This paragraph will be used on its own, so
it must make sense without the JSON blocks above.>
""".strip()

def _load_qwen_model(
    model_name: str,
    load_in_4bit: bool = False,
    attn_implementation: str = "sdpa",
):
    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )

    print(f"[scene_analysis] Carregando {model_name} " f"(4bit={load_in_4bit})...")

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        # "sdpa" funciona em CUDA e ROCm sem depender do pacote
        # flash-attn (que não tem build pra ROCm).
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

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


import json
import re


def _extract_json_block(raw: str, start_marker: str, end_marker: str | None) -> str | None:
    """Extrai o texto entre dois marcadores (ou até o fim, se end_marker não aparecer)."""
    start_idx = raw.find(start_marker)
    if start_idx == -1:
        return None
    start_idx += len(start_marker)

    if end_marker:
        end_idx = raw.find(end_marker, start_idx)
        if end_idx == -1:
            return raw[start_idx:].strip()
        return raw[start_idx:end_idx].strip()

    return raw[start_idx:].strip()


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` ou ``` ... ``` se o modelo envolver o JSON em fence."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


def parse_structured_response(raw: str) -> tuple[list[dict], dict, str]:
    """
    Faz o parsing da resposta em três blocos (CHARACTERS_JSON,
    SCENE_EVENTS_JSON, SUMMARY). Tolerante a falhas: se o JSON vier
    malformado ou os marcadores não aparecerem, cai de volta pro texto
    bruto como resumo, sem derrubar o pipeline.

    Retorna (characters, scene_events, summary).
    """
    characters: list[dict] = []
    scene_events: dict = {}
    summary = raw.strip()

    chars_block = _extract_json_block(raw, "CHARACTERS_JSON:", "SCENE_EVENTS_JSON:")
    events_block = _extract_json_block(raw, "SCENE_EVENTS_JSON:", "SUMMARY:")
    summary_block = _extract_json_block(raw, "SUMMARY:", None)

    if chars_block:
        try:
            characters = json.loads(_strip_code_fence(chars_block))
            if not isinstance(characters, list):
                characters = []
        except (json.JSONDecodeError, ValueError):
            print("[scene_analysis]   aviso: CHARACTERS_JSON malformado, ignorando.")
            characters = []

    if events_block:
        try:
            scene_events = json.loads(_strip_code_fence(events_block))
            if not isinstance(scene_events, dict):
                scene_events = {}
        except (json.JSONDecodeError, ValueError):
            print("[scene_analysis]   aviso: SCENE_EVENTS_JSON malformado, ignorando.")
            scene_events = {}

    if summary_block:
        summary = summary_block

    return characters, scene_events, summary


def _describe_scene(
    model,
    processor,
    frame_paths: list[str],
    dialogue: str,
    question_template: str,
    language: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int | None,
    do_sample: bool,
) -> str:

    from qwen_vl_utils import process_vision_info

    question = question_template.format(
        dialogue=dialogue,
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

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
    }
    # no_repeat_ngram_size bane a sequência de N tokens em TODA a
    # entrada (prompt + gerado) — com diálogos longos/repetitivos isso
    # pode banir o próprio nome de um personagem que já apareceu antes
    # no diálogo, corrompendo a grafia. Só ativa se explicitamente
    # configurado com um valor.
    if no_repeat_ngram_size:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
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


def group_into_scenes(
    segments: list[dict],
    gap_threshold_seconds: float = 3.0,
) -> list[dict]:

    if not segments:
        return []

    scenes = []

    current = [segments[0]]

    for segment in segments[1:]:

        gap = segment["start"] - current[-1]["end"]

        if gap > gap_threshold_seconds:
            scenes.append(current)
            current = [segment]
        else:
            current.append(segment)

    scenes.append(current)

    return [
        {
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "segments": group,
        }
        for group in scenes
    ]


def describe_scenes(
    scenes: list[dict],
    video_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    frames_per_scene: int = 12,
    padding_seconds: float = 0.5,
    output_dir: str = "temp/frames_cenas",
    question: str = DEFAULT_QUESTION,
    language: str = "pt-BR",
    max_new_tokens: int = 400,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int | None = None,
    do_sample: bool = False,
    load_in_4bit: bool = False,
) -> list[dict]:

    model, processor = _load_qwen_model(
        model_name,
        load_in_4bit=load_in_4bit,
    )

    print(f"[scene_analysis] Modelo carregado. " f"Analisando {len(scenes)} cenas...")

    results = []

    try:

        for idx, scene in enumerate(scenes):

            start = scene["start"]
            end = scene["end"]

            duration = max(
                end - start,
                0.1,
            )

            if frames_per_scene == 1:

                timestamps = [start + duration / 2]

            else:

                timestamps = [
                    start
                    - padding_seconds
                    + (duration + 2 * padding_seconds) * i / (frames_per_scene - 1)
                    for i in range(frames_per_scene)
                ]

            timestamps = [max(timestamp, 0) for timestamp in timestamps]

            frame_paths = []

            for i, timestamp in enumerate(timestamps):

                frame_path = str(Path(output_dir) / f"cena{idx:04d}_{i}.jpg")

                try:

                    _extract_frame_at(
                        video_path,
                        timestamp,
                        frame_path,
                    )

                    frame_paths.append(frame_path)

                except RuntimeError as error:

                    print(
                        "[scene_analysis] "
                        f"Erro ao extrair frame "
                        f"{timestamp:.1f}s: {error}"
                    )

            dialogue = "\n".join(
                f"{segment.get('speaker', '?').upper()}: " f"{segment.get('text', '')}"
                for segment in scene["segments"]
            )

            description = ""
            visual_descriptors: list[dict] = []
            scene_events: dict = {}

            if frame_paths:

                try:

                    raw_response = _describe_scene(
                        model=model,
                        processor=processor,
                        frame_paths=frame_paths,
                        dialogue=dialogue,
                        question_template=question,
                        language=language,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        do_sample=do_sample,
                    )

                    visual_descriptors, scene_events, description = parse_structured_response(
                        raw_response
                    )

                except Exception as error:

                    print("[scene_analysis] " f"Erro ao analisar cena: {error}")

            results.append(
                {
                    "timestamp": start,
                    "end": end,
                    "description": description,
                    "dialogue": dialogue,
                    "visual_descriptors": visual_descriptors,
                    "scene_events": scene_events,
                }
            )

            print(
                f"[scene_analysis] Cena "
                f"{idx + 1}/{len(scenes)} "
                f"({start:.1f}s - {end:.1f}s)"
            )

            if description:
                print(description)

    finally:

        del model
        del processor

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def describe_dialogue_scenes(
    segments: list[dict],
    video_path: str,
    **kwargs,
) -> list[dict]:

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