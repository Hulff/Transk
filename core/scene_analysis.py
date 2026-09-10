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


DEFAULT_QUESTION = """
Analise cuidadosamente todos os frames desta cena em conjunto.

O diálogo transcrito é:

{dialogue}

Sua tarefa é produzir uma descrição audiovisual objetiva do que
acontece nesta parte do vídeo.

REGISTRE PRINCIPALMENTE:

- quem aparece;
- onde a cena acontece, se for relevante;
- ações físicas realizadas pelos personagens;
- interações entre personagens;
- objetos que são segurados, usados, retirados, colocados ou destruídos;
- movimentos importantes;
- ataques, quedas, golpes, explosões ou outros acontecimentos;
- entradas e saídas de personagens;
- mudanças importantes na situação;
- expressões faciais e estados aparentes;
- acontecimentos visuais importantes mesmo quando não há diálogo.

Exemplos de acontecimentos que devem ser descritos quando observados:

"o personagem pega uma espada"

"o personagem abre a porta"

"o personagem derruba o outro no chão"

"o personagem pisa sobre a cabeça do Android"

"um personagem começa a correr"

"uma explosão destrói parte do cenário"

"o personagem entra na sala"

"o personagem olha para outro personagem"

NÃO descreva apenas uma imagem estática.

Observe a sequência dos frames e tente identificar ações que acontecem
ao longo da cena.

Não invente acontecimentos.

O diálogo serve apenas como contexto.
Não diga que uma ação aconteceu apenas porque alguém falou sobre ela.

Se uma ação não puder ser confirmada visualmente, não a apresente como
um fato.

Não invente nomes de personagens.
Quando o nome estiver disponível no diálogo, ele pode ser usado para
identificar um personagem visualmente correspondente.

Não invente locais, objetos, relações ou acontecimentos.

Evite interpretações narrativas exageradas.

Em vez de:

"Cell tenta demonstrar sua superioridade porque odeia Android 17."

prefira:

"Cell permanece sobre Android 17 e pisa sobre sua cabeça enquanto fala."

Descreva acontecimentos concretos.

Retorne somente o texto final da descrição.

Escreva em {language}.

Formato:

CONTEXTO
<descrição geral e objetiva da situação>

PERSONAGENS
- <personagem>: <estado ou ação relevante>

AMBIENTE
<ambiente relevante para compreender a cena>

ACONTECIMENTOS
- <acontecimento visual>
- <acontecimento visual>

AÇÕES
- <ação realizada>
- <ação realizada>

FALAS IMPORTANTES
- <personagem>: <fala ou resumo>

ESTADO / EMOÇÕES OBSERVÁVEIS
- <personagem>: <estado aparente>

Não identificado.
""".strip()


def _load_qwen_model(
    model_name: str,
    load_in_4bit: bool = False,
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


def _describe_scene(
    model,
    processor,
    frame_paths: list[str],
    dialogue: str,
    question_template: str,
    language: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
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

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
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
    no_repeat_ngram_size: int = 3,
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

            if frame_paths:

                try:

                    description = _describe_scene(
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

                except Exception as error:

                    print("[scene_analysis] " f"Erro ao analisar cena: {error}")

            results.append(
                {
                    "timestamp": start,
                    "end": end,
                    "description": description,
                    "dialogue": dialogue,
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
