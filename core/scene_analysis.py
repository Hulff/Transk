"""
Análise de cena/ações usando um modelo multimodal (Qwen2.5-VL).

Diferente do BLIP-2 (legendagem isolada por frame), o Qwen2.5-VL recebe
VÁRIOS frames + o diálogo completo daquela janela de tempo no mesmo
prompt, e consegue raciocinar sobre o que está acontecendo de verdade
(ação, gesto, expressão, interação), não só descrever pixels soltos.

Requer GPU com VRAM suficiente (recomendado: Colab com GPU, ou GPU
local com pelo menos ~16GB pra rodar em float16; use load_in_4bit=True
no config pra caber em GPUs menores).
"""

import subprocess
from pathlib import Path

from core.ffmpeg_utils import get_ffmpeg_path

# ---------------------------------------------------------------------
# Extração de frames
# ---------------------------------------------------------------------


def extract_frames(
    video_path: str, output_dir: str, interval_seconds: int = 5
) -> list[dict]:
    """Extrai um frame a cada N segundos do vídeo (modo "intervalo", legado)."""
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao extrair frames:\n{result.stderr}")

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [
        {"timestamp": i * interval_seconds, "frame_path": str(f)}
        for i, f in enumerate(frames)
    ]


def _extract_frame_at(video_path: str, timestamp: float, output_path: str):
    """Extrai um único frame no timestamp exato (segundos), via seek do ffmpeg."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao extrair frame em {timestamp}s:\n{result.stderr}")


# ---------------------------------------------------------------------
# Modelo multimodal (Qwen2.5-VL)
# ---------------------------------------------------------------------

DEFAULT_QUESTION = (
    "Analise os frames do vídeo em conjunto.\n\n"
    "O diálogo falado durante esta cena é:\n\n"
    "{dialogue}\n\n"
    "Descreva a situação visível em UMA única frase concisa.\n\n"
    "Concentre-se em:\n"
    "- personagem principal visível;\n"
    "- expressão facial;\n"
    "- postura corporal ou gesto;\n"
    "- ação física;\n"
    "- direção do olhar;\n"
    "- interação com outros personagens visíveis.\n\n"
    "Use o diálogo apenas como informação de contexto.\n"
    "Não presuma que algo aconteceu apenas porque o diálogo diz que aconteceu.\n\n"
    "Descreva somente aquilo que pode ser confirmado visualmente pelos frames.\n\n"
    "NÃO invente:\n"
    "- nomes de personagens;\n"
    "- ações;\n"
    "- emoções;\n"
    "- objetos;\n"
    "- acontecimentos;\n"
    "- relacionamentos;\n"
    "- locais.\n\n"
    "Não descreva o fundo ou o cenário, a menos que seja relevante para a ação.\n\n"
    "Se houver vários personagens visíveis, descreva a interação observável entre eles.\n\n"
    "Responda exclusivamente em {language}.\n\n"
    "Retorne SOMENTE a descrição visual final.\n"
    "Não mencione o diálogo.\n"
    "Não explique seu raciocínio.\n"
    "Não use tópicos ou listas."
)


def _load_qwen_model(model_name: str, load_in_4bit: bool = False):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"[scene_analysis] Carregando {model_name} (4bit={load_in_4bit})...")

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
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

    question = question_template.format(dialogue=dialogue, language=language)

    content = [{"type": "image", "image": p} for p in frame_paths]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
        out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


# ---------------------------------------------------------------------
# Modo "fala" — ancorado em cada linha de diálogo
# ---------------------------------------------------------------------


def describe_dialogue_scenes(
    segments: list[dict],
    video_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    frames_per_fala: int = 5,
    padding_seconds: float = 0.5,
    output_dir: str = "temp/frames_falas",
    question: str = DEFAULT_QUESTION,
    language: str = "pt-BR",
    max_new_tokens: int = 80,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = False,
    load_in_4bit: bool = False,
) -> list[dict]:
    """
    Para cada fala transcrita, extrai `frames_per_fala` frames dentro da
    janela de tempo dela e pede pro Qwen2.5-VL descrever a cena
    combinando todos esses frames + o texto da fala no mesmo prompt.

    Retorna lista de {"timestamp": start_da_fala, "description": texto}.
    """
    model, processor = _load_qwen_model(model_name, load_in_4bit=load_in_4bit)
    print(
        f"[scene_analysis] Modelo carregado. Analisando cena de {len(segments)} falas..."
    )

    scenes = []
    for idx, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        duration = max(end - start, 0.1)

        if frames_per_fala == 1:
            timestamps = [start + duration / 2]
        else:
            timestamps = [
                start
                - padding_seconds
                + (duration + 2 * padding_seconds) * i / (frames_per_fala - 1)
                for i in range(frames_per_fala)
            ]
        timestamps = [max(t, 0) for t in timestamps]

        frame_paths = []
        for i, t in enumerate(timestamps):
            frame_path = str(Path(output_dir) / f"seg{idx:04d}_{i}.jpg")
            try:
                _extract_frame_at(video_path, t, frame_path)
                frame_paths.append(frame_path)
            except RuntimeError as e:
                print(f"[scene_analysis]   erro ao extrair frame em {t:.1f}s: {e}")
                continue

        dialogue = f"{seg.get('speaker', '?').upper()}: {seg['text']}"

        description = ""
        if frame_paths:
            try:
                description = _describe_scene(
                    model,
                    processor,
                    frame_paths,
                    dialogue,
                    question,
                    language=language,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                )
            except Exception as e:
                print(f"[scene_analysis]   erro ao gerar descrição: {e}")

        scenes.append({"timestamp": start, "description": description})
        print(
            f"[scene_analysis] Fala {idx+1}/{len(segments)} ({start}s): {description or '(sem descrição)'}"
        )

    return scenes


# ---------------------------------------------------------------------
# Modo "cena" — agrupa falas consecutivas e analisa tudo de uma vez
# ---------------------------------------------------------------------


def group_into_scenes(
    segments: list[dict], gap_threshold_seconds: float = 3.0
) -> list[dict]:
    """
    Agrupa falas consecutivas em cenas. Uma nova cena começa sempre que
    o intervalo de silêncio entre o fim de uma fala e o início da
    próxima ultrapassa `gap_threshold_seconds`.

    Retorna lista de {"start", "end", "segments": [...]}.
    """
    if not segments:
        return []

    scenes = []
    current = [segments[0]]

    for seg in segments[1:]:
        gap = seg["start"] - current[-1]["end"]
        if gap > gap_threshold_seconds:
            scenes.append(current)
            current = [seg]
        else:
            current.append(seg)
    scenes.append(current)

    return [
        {"start": group[0]["start"], "end": group[-1]["end"], "segments": group}
        for group in scenes
    ]


def describe_scenes(
    scenes: list[dict],
    video_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    frames_per_scene: int = 8,
    padding_seconds: float = 0.5,
    output_dir: str = "temp/frames_cenas",
    question: str = DEFAULT_QUESTION,
    language: str = "pt-BR",
    max_new_tokens: int = 120,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = False,
    load_in_4bit: bool = False,
) -> list[dict]:
    """
    Para cada cena (grupo de falas agrupadas por group_into_scenes),
    extrai `frames_per_scene` frames espalhados pela duração INTEIRA da
    cena e pede pro Qwen2.5-VL descrever a cena combinando TODOS esses
    frames + TODAS as falas da cena (diálogo completo) no mesmo prompt.

    Retorna lista de {"timestamp": start_da_cena, "description": texto},
    compatível com merge_timeline (mesmo formato usado pelo modo "fala").
    """
    model, processor = _load_qwen_model(model_name, load_in_4bit=load_in_4bit)
    print(f"[scene_analysis] Modelo carregado. Analisando {len(scenes)} cenas...")

    results = []
    for idx, scene in enumerate(scenes):
        start, end = scene["start"], scene["end"]
        duration = max(end - start, 0.1)

        if frames_per_scene == 1:
            timestamps = [start + duration / 2]
        else:
            timestamps = [
                start
                - padding_seconds
                + (duration + 2 * padding_seconds) * i / (frames_per_scene - 1)
                for i in range(frames_per_scene)
            ]
        timestamps = [max(t, 0) for t in timestamps]

        frame_paths = []
        for i, t in enumerate(timestamps):
            frame_path = str(Path(output_dir) / f"cena{idx:04d}_{i}.jpg")
            try:
                _extract_frame_at(video_path, t, frame_path)
                frame_paths.append(frame_path)
            except RuntimeError as e:
                print(f"[scene_analysis]   erro ao extrair frame em {t:.1f}s: {e}")
                continue

        # diálogo completo da cena, todas as falas na ordem
        dialogue = "\n".join(
            f"{seg.get('speaker', '?').upper()}: {seg['text']}"
            for seg in scene["segments"]
        )

        description = ""
        if frame_paths:
            try:
                description = _describe_scene(
                    model,
                    processor,
                    frame_paths,
                    dialogue,
                    question,
                    language=language,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                )
            except Exception as e:
                print(f"[scene_analysis]   erro ao gerar descrição: {e}")

        results.append({"timestamp": start, "description": description})
        print(
            f"[scene_analysis] Cena {idx+1}/{len(scenes)} "
            f"({start:.1f}s-{end:.1f}s, {len(scene['segments'])} falas, {len(frame_paths)} frames): "
            f"{description or '(sem descrição)'}"
        )

    return results
