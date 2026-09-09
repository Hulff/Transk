"""
Análise contextual de cenas usando Qwen2.5-VL.

O modelo recebe:
    - o diálogo da fala;
    - 1 a 3 frames da cena.

E retorna UMA descrição visual contextual.

O objetivo é descrever:
    - personagem principal visível;
    - expressão facial;
    - postura;
    - ação;
    - direção do olhar;
    - interação com outros personagens.

Toda a configuração pode ser controlada pelo config.yaml.
"""

from pathlib import Path
import re
import subprocess


# ================================================================
# CONFIGURAÇÕES PADRÃO
# ================================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

DEFAULT_LANGUAGE = "en-US"

DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_DO_SAMPLE = False

DEFAULT_QUESTION = """
Analise os frames do vídeo em conjunto.

O diálogo falado durante esta cena é:

{dialogue}

Descreva a situação visível em UMA única frase concisa.

Concentre-se em:
- personagem principal visível;
- expressão facial;
- postura corporal ou gesto;
- ação física;
- direção do olhar;
- interação com outros personagens visíveis.

Use o diálogo apenas como informação contextual.

Não assuma que algo aconteceu apenas porque o diálogo diz que aconteceu.

Descreva somente aquilo que é visualmente sustentado pelos frames.

Não invente:
- nomes de personagens;
- ações;
- emoções;
- objetos;
- acontecimentos;
- relacionamentos;
- locais.

Não descreva o fundo, a menos que ele seja relevante para a ação.

Se houver vários personagens visíveis, descreva a interação que pode ser observada.

Responda exclusivamente em {language}.

Retorne SOMENTE a descrição visual final.
Não repita esta instrução.
Não mencione o diálogo.
Não explique seu raciocínio.
Não use tópicos ou listas.
"""


# ================================================================
# FFMPEG
# ================================================================

def _extract_frame_at(
    video_path: str,
    timestamp: float,
    output_path: str,
):
    """
    Extrai um frame específico do vídeo.
    """

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        "ffmpeg",
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


# ================================================================
# MODELO
# ================================================================

def _load_qwen_model(
    model_name: str = DEFAULT_MODEL,
):
    """
    Carrega Qwen2.5-VL.

    Usa transformers + Qwen2_5_VLForConditionalGeneration.
    """

    import torch

    print(
        f"[scene_analysis] Carregando modelo "
        f"{model_name}..."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"[scene_analysis] device={device}"
    )

    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )

    if device == "cuda":

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    else:

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_name
    )

    model.eval()

    print(
        "[scene_analysis] Modelo Qwen2.5-VL carregado."
    )

    return processor, model, device


# ================================================================
# LIMPEZA
# ================================================================

def _clean_caption(text: str) -> str:
    """
    Limpa a resposta produzida pelo modelo.

    Importante:
    NÃO retorna a pergunta.
    """

    if not text:
        return ""

    text = text.strip()

    # ------------------------------------------------------------
    # Remove blocos de raciocínio
    # ------------------------------------------------------------

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"<\|im_start\|>.*?<\|im_end\|>",
        "",
        text,
        flags=re.DOTALL,
    )

    # ------------------------------------------------------------
    # Remove prefixos comuns
    # ------------------------------------------------------------

    prefixes = [
        "Answer:",
        "answer:",
        "Description:",
        "description:",
        "Visual description:",
        "visual description:",
        "Scene:",
        "scene:",
        "Resposta:",
        "resposta:",
        "Descrição:",
        "descrição:",
        "Descrição visual:",
        "descrição visual:",
        "Cena:",
        "cena:",
    ]

    for prefix in prefixes:

        if text.startswith(prefix):

            text = text[len(prefix):].strip()

    # ------------------------------------------------------------
    # Remove aspas externas
    # ------------------------------------------------------------

    if (
        len(text) >= 2
        and text[0] == '"'
        and text[-1] == '"'
    ):
        text = text[1:-1].strip()

    if (
        len(text) >= 2
        and text[0] == "'"
        and text[-1] == "'"
    ):
        text = text[1:-1].strip()

    # ------------------------------------------------------------
    # Normaliza espaços
    # ------------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # ------------------------------------------------------------
    # Frases inválidas
    # ------------------------------------------------------------

    invalid_phrases = [

        "use the following examples",
        "no more than two sentences",
        "use a single word or phrase",

        "return only the visual description",
        "describe the visible scene",
        "who is the main visible character",
        "focus on the main visible character",

        "the dialogue is",
        "the character speaking is",

        "question:",
        "answer:",

        "png file",
        "dragon ball z png",

        "[1]",
        "[2]",
        "[3]",

        "retorne somente",
        "não mencione o diálogo",
        "não explique seu raciocínio",
    ]

    lowered = text.lower()

    for phrase in invalid_phrases:

        if phrase in lowered:
            return ""

    # ------------------------------------------------------------
    # Evita lixo muito curto
    # ------------------------------------------------------------

    if len(text) < 8:
        return ""

    # ------------------------------------------------------------
    # Evita respostas extremamente longas
    # ------------------------------------------------------------

    if len(text) > 500:

        text = text[:500]

        last_dot = text.rfind(".")

        if last_dot > 50:
            text = text[:last_dot + 1]

    return text.strip()


# ================================================================
# DETECTOR DE RESPOSTA RUIM
# ================================================================

def _is_degenerate(text: str) -> bool:
    """
    Detecta respostas claramente quebradas.
    """

    if not text:
        return True

    lowered = text.lower()

    bad_patterns = [

        "png file",
        "dragon ball z png",

        "use the following examples",
        "use a single word",

        "no more than two sentences",

        "question:",
        "answer:",

        "[1]",
        "[2]",
        "[3]",

        "describe the visible scene",
        "who is the main visible character",

        "retorne somente",
        "não mencione o diálogo",
        "não explique seu raciocínio",
    ]

    for pattern in bad_patterns:

        if pattern in lowered:
            return True

    # ------------------------------------------------------------
    # Repetição de palavras
    # ------------------------------------------------------------

    words = re.findall(
        r"\b\w+\b",
        lowered,
    )

    if len(words) >= 6:

        from collections import Counter

        counts = Counter(words)

        _, frequency = counts.most_common(1)[0]

        if frequency >= 4:
            return True

    # ------------------------------------------------------------
    # Repetição de bigramas
    # ------------------------------------------------------------

    if len(words) >= 6:

        bigrams = [
            f"{words[i]} {words[i + 1]}"
            for i in range(len(words) - 1)
        ]

        from collections import Counter

        counts = Counter(bigrams)

        if counts.most_common(1)[0][1] >= 3:
            return True

    return False


# ================================================================
# QWEN - ANÁLISE DOS FRAMES
# ================================================================

def _analyze_frames(
    processor,
    model,
    device: str,
    frame_paths: list[str],
    dialogue: str,
    question: str = DEFAULT_QUESTION,
    language: str = DEFAULT_LANGUAGE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    do_sample: bool = DEFAULT_DO_SAMPLE,
) -> str:
    """
    Envia vários frames simultaneamente para o Qwen2.5-VL.
    """

    from PIL import Image
    import torch

    images = []

    # ------------------------------------------------------------
    # Abre os frames
    # ------------------------------------------------------------

    for path in frame_paths:

        try:

            image = Image.open(path).convert(
                "RGB"
            )

            images.append(image)

        except Exception as e:

            print(
                f"[scene_analysis] "
                f"erro ao abrir frame {path}: {e}"
            )

    if not images:
        return ""

    # ------------------------------------------------------------
    # Monta prompt
    # ------------------------------------------------------------

    try:

        prompt = question.format(
            dialogue=dialogue,
            language=language,
        )

    except KeyError as e:

        raise ValueError(
            f"Variável desconhecida no question do config.yaml: {e}. "
            f"Use apenas {{dialogue}} e {{language}}."
        )

    # ------------------------------------------------------------
    # Conteúdo multimodal
    # ------------------------------------------------------------

    content = []

    for image in images:

        content.append(
            {
                "type": "image",
                "image": image,
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

    # ------------------------------------------------------------
    # Template Qwen
    # ------------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )

    # ------------------------------------------------------------
    # Move tensores para dispositivo
    # ------------------------------------------------------------

    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    # ------------------------------------------------------------
    # Geração
    # ------------------------------------------------------------

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,

            max_new_tokens=max_new_tokens,

            repetition_penalty=repetition_penalty,

            no_repeat_ngram_size=no_repeat_ngram_size,

            do_sample=do_sample,
        )

    # ------------------------------------------------------------
    # Remove tokens do prompt
    # ------------------------------------------------------------

    input_ids = inputs.get("input_ids")

    if input_ids is not None:

        generated_ids_trimmed = [

            output_ids[len(input_ids[i]):]

            for i, output_ids in enumerate(
                generated_ids
            )
        ]

    else:

        generated_ids_trimmed = generated_ids

    # ------------------------------------------------------------
    # Decodifica
    # ------------------------------------------------------------

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
    )[0]

    return _clean_caption(
        output_text
    )


# ================================================================
# ANÁLISE DE UMA FALA
# ================================================================

def _analyze_dialogue_segment(
    processor,
    model,
    device,
    video_path: str,
    seg: dict,
    index: int,
    frames_per_fala: int,
    padding_seconds: float,
    output_dir: str,
    question: str,
    language: str = DEFAULT_LANGUAGE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    do_sample: bool = DEFAULT_DO_SAMPLE,
) -> str:

    start = float(
        seg.get("start", 0)
    )

    end = float(
        seg.get(
            "end",
            start + 1.0,
        )
    )

    duration = max(
        end - start,
        0.1,
    )

    # ------------------------------------------------------------
    # Escolhe timestamps
    # ------------------------------------------------------------

    if frames_per_fala <= 1:

        timestamps = [
            start + duration / 2
        ]

    else:

        effective_start = max(
            0,
            start - padding_seconds,
        )

        effective_end = (
            end + padding_seconds
        )

        if effective_end <= effective_start:

            effective_end = (
                effective_start + 0.1
            )

        step = (
            effective_end - effective_start
        ) / (frames_per_fala - 1)

        timestamps = [

            effective_start + step * i

            for i in range(
                frames_per_fala
            )
        ]

    # ------------------------------------------------------------
    # Extrai frames
    # ------------------------------------------------------------

    frame_paths = []

    for frame_index, timestamp in enumerate(
        timestamps
    ):

        frame_path = str(
            Path(output_dir)
            / f"seg{index:04d}_{frame_index}.jpg"
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

        except Exception as e:

            print(
                f"[scene_analysis] "
                f"erro extraindo frame: {e}"
            )

    if not frame_paths:
        return ""

    # ------------------------------------------------------------
    # Diálogo
    # ------------------------------------------------------------

    dialogue = str(
        seg.get("text", "")
    ).strip()

    # ------------------------------------------------------------
    # Analisa frames
    # ------------------------------------------------------------

    description = _analyze_frames(
        processor=processor,
        model=model,
        device=device,
        frame_paths=frame_paths,
        dialogue=dialogue,
        question=question,
        language=language,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
    )

    return description


# ================================================================
# MODO FALA
# ================================================================

def describe_dialogue_scenes(
    segments: list[dict],
    video_path: str,
    model_name: str = DEFAULT_MODEL,
    frames_per_fala: int = 3,
    padding_seconds: float = 0.25,
    output_dir: str = "temp/frames_falas",
    question: str = DEFAULT_QUESTION,
    language: str = DEFAULT_LANGUAGE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    do_sample: bool = DEFAULT_DO_SAMPLE,
) -> list[dict]:
    """
    Analisa cada fala usando Qwen2.5-VL.

    Retorna:

        [
            {
                "timestamp": 2.84,
                "description": "..."
            }
        ]

    A descrição nunca deve conter a pergunta.
    """

    processor, model, device = _load_qwen_model(
        model_name
    )

    print(
        f"[scene_analysis] "
        f"Modelo carregado. "
        f"Idioma={language}. "
        f"Analisando {len(segments)} falas..."
    )

    scenes = []

    for index, seg in enumerate(
        segments
    ):

        start = float(
            seg.get("start", 0)
        )

        end = float(
            seg.get(
                "end",
                start + 1,
            )
        )

        speaker = seg.get(
            "speaker",
            "",
        )

        dialogue = seg.get(
            "text",
            "",
        )

        print()
        print("=" * 70)

        print(
            f"[scene_analysis] "
            f"Fala {index + 1}/{len(segments)}"
        )

        print(
            f"[scene_analysis] "
            f"tempo: {start:.2f}s -> {end:.2f}s"
        )

        print(
            f"[scene_analysis] "
            f"speaker: {speaker}"
        )

        print(
            f"[scene_analysis] "
            f"diálogo: {dialogue}"
        )

        print(
            f"[scene_analysis] "
            f"idioma: {language}"
        )

        print(
            f"[scene_analysis] "
            f"frames: {frames_per_fala}"
        )

        try:

            description = _analyze_dialogue_segment(
                processor=processor,
                model=model,
                device=device,
                video_path=video_path,
                seg=seg,
                index=index,
                frames_per_fala=frames_per_fala,
                padding_seconds=padding_seconds,
                output_dir=output_dir,
                question=question,
                language=language,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
            )

        except Exception as e:

            print(
                f"[scene_analysis] "
                f"erro na análise: {e}"
            )

            description = ""

        # --------------------------------------------------------
        # Validação
        # --------------------------------------------------------

        if _is_degenerate(
            description
        ):

            print(
                "[scene_analysis] "
                "✗ resposta inválida"
            )

            description = ""

        else:

            print(
                "[scene_analysis] "
                f"✓ {description}"
            )

        scenes.append(
            {
                "timestamp": start,
                "description": description,
            }
        )

        print(
            "[scene_analysis] "
            "RESULTADO FINAL:"
        )

        print(
            description
            if description
            else "(sem descrição útil)"
        )

        print("=" * 70)

    return scenes


# ================================================================
# MODO INTERVALO
# ================================================================

def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: int = 5,
) -> list[dict]:
    """
    Extrai frames a cada N segundos.
    """

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    pattern = str(
        Path(output_dir)
        / "frame_%05d.jpg"
    )

    cmd = [
        "ffmpeg",
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
            f"Erro ao extrair frames:\n"
            f"{result.stderr}"
        )

    frames = sorted(
        Path(output_dir).glob(
            "frame_*.jpg"
        )
    )

    return [
        {
            "timestamp":
                i * interval_seconds,

            "frame_path":
                str(frame),
        }

        for i, frame in enumerate(
            frames
        )
    ]


# ================================================================
# DESCRIÇÃO DE FRAMES INDIVIDUAIS
# ================================================================

def describe_frames(
    frames: list[dict],
    model_name: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    question: str | None = None,
) -> list[dict]:
    """
    Descreve frames individuais.

    Mantido para compatibilidade com o pipeline antigo.

    Também respeita idioma e parâmetros de geração.
    """

    processor, model, device = _load_qwen_model(
        model_name
    )

    print(
        f"[scene_analysis] "
        f"Descrevendo {len(frames)} frames..."
    )

    if question is None:

        question = """
Descreva a imagem visível em UMA única frase concisa.

Concentre-se no personagem principal visível,
expressão facial, postura corporal,
ação física, direção do olhar
e interação com outros personagens.

Descreva somente aquilo que é visualmente sustentado pela imagem.

Não invente nomes, acontecimentos ou ações.

Responda exclusivamente em {language}.

Retorne somente a descrição visual.
"""

    for index, frame in enumerate(
        frames
    ):

        try:

            description = _analyze_frames(
                processor=processor,
                model=model,
                device=device,
                frame_paths=[
                    frame["frame_path"]
                ],
                dialogue="",
                question=question,
                language=language,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
            )

        except Exception as e:

            print(
                f"[scene_analysis] "
                f"erro: {e}"
            )

            description = ""

        if _is_degenerate(
            description
        ):

            description = ""

        frame["description"] = (
            description
        )

        print(
            f"[scene_analysis] "
            f"Frame {index + 1}/{len(frames)}: "
            f"{description or '(sem descrição)'}"
        )

    return frames
