"""
Ferramenta pra montar um dataset de exemplos corrigidos manualmente,
a partir do que o pipeline já processou (cache de cenas: frames +
diálogo + descrição gerada pelo modelo).

Cada exemplo guarda:
  - o diálogo da cena
  - os frames usados na análise (copiados pra pasta persistente, já
    que temp/ pode ser limpo)
  - o que o MODELO gerou (pra referência/comparação)
  - a versão CORRIGIDA por você (o "gabarito")

Isso serve tanto pra:
  - few-shot: puxar 1-5 exemplos bons e colar direto no prompt
  - fine-tuning (mais pra frente, quando tiver dataset maior): exportar
    pro formato de conversa (imagens + texto -> resposta) que ferramentas
    de fine-tuning de VLM (LLaMA-Factory, ms-swift) esperam.
"""

import json
import shutil
from pathlib import Path

from core.cache import load_cache

DATASET_DIR = Path("dataset")
DATASET_FILE = DATASET_DIR / "dataset.jsonl"
DATASET_FRAMES_DIR = DATASET_DIR / "frames"


def list_scenes(base_name: str) -> list[dict]:
    """Lista as cenas já processadas (cache) pra você escolher qual corrigir."""
    scenes = load_cache(base_name, "cenas")
    if not scenes:
        raise RuntimeError(
            f"Nenhuma cena em cache pra '{base_name}'. Rode 'python app.py cena {base_name}...' primeiro."
        )
    return scenes


def _find_scene_frames(base_name: str, scene_index: int) -> list[str]:
    """
    Localiza os arquivos de frame já extraídos pra essa cena, seguindo o
    padrão usado em scene_analysis.py (temp/frames_cenas/cena{idx:04d}_{i}.jpg).
    """
    pattern = f"cena{scene_index:04d}_*.jpg"
    frames = sorted(Path("temp/frames_cenas").glob(pattern))
    return [str(f) for f in frames]


def _existing_ids(dataset_file: Path) -> set[str]:
    if not dataset_file.exists():
        return set()
    ids = set()
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line).get("id"))
    return ids


def _build_record(
    base_name: str,
    scene_index: int,
    scene: dict,
    frames_dir: Path,
    corrected_output: str | None,
) -> dict:
    example_id = f"{base_name}_cena{scene_index:04d}"

    original_frames = _find_scene_frames(base_name, scene_index)
    example_frames_dir = frames_dir / example_id
    example_frames_dir.mkdir(parents=True, exist_ok=True)

    saved_frame_paths = []
    for i, frame_path in enumerate(original_frames):
        dest = example_frames_dir / f"frame_{i}.jpg"
        shutil.copy(frame_path, dest)
        saved_frame_paths.append(str(dest))

    return {
        "id": example_id,
        "video": base_name,
        "scene_index": scene_index,
        "start": scene.get("timestamp"),
        "end": scene.get("end"),
        "dialogue": scene.get("dialogue", ""),
        "frames": saved_frame_paths,
        "model_output": scene.get("description", ""),
        # None = ainda não revisado por humano; corrija depois com
        # 'dataset-revisar' antes de usar em few-shot ou fine-tuning.
        "corrected_output": corrected_output,
    }


def add_example(
    base_name: str,
    scene_index: int,
    corrected_output: str,
    dataset_dir: str = "dataset",
) -> str:
    """
    Adiciona um exemplo ao dataset: copia os frames da cena escolhida
    pra uma pasta persistente e grava o registro (diálogo + saída do
    modelo + sua correção) em dataset.jsonl.

    Retorna o id do exemplo criado.
    """
    dataset_dir = Path(dataset_dir)
    dataset_file = dataset_dir / "dataset.jsonl"
    frames_dir = dataset_dir / "frames"

    scenes = list_scenes(base_name)
    if scene_index < 0 or scene_index >= len(scenes):
        raise IndexError(
            f"Índice de cena inválido: {scene_index} (só há {len(scenes)} cenas)"
        )

    record = _build_record(
        base_name,
        scene_index,
        scenes[scene_index],
        frames_dir,
        corrected_output.strip(),
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record["id"]


def save_run_to_dataset(
    base_name: str,
    scenes: list[dict],
    dataset_dir: str = "dataset",
) -> list[str]:
    """
    Grava TODAS as cenas de uma execução no dataset de uma vez, sem
    correção humana ainda (`corrected_output: null` — pendente de
    revisão). Pensado pra ser chamado automaticamente no fim do
    'processar' quando a flag --save é usada.

    Pula cenas cujo id já existe no dataset (evita duplicar ao rodar
    o mesmo vídeo de novo). Retorna a lista de ids adicionados.
    """
    dataset_dir_path = Path(dataset_dir)
    dataset_file = dataset_dir_path / "dataset.jsonl"
    frames_dir = dataset_dir_path / "frames"

    dataset_dir_path.mkdir(parents=True, exist_ok=True)
    existing = _existing_ids(dataset_file)

    added_ids = []
    with open(dataset_file, "a", encoding="utf-8") as f:
        for i, scene in enumerate(scenes):
            example_id = f"{base_name}_cena{i:04d}"
            if example_id in existing:
                continue

            record = _build_record(
                base_name, i, scene, frames_dir, corrected_output=None
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            added_ids.append(record["id"])

    return added_ids


def list_pending_review(dataset_dir: str = "dataset") -> list[dict]:
    """Lista exemplos salvos automaticamente que ainda não foram corrigidos."""
    return [
        ex for ex in load_dataset(dataset_dir) if ex.get("corrected_output") is None
    ]


def update_correction(
    example_id: str,
    corrected_output: str,
    dataset_dir: str = "dataset",
) -> bool:
    """
    Atualiza o `corrected_output` de um exemplo já salvo (reescreve o
    dataset.jsonl inteiro com o registro corrigido). Retorna True se
    encontrou e atualizou, False se o id não existe.
    """
    dataset_file = Path(dataset_dir) / "dataset.jsonl"
    examples = load_dataset(dataset_dir)

    found = False
    for ex in examples:
        if ex.get("id") == example_id:
            ex["corrected_output"] = corrected_output.strip()
            found = True
            break

    if not found:
        return False

    with open(dataset_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return True


def load_dataset(dataset_dir: str = "dataset") -> list[dict]:
    """Carrega todos os exemplos já salvos."""
    dataset_file = Path(dataset_dir) / "dataset.jsonl"
    if not dataset_file.exists():
        return []
    with open(dataset_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_fewshot_snippet(dataset_dir: str = "dataset", n: int = 3) -> str:
    """
    Monta um trecho de texto com os N exemplos revisados mais
    recentes, pronto pra colar dentro do prompt (campo `question` do
    config.yaml) como demonstração few-shot — sem precisar de nenhum
    treino. Ignora exemplos ainda não corrigidos (corrected_output=null).
    """
    reviewed = [ex for ex in load_dataset(dataset_dir) if ex.get("corrected_output")]
    examples = reviewed[-n:]
    if not examples:
        return ""

    blocks = []
    for i, ex in enumerate(examples, start=1):
        blocks.append(
            f"EXAMPLE {i}\n"
            f"DIALOGUE:\n{ex['dialogue']}\n\n"
            f"CORRECT DESCRIPTION:\n{ex['corrected_output']}"
        )

    return (
        "Here are examples of dialogue and the correct scene description "
        "you should produce for similar cases:\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\n---\n\nNow analyze the actual scene below, following the same style:\n"
    )


def export_finetune_jsonl(
    dataset_dir: str = "dataset", output_path: str = "dataset/finetune.jsonl"
) -> str:
    """
    Converte o dataset pro formato de conversa (imagens + texto ->
    resposta) usado por ferramentas de fine-tuning de VLM como
    LLaMA-Factory ou ms-swift. Só vale a pena rodar isso quando o
    dataset já tiver um volume razoável de exemplos (dezenas a
    centenas, não só 2-3).
    """
    examples = [ex for ex in load_dataset(dataset_dir) if ex.get("corrected_output")]

    records = []
    for ex in examples:
        content = [{"type": "image", "image": p} for p in ex["frames"]]
        content.append(
            {
                "type": "text",
                "text": f"Dialogue:\n{ex['dialogue']}\n\nDescribe the scene.",
            }
        )
        records.append(
            {
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": ex["corrected_output"]},
                ]
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path
