"""
Cache simples em JSON para resultados intermediários (falas e cenas),
permitindo rodar a análise de áudio e a de cena separadamente, sem
precisar refazer tudo do zero quando for juntar os dois depois.
"""
import json
from pathlib import Path

CACHE_DIR = Path("temp/cache")


def _cache_path(base_name: str, kind: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{base_name}_{kind}.json"


def save_cache(base_name: str, kind: str, data) -> str:
    """kind: 'falas' ou 'cenas'"""
    path = _cache_path(base_name, kind)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def load_cache(base_name: str, kind: str):
    """Retorna os dados salvos, ou None se não existir cache pra esse item."""
    path = _cache_path(base_name, kind)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def has_cache(base_name: str, kind: str) -> bool:
    return _cache_path(base_name, kind).exists()