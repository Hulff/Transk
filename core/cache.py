"""
Cache simples em JSON para resultados intermediários.

Tipos usados pelo pipeline:

    falas
    speaker_mapping
    cenas
    identity_resolution
    character_registry
    fichas_personagens
    cenas_consistentes
    analise_rascunho
    analise_validada
"""

import json
from pathlib import Path

CACHE_DIR = Path("temp/cache")


VALID_KINDS = {
    "falas",
    "speaker_mapping",
    "cenas",
    "identity_resolution",
    "character_registry",
    "fichas_personagens",
    "cenas_consistentes",
    "analise_rascunho",
    "analise_validada",
}


def _cache_path(
    base_name: str,
    kind: str,
) -> Path:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return CACHE_DIR / f"{base_name}_{kind}.json"


def save_cache(
    base_name: str,
    kind: str,
    data,
) -> str:
    """
    Salva qualquer resultado intermediário.

    `kind` pode ser qualquer nome, mas os tipos conhecidos estão
    listados em VALID_KINDS.
    """

    path = _cache_path(
        base_name,
        kind,
    )

    serialized = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )

    path.write_text(
        serialized,
        encoding="utf-8",
    )

    return str(path)


def load_cache(
    base_name: str,
    kind: str,
):
    """
    Retorna os dados salvos.

    Se o arquivo não existir:
        None

    Se o JSON estiver inválido:
        levanta RuntimeError com contexto.
    """

    path = _cache_path(
        base_name,
        kind,
    )

    if not path.exists():
        return None

    try:

        return json.loads(path.read_text(encoding="utf-8"))

    except json.JSONDecodeError as error:

        raise RuntimeError(f"Cache corrompido: {path}") from error


def has_cache(
    base_name: str,
    kind: str,
) -> bool:

    return _cache_path(
        base_name,
        kind,
    ).exists()


def delete_cache(
    base_name: str,
    kind: str,
) -> bool:
    """
    Remove um cache específico.

    Retorna True se removeu.
    """

    path = _cache_path(
        base_name,
        kind,
    )

    if not path.exists():
        return False

    path.unlink()

    return True


def clear_identity_cache(
    base_name: str,
) -> None:
    """
    Limpa somente caches relacionados à identidade.

    Útil quando você altera:

    - threshold de voz;
    - pesos visuais;
    - algoritmo de matching.

    Não precisa reexecutar WhisperX ou Qwen.
    """

    for kind in (
        "speaker_mapping",
        "identity_resolution",
        "character_registry",
        "fichas_personagens",
        "cenas_consistentes",
    ):

        delete_cache(
            base_name,
            kind,
        )
