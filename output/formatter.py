"""
Formata a timeline final (falas + ações) em diferentes formatos de saída.
"""

import json
from pathlib import Path


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def to_txt(timeline: list[dict]) -> str:
    """Formato tipo roteiro, legível por humanos."""
    lines = []
    for item in timeline:
        minutos = int(item["start"] // 60)
        segundos = int(item["start"] % 60)
        marca = f"[{minutos:02d}:{segundos:02d}]"

        if item["type"] == "fala":
            lines.append(f"{marca} {item['speaker'].upper()}: {item['text']}")
        elif item["type"] == "acao":
            lines.append(f"{marca} (ação: {item['description']})")

    return "\n".join(lines)


def to_json(timeline: list[dict]) -> str:
    """Formato estruturado, útil pra reaproveitar em outro app/site."""
    return json.dumps(timeline, ensure_ascii=False, indent=2)


def to_srt(timeline: list[dict], duracao_padrao: float = 3.0) -> str:
    """
    Formato de legenda .srt (só falas — ações normalmente não entram
    em legenda, mas você pode incluir se quiser).
    """
    blocos = []
    contador = 1

    for item in timeline:
        if item["type"] != "fala":
            continue

        inicio = item["start"]
        fim = item.get("end", inicio + duracao_padrao)

        texto = f"{item['speaker'].upper()}: {item['text']}"
        blocos.append(
            f"{contador}\n{_format_timestamp(inicio)} --> {_format_timestamp(fim)}\n{texto}\n"
        )
        contador += 1

    return "\n".join(blocos)


def save_outputs(
    timeline: list[dict], output_dir: str, base_name: str, formats: list[str]
):
    """Salva a timeline nos formatos pedidos (txt, json, srt)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    caminhos = {}

    if "txt" in formats:
        caminho = Path(output_dir) / f"{base_name}.txt"
        caminho.write_text(to_txt(timeline), encoding="utf-8")
        caminhos["txt"] = str(caminho)

    if "json" in formats:
        caminho = Path(output_dir) / f"{base_name}.json"
        caminho.write_text(to_json(timeline), encoding="utf-8")
        caminhos["json"] = str(caminho)

    if "srt" in formats:
        caminho = Path(output_dir) / f"{base_name}.srt"
        caminho.write_text(to_srt(timeline), encoding="utf-8")
        caminhos["srt"] = str(caminho)

    return caminhos
