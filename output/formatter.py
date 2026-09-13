"""
Formata a timeline final.

Mantém separadas:

    speaker_id
        identidade original da diarização.

    character_id
        identidade global resolvida.

    character_name
        nome humano do personagem, quando conhecido.

TXT/SRT:
    preferem character_name.

JSON:
    preserva todas as informações.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_timestamp(
    seconds: float,
) -> str:

    seconds = float(seconds or 0)

    h = int(seconds // 3600)

    m = int((seconds % 3600) // 60)

    s = seconds % 60

    return (f"{h:02d}:" f"{m:02d}:" f"{s:06.3f}").replace(
        ".",
        ",",
    )


def _get_speaker_id(
    item: dict[str, Any],
) -> str:

    speaker_id = item.get("speaker_id") or item.get("speaker")

    if not speaker_id:
        return "SPEAKER_UNKNOWN"

    return str(speaker_id)


def _get_character_name(
    item: dict[str, Any],
) -> str | None:

    name = item.get("character_name")

    if name and str(name).strip():
        return str(name).strip()

    return None


def _get_character_id(
    item: dict[str, Any],
) -> str | None:

    character_id = item.get("character_id")

    if character_id:
        return str(character_id)

    return None


def _get_display_label(
    item: dict[str, Any],
) -> str:

    character_name = _get_character_name(item)

    if character_name:
        return character_name.upper()

    character_id = _get_character_id(item)

    if character_id:
        return character_id.upper()

    return _get_speaker_id(item).upper()


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def to_txt(
    timeline: list[dict],
) -> str:

    lines = []

    for item in timeline:

        start = float(
            item.get(
                "start",
                0,
            )
        )

        minutos = int(start // 60)

        segundos = int(start % 60)

        marca = f"[{minutos:02d}:" f"{segundos:02d}]"

        item_type = item.get("type")

        if item_type == "fala":

            label = _get_display_label(item)

            text = (
                item.get(
                    "text",
                    "",
                )
                or ""
            )

            lines.append(f"{marca} " f"{label}: " f"{text}")

        elif item_type == "acao":

            description = (
                item.get(
                    "description",
                    "",
                )
                or ""
            )

            lines.append(f"{marca} " f"(ação: " f"{description})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _normalize_json_item(
    item: dict[str, Any],
) -> dict[str, Any]:

    result = dict(item)

    # Sempre preservar speaker_id.
    result["speaker_id"] = _get_speaker_id(item)

    # Preservar identidade global.
    result["character_id"] = _get_character_id(item)

    result["character_name"] = _get_character_name(item)

    # Campos de confiança.
    identity_fields = [
        "identity_status",
        "status",
        "identity_confidence",
        "identity_margin",
        "visual_score",
        "voice_score",
        "temporal_score",
        "context_score",
        "voice_status",
        "voice_margin",
        "voice_character",
        "candidate_character_ids",
        "speaker_associations",
        "voice_evidence",
    ]

    for field in identity_fields:

        if field in item:
            result[field] = item[field]

    return result


def to_json(
    timeline: list[dict],
) -> str:

    normalized = [_normalize_json_item(item) for item in timeline]

    return json.dumps(
        normalized,
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------


def to_srt(
    timeline: list[dict],
    duracao_padrao: float = 3.0,
) -> str:

    blocos = []

    contador = 1

    for item in timeline:

        if item.get("type") != "fala":
            continue

        inicio = float(
            item.get(
                "start",
                0,
            )
        )

        fim = item.get("end")

        if fim is None:
            fim = inicio + duracao_padrao

        fim = float(fim)

        label = _get_display_label(item)

        texto = (
            item.get(
                "text",
                "",
            )
            or ""
        )

        subtitle = f"{label}: " f"{texto}"

        blocos.append(
            f"{contador}\n"
            f"{_format_timestamp(inicio)} "
            f"--> "
            f"{_format_timestamp(fim)}\n"
            f"{subtitle}\n"
        )

        contador += 1

    return "\n".join(blocos)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_outputs(
    timeline: list[dict],
    output_dir: str,
    base_name: str,
    formats: list[str],
):

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    caminhos = {}

    if "txt" in formats:

        caminho = Path(output_dir) / f"{base_name}.txt"

        caminho.write_text(
            to_txt(timeline),
            encoding="utf-8",
        )

        caminhos["txt"] = str(caminho)

    if "json" in formats:

        caminho = Path(output_dir) / f"{base_name}.json"

        caminho.write_text(
            to_json(timeline),
            encoding="utf-8",
        )

        caminhos["json"] = str(caminho)

    if "srt" in formats:

        caminho = Path(output_dir) / f"{base_name}.srt"

        caminho.write_text(
            to_srt(timeline),
            encoding="utf-8",
        )

        caminhos["srt"] = str(caminho)

    return caminhos
