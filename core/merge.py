"""
Junta falas e descrições de cena.

No modo "fala", cada descrição de cena é associada
à fala correspondente pelo timestamp.

Resultado:

[
    {
        "type": "fala",
        "start": 2.84,
        "end": 8.54,
        "speaker": "SPEAKER_03",
        "text": "...",
        "acao": "..."
    }
]
"""

from typing import Optional


def merge_timeline(
    segments: list[dict],
    scenes: Optional[list[dict]] = None,
) -> list[dict]:

    timeline = []

    # ============================================================
    # Cria índice das cenas por timestamp
    # ============================================================

    scene_by_timestamp = {}

    if scenes:
        for scene in scenes:

            description = str(
                scene.get("description", "")
            ).strip()

            if not description:
                continue

            timestamp = float(
                scene.get("timestamp", 0)
            )

            scene_by_timestamp[timestamp] = description

    # ============================================================
    # Junta cada fala com sua ação
    # ============================================================

    for seg in segments:

        start = float(
            seg.get("start", 0)
        )

        end = float(
            seg.get("end", start)
        )

        item = {
            "type": "fala",
            "start": start,
            "end": end,
            "speaker": seg.get(
                "speaker",
                "",
            ),
            "text": seg.get(
                "text",
                "",
            ),
        }

        # --------------------------------------------------------
        # Procura ação correspondente
        # --------------------------------------------------------

        description = scene_by_timestamp.get(
            start
        )

        if description:

            item["acao"] = description

        timeline.append(item)

    # ============================================================
    # Cenas que não pertencem a nenhuma fala
    #
    # Útil caso futuramente o modo "intervalo" seja usado.
    # ============================================================

    fala_timestamps = {
        float(seg.get("start", 0))
        for seg in segments
    }

    if scenes:

        for scene in scenes:

            description = str(
                scene.get("description", "")
            ).strip()

            if not description:
                continue

            timestamp = float(
                scene.get("timestamp", 0)
            )

            # Já foi associada a uma fala
            if timestamp in fala_timestamps:
                continue

            timeline.append({
                "type": "acao",
                "start": timestamp,
                "description": description,
            })

    # ============================================================
    # Ordena
    # ============================================================

    timeline.sort(
        key=lambda item: item["start"]
    )

    return timeline
