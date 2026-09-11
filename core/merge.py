"""
Junta falas e descrições de cena.

Cada cena analisada (que pode agrupar uma ou várias falas) vira um
item "acao" independente na timeline, posicionado no timestamp de
início da cena — antes das falas que pertencem a ela. Isso evita que,
em cenas com múltiplas falas, só a primeira receba a anotação visual.

Resultado:

[
    {"type": "acao", "start": 2.84, "description": "..."},
    {"type": "fala", "start": 2.84, "end": 8.54, "speaker": "SPEAKER_03", "text": "..."},
    {"type": "fala", "start": 8.56, "end": 9.72, "speaker": "SPEAKER_03", "text": "..."},
    ...
]
"""

from typing import Optional


def merge_timeline(
    segments: list[dict],
    scenes: Optional[list[dict]] = None,
) -> list[dict]:

    timeline = []

    for seg in segments:

        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))

        timeline.append(
            {
                "type": "fala",
                "start": start,
                "end": end,
                "speaker": seg.get("speaker", ""),
                "text": seg.get("text", ""),
            }
        )

    if scenes:
        for scene in scenes:

            description = str(scene.get("description", "")).strip()

            if not description:
                continue

            timeline.append(
                {
                    "type": "acao",
                    "start": float(scene.get("timestamp", 0)),
                    "description": description,
                }
            )

    # Ordena por tempo; em empate (ação e fala no mesmo timestamp,
    # comum quando a cena começa exatamente na primeira fala), a ação
    # vem antes — faz mais sentido ler o contexto da cena antes do
    # diálogo que acontece dentro dela.
    timeline.sort(key=lambda item: (item["start"], item["type"] != "acao"))

    return timeline
