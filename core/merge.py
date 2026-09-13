"""
Junta falas e descrições de cena.

A timeline preserva a identidade completa:

    speaker_id
    voice_character
    character_id
    character_name

Isso permite que o JSON final mantenha rastreabilidade da decisão
de identidade.
"""

from typing import Optional


def merge_timeline(
    segments: list[dict],
    scenes: Optional[list[dict]] = None,
) -> list[dict]:

    timeline = []

    # ------------------------------------------------------------------
    # Falas
    # ------------------------------------------------------------------

    for segment in segments:

        start = float(
            segment.get(
                "start",
                0,
            )
        )

        end = float(
            segment.get(
                "end",
                start,
            )
        )

        timeline.append(
            {
                "type": "fala",
                "start": start,
                "end": end,
                # Identidade original da diarização
                "speaker_id": (
                    segment.get("speaker_id")
                    or segment.get(
                        "speaker",
                        "",
                    )
                ),
                # Compatibilidade com versões antigas
                "speaker": segment.get(
                    "speaker",
                    segment.get(
                        "speaker_id",
                        "",
                    ),
                ),
                # Evidência de voz
                "voice_character": segment.get("voice_character"),
                "voice_score": segment.get(
                    "voice_score",
                    0.0,
                ),
                "voice_margin": segment.get(
                    "voice_margin",
                    0.0,
                ),
                "voice_status": segment.get("voice_status"),
                # Identidade global, quando disponível
                "character_id": segment.get("character_id"),
                "character_name": segment.get("character_name"),
                "text": segment.get(
                    "text",
                    "",
                ),
            }
        )

    # ------------------------------------------------------------------
    # Cenas
    # ------------------------------------------------------------------

    if scenes:

        for scene in scenes:

            description = str(
                scene.get(
                    "description",
                    "",
                )
            ).strip()

            if not description:
                continue

            timeline.append(
                {
                    "type": "acao",
                    "start": float(
                        scene.get(
                            "timestamp",
                            scene.get(
                                "start",
                                0,
                            ),
                        )
                    ),
                    "end": float(
                        scene.get(
                            "end",
                            scene.get(
                                "timestamp",
                                0,
                            ),
                        )
                    ),
                    "description": description,
                    # Evidência visual
                    "visual_descriptors": scene.get(
                        "visual_descriptors",
                        [],
                    ),
                    "resolved_characters": scene.get(
                        "resolved_characters",
                        [],
                    ),
                    "scene_events": scene.get(
                        "scene_events",
                        {},
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Ordenação
    # ------------------------------------------------------------------

    # Em empate:
    #
    # ação vem antes da fala.
    #
    # Isso faz a descrição visual aparecer antes do diálogo
    # correspondente.

    timeline.sort(
        key=lambda item: (
            item["start"],
            item["type"] != "acao",
        )
    )

    return timeline
