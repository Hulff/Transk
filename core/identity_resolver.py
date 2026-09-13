"""
Identity Resolver

Resolve a identidade global de personagens combinando:

- identidade do speaker produzida pela diarização;
- evidência de voz;
- descritores visuais produzidos pelo Qwen2.5-VL;
- histórico das cenas;
- ficha de personagens previamente conhecida.

Princípio importante:

O Qwen NÃO decide sozinho quem é o personagem.

O Qwen fornece evidência visual.
O speaker mapping fornece evidência de voz.
Este módulo combina essas evidências e produz:

    character_id
    status
    confidence
    visual_score
    voice_score
    temporal_score
    context_score

Status possíveis:

    CONFIRMADO
    PROVÁVEL
    AMBÍGUO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


UNKNOWN_VALUES = {
    "",
    "unknown",
    "unk",
    "unknown",
    "indeterminado",
    "não identificado",
    "nao identificado",
    "não observável",
    "nao observavel",
    "none",
    "null",
    "n/a",
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = re.sub(r"\s+", " ", text)

    return text


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return _normalize_text(value) in UNKNOWN_VALUES

    return False


def _as_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _flatten_text(value: Any) -> list[str]:
    """
    Converte valores simples/listas/dicionários em uma lista de textos
    comparáveis.
    """

    if value is None:
        return []

    if isinstance(value, dict):
        result = []

        for key, item in value.items():
            result.extend(_flatten_text(key))
            result.extend(_flatten_text(item))

        return result

    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(_flatten_text(item))

        return result

    text = _normalize_text(value)

    if _is_unknown(text):
        return []

    return [text]


# ---------------------------------------------------------------------------
# Estrutura
# ---------------------------------------------------------------------------


@dataclass
class IdentityMatch:
    character_id: str
    visual_score: float
    voice_score: float
    temporal_score: float
    context_score: float
    identity_score: float
    status: str

    def to_dict(self) -> dict:
        return {
            "character_id": self.character_id,
            "visual_score": round(self.visual_score, 4),
            "voice_score": round(self.voice_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "context_score": round(self.context_score, 4),
            "identity_score": round(self.identity_score, 4),
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Normalização dos descritores
# ---------------------------------------------------------------------------


def normalize_character_descriptor(
    character: dict[str, Any],
) -> dict[str, Any]:
    """
    Normaliza o formato retornado pelo Qwen.

    Aceita tanto:

        {
            "character_ref": "personagem_A",
            "skin_tone": "...",
            "hair_color": "..."
        }

    quanto:

        {
            "character_ref": "personagem_A",
            "stable_appearance": {
                ...
            },
            "temporary_appearance": {
                ...
            }
        }

    Retorna sempre uma estrutura padronizada.
    """

    if not isinstance(character, dict):
        return {
            "character_ref": "",
            "stable_appearance": {},
            "temporary_appearance": {},
            "distinctive_features": [],
        }

    stable = character.get("stable_appearance")

    if not isinstance(stable, dict):
        stable = {}

        stable_keys = [
            "skin_tone",
            "hair_color",
            "hair_length",
            "hair_style",
            "face_shape",
            "facial_hair",
            "build",
            "body_proportions",
            "silhouette",
            "eyes",
            "eyebrows",
            "ears",
            "nose",
            "mouth",
            "visible_marks",
        ]

        for key in stable_keys:
            if key in character:
                stable[key] = character[key]

    temporary = character.get("temporary_appearance")

    if not isinstance(temporary, dict):
        temporary = {}

        temporary_keys = [
            "clothing",
            "clothing_colors",
            "clothing_patterns",
            "footwear",
            "headwear",
            "accessories",
            "weapons",
            "temporary_features",
        ]

        for key in temporary_keys:
            if key in character:
                temporary[key] = character[key]

    distinctive = character.get(
        "distinctive_features",
        stable.get("distinctive_features", []),
    )

    return {
        "character_ref": character.get(
            "character_ref",
            character.get("ref", ""),
        ),
        "name": character.get("name"),
        "stable_appearance": stable,
        "temporary_appearance": temporary,
        "distinctive_features": _as_list(distinctive),
        "speaker_ids": _as_list(character.get("speaker_ids", [])),
        "voice_character": character.get("voice_character"),
        "voice_score": character.get("voice_score"),
    }


def build_visual_signature(
    character: dict[str, Any],
) -> dict[str, Any]:
    """
    Constrói uma assinatura visual comparável.

    Características estáveis têm prioridade sobre roupa/acessórios.
    """

    normalized = normalize_character_descriptor(character)

    stable = normalized["stable_appearance"]
    temporary = normalized["temporary_appearance"]

    return {
        "skin_tone": stable.get("skin_tone"),
        "hair_color": stable.get("hair_color"),
        "hair_length": stable.get("hair_length"),
        "hair_style": stable.get("hair_style"),
        "face_shape": stable.get("face_shape"),
        "facial_hair": stable.get("facial_hair"),
        "build": stable.get("build"),
        "body_proportions": stable.get("body_proportions"),
        "silhouette": stable.get("silhouette"),
        "eyes": stable.get("eyes"),
        "eyebrows": stable.get("eyebrows"),
        "ears": stable.get("ears"),
        "nose": stable.get("nose"),
        "mouth": stable.get("mouth"),
        "visible_marks": stable.get("visible_marks"),
        "distinctive_features": normalized["distinctive_features"],
        "clothing": temporary.get("clothing"),
        "clothing_colors": temporary.get("clothing_colors"),
        "clothing_patterns": temporary.get("clothing_patterns"),
        "footwear": temporary.get("footwear"),
        "headwear": temporary.get("headwear"),
        "accessories": temporary.get("accessories"),
        "weapons": temporary.get("weapons"),
    }


# ---------------------------------------------------------------------------
# Comparação visual
# ---------------------------------------------------------------------------


VISUAL_WEIGHTS = {
    "distinctive_features": 0.25,
    "hair": 0.20,
    "face": 0.20,
    "body": 0.15,
    "skin": 0.10,
    "clothing": 0.07,
    "accessories": 0.03,
}


def _token_similarity(
    a: Any,
    b: Any,
) -> float | None:
    """
    Similaridade simples baseada em tokens.

    Não tenta substituir um modelo multimodal.
    Serve apenas para comparar descrições estruturadas produzidas
    pelo mesmo modelo.
    """

    values_a = _flatten_text(a)
    values_b = _flatten_text(b)

    if not values_a or not values_b:
        return None

    tokens_a = set()

    tokens_b = set()

    for value in values_a:
        tokens_a.update(value.split())

    for value in values_b:
        tokens_b.update(value.split())

    if not tokens_a or not tokens_b:
        return None

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    if not union:
        return None

    return len(intersection) / len(union)


def _field_similarity(
    a: dict,
    b: dict,
    fields: list[str],
) -> float | None:
    scores = []

    for field in fields:
        score = _token_similarity(
            a.get(field),
            b.get(field),
        )

        if score is not None:
            scores.append(score)

    if not scores:
        return None

    return sum(scores) / len(scores)


def _distinctive_similarity(
    a: dict,
    b: dict,
) -> float | None:
    return _token_similarity(
        a.get("distinctive_features"),
        b.get("distinctive_features"),
    )


def visual_similarity(
    a: dict[str, Any],
    b: dict[str, Any],
) -> float:
    """
    Compara duas descrições visuais.

    Roupa e acessórios possuem peso baixo porque podem mudar.

    Cabelo, rosto e características distintivas possuem peso maior.
    """

    sig_a = build_visual_signature(a)
    sig_b = build_visual_signature(b)

    weighted_scores = []
    total_weight = 0.0

    # Características distintivas
    score = _distinctive_similarity(sig_a, sig_b)

    if score is not None:
        weighted_scores.append(
            (
                score,
                VISUAL_WEIGHTS["distinctive_features"],
            )
        )
        total_weight += VISUAL_WEIGHTS["distinctive_features"]

    # Cabelo
    score = _field_similarity(
        sig_a,
        sig_b,
        [
            "hair_color",
            "hair_length",
            "hair_style",
        ],
    )

    if score is not None:
        weighted_scores.append(
            (
                score,
                VISUAL_WEIGHTS["hair"],
            )
        )
        total_weight += VISUAL_WEIGHTS["hair"]

    # Rosto
    score = _field_similarity(
        sig_a,
        sig_b,
        [
            "face_shape",
            "facial_hair",
            "eyes",
            "eyebrows",
            "ears",
            "nose",
            "mouth",
        ],
    )

    if score is not None:
        weighted_scores.append(
            (
                score,
                VISUAL_WEIGHTS["face"],
            )
        )
        total_weight += VISUAL_WEIGHTS["face"]

    # Corpo
    score = _field_similarity(
        sig_a,
        sig_b,
        [
            "build",
            "body_proportions",
            "silhouette",
        ],
    )

    if score is not None:
        weighted_scores.append(
            (
                score,
                VISUAL_WEIGHTS["body"],
            )
        )
        total_weight += VISUAL_WEIGHTS["body"]

    # Pele
    score = _token_similarity(
        sig_a.get("skin_tone"),
        sig_b.get("skin_tone"),
    )

    if score is not None:
        weighted_scores.append(
            (
                score,
                VISUAL_WEIGHTS["skin"],
            )
        )
        total_weight += VISUAL_WEIGHTS["skin"]

    # Roupa
    score = _field_similarity(
        sig_a,
        sig_b,
        [
            "clothing",
            "clothing_colors",
            "clothing_patterns",
        ],
    )

    if score is not None:
        weighted_scores.append(
            (
                score,
                VISUAL_WEIGHTS["clothing"],
            )
        )
        total_weight += VISUAL_WEIGHTS["clothing"]

    # Acessórios
    score = _field_similarity(
        sig_a,
        sig_b,
        [
            "accessories",
            "headwear",
            "footwear",
        ],
    )

    if score is not None:
        weighted_scores.append(
            (
                score,
                VISUAL_WEIGHTS["accessories"],
            )
        )
        total_weight += VISUAL_WEIGHTS["accessories"]

    if total_weight == 0:
        return 0.0

    score = sum(value * weight for value, weight in weighted_scores) / total_weight

    return max(
        0.0,
        min(1.0, score),
    )


# ---------------------------------------------------------------------------
# Evidência temporal
# ---------------------------------------------------------------------------


def temporal_similarity(
    scene_character: dict[str, Any],
    known_character: dict[str, Any],
) -> float:
    """
    Avalia continuidade temporal.

    Atualmente é conservadora.

    Speaker IDs iguais recebem forte evidência.
    Character refs iguais também ajudam.

    Isso evita que duas pessoas visualmente parecidas sejam
    constantemente trocadas.
    """

    scene_speakers = set(
        str(x) for x in _as_list(scene_character.get("speaker_ids", [])) if x
    )

    known_speakers = set(
        str(x) for x in _as_list(known_character.get("speaker_ids", [])) if x
    )

    if scene_speakers and known_speakers:
        if scene_speakers & known_speakers:
            return 1.0

        return 0.0

    scene_ref = _normalize_text(scene_character.get("character_ref"))

    known_ref = _normalize_text(known_character.get("character_ref"))

    if scene_ref and known_ref and scene_ref == known_ref:
        return 1.0

    return 0.5


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------


def context_similarity(
    scene_character: dict[str, Any],
    known_character: dict[str, Any],
) -> float:
    """
    Compara informações contextuais simples.

    Essa função propositalmente não tenta inferir identidade através
    de conhecimento externo.
    """

    scores = []

    scene_voice = _normalize_text(scene_character.get("voice_character"))

    known_name = _normalize_text(known_character.get("name"))

    if scene_voice and known_name:
        if scene_voice == known_name:
            scores.append(1.0)
        else:
            scores.append(0.0)

    scene_name = _normalize_text(scene_character.get("name"))

    if scene_name and known_name:
        if scene_name == known_name:
            scores.append(1.0)

    if not scores:
        return 0.5

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------


def voice_score_for_character(
    scene_character: dict[str, Any],
    known_character: dict[str, Any],
) -> float:
    """
    Retorna a evidência de voz disponível para essa identidade.
    """

    scene_voice_character = _normalize_text(scene_character.get("voice_character"))

    known_name = _normalize_text(known_character.get("name"))

    if scene_voice_character and known_name:
        if scene_voice_character == known_name:
            raw_score = scene_character.get(
                "voice_score",
                1.0,
            )

            try:
                return max(
                    0.0,
                    min(1.0, float(raw_score)),
                )
            except (TypeError, ValueError):
                return 1.0

        return 0.0

    # Caso voice_character ainda seja o ID do speaker,
    # não temos uma associação direta com o personagem.
    return 0.5


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


def match_character(
    scene_character: dict[str, Any],
    known_characters: list[dict[str, Any]],
    *,
    visual_weight: float = 0.40,
    voice_weight: float = 0.35,
    temporal_weight: float = 0.15,
    context_weight: float = 0.10,
    confirmed_threshold: float = 0.80,
    probable_threshold: float = 0.65,
) -> IdentityMatch | None:
    """
    Encontra o personagem conhecido mais provável.
    """

    if not known_characters:
        return None

    candidates = []

    for known in known_characters:
        character_id = known.get("character_id")

        if not character_id:
            continue

        visual = visual_similarity(
            scene_character,
            known,
        )

        voice = voice_score_for_character(
            scene_character,
            known,
        )

        temporal = temporal_similarity(
            scene_character,
            known,
        )

        context = context_similarity(
            scene_character,
            known,
        )

        score = (
            visual * visual_weight
            + voice * voice_weight
            + temporal * temporal_weight
            + context * context_weight
        )

        candidates.append(
            IdentityMatch(
                character_id=character_id,
                visual_score=visual,
                voice_score=voice,
                temporal_score=temporal,
                context_score=context,
                identity_score=score,
                status="",
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item.identity_score,
        reverse=True,
    )

    best = candidates[0]

    second_score = candidates[1].identity_score if len(candidates) > 1 else 0.0

    margin = best.identity_score - second_score

    if best.identity_score >= confirmed_threshold and margin >= 0.10:
        best.status = "CONFIRMADO"

    elif best.identity_score >= probable_threshold and margin >= 0.05:
        best.status = "PROVÁVEL"

    else:
        best.status = "AMBÍGUO"

    return best


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _new_character(
    character_id: str,
    scene_character: dict[str, Any],
    scene_index: int,
) -> dict[str, Any]:
    normalized = normalize_character_descriptor(scene_character)

    return {
        "character_id": character_id,
        "name": normalized.get("name"),
        "aliases": [],
        "stable_appearance": normalized.get(
            "stable_appearance",
            {},
        ),
        "temporary_appearance": normalized.get(
            "temporary_appearance",
            {},
        ),
        "distinctive_features": normalized.get(
            "distinctive_features",
            [],
        ),
        "speaker_ids": list(normalized.get("speaker_ids", [])),
        "voice_character": normalized.get("voice_character"),
        "voice_confidence": normalized.get("voice_score"),
        "visual_confidence": None,
        "identity_confidence": None,
        "scenes": [scene_index],
    }


def _merge_character_evidence(
    character: dict[str, Any],
    scene_character: dict[str, Any],
    scene_index: int,
    identity_match: IdentityMatch | None,
) -> None:
    normalized = normalize_character_descriptor(scene_character)

    # Nome
    name = normalized.get("name")

    if name and not _is_unknown(name):
        character["name"] = name

    # Speaker
    for speaker in normalized.get(
        "speaker_ids",
        [],
    ):
        if speaker not in character["speaker_ids"]:
            character["speaker_ids"].append(speaker)

    # Voz
    if normalized.get("voice_character"):
        character["voice_character"] = normalized["voice_character"]

    voice_score = normalized.get("voice_score")

    if voice_score is not None:
        try:
            current = character.get("voice_confidence")

            if current is None:
                character["voice_confidence"] = float(voice_score)
            else:
                character["voice_confidence"] = max(
                    float(current),
                    float(voice_score),
                )

        except (TypeError, ValueError):
            pass

    # Stable appearance
    stable = normalized.get(
        "stable_appearance",
        {},
    )

    character_stable = character.setdefault(
        "stable_appearance",
        {},
    )

    for key, value in stable.items():
        if _is_unknown(value):
            continue

        if key not in character_stable or _is_unknown(character_stable[key]):
            character_stable[key] = value

    # Temporary appearance
    temporary = normalized.get(
        "temporary_appearance",
        {},
    )

    character_temporary = character.setdefault(
        "temporary_appearance",
        {},
    )

    for key, value in temporary.items():
        if _is_unknown(value):
            continue

        character_temporary[key] = value

    # Distinctive features
    features = character.setdefault(
        "distinctive_features",
        [],
    )

    for feature in normalized.get(
        "distinctive_features",
        [],
    ):
        if feature not in features:
            features.append(feature)

    # Scene
    if scene_index not in character["scenes"]:
        character["scenes"].append(scene_index)

    # Confidence
    if identity_match is not None:
        visual = identity_match.visual_score

        current_visual = character.get("visual_confidence")

        if current_visual is None:
            character["visual_confidence"] = visual
        else:
            character["visual_confidence"] = max(
                current_visual,
                visual,
            )

        identity = identity_match.identity_score

        current_identity = character.get("identity_confidence")

        if current_identity is None:
            character["identity_confidence"] = identity
        else:
            character["identity_confidence"] = max(
                current_identity,
                identity,
            )


def resolve_scene_characters(
    scenes: list[dict[str, Any]],
    previous_registry: dict[str, Any] | None = None,
    *,
    visual_weight: float = 0.40,
    voice_weight: float = 0.35,
    temporal_weight: float = 0.15,
    context_weight: float = 0.10,
    confirmed_threshold: float = 0.80,
    probable_threshold: float = 0.65,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Resolve os personagens de todas as cenas.

    Retorna:

        resolved_scenes
        registry
    """

    registry: dict[str, dict[str, Any]] = {}

    if previous_registry:
        for character_id, character in previous_registry.items():
            registry[character_id] = character

    next_id = 1

    if registry:
        existing_numbers = []

        for character_id in registry:
            match = re.search(
                r"(\d+)$",
                character_id,
            )

            if match:
                existing_numbers.append(int(match.group(1)))

        if existing_numbers:
            next_id = max(existing_numbers) + 1

    resolved_scenes = []

    for scene_index, scene in enumerate(
        scenes,
        start=1,
    ):
        scene_copy = dict(scene)

        visual_characters = scene.get(
            "visual_descriptors",
            [],
        )

        if not isinstance(
            visual_characters,
            list,
        ):
            visual_characters = []

        scene_resolved = []

        for visual_character in visual_characters:
            normalized = normalize_character_descriptor(visual_character)

            # Speaker associado à evidência visual
            speaker = visual_character.get(
                "speaker_id",
                visual_character.get("speaker"),
            )

            if speaker:
                normalized["speaker_ids"] = [speaker]

            # Tenta encontrar identidade existente
            known = list(registry.values())

            match = match_character(
                normalized,
                known,
                visual_weight=visual_weight,
                voice_weight=voice_weight,
                temporal_weight=temporal_weight,
                context_weight=context_weight,
                confirmed_threshold=confirmed_threshold,
                probable_threshold=probable_threshold,
            )

            if match is not None and match.status != "AMBÍGUO":
                character_id = match.character_id

            else:
                character_id = f"character_{next_id:03d}"

                next_id += 1

                registry[character_id] = _new_character(
                    character_id,
                    normalized,
                    scene_index,
                )

                match = IdentityMatch(
                    character_id=character_id,
                    visual_score=0.0,
                    voice_score=0.0,
                    temporal_score=0.0,
                    context_score=0.0,
                    identity_score=0.0,
                    status="AMBÍGUO",
                )

            _merge_character_evidence(
                registry[character_id],
                normalized,
                scene_index,
                match,
            )

            resolved_character = {
                "scene_character_ref": normalized.get("character_ref"),
                "character_id": character_id,
                "name": registry[character_id].get("name"),
                "status": match.status,
                "identity_confidence": round(
                    match.identity_score,
                    4,
                ),
                "visual_score": round(
                    match.visual_score,
                    4,
                ),
                "voice_score": round(
                    match.voice_score,
                    4,
                ),
                "temporal_score": round(
                    match.temporal_score,
                    4,
                ),
                "context_score": round(
                    match.context_score,
                    4,
                ),
                "speaker_id": speaker,
            }

            scene_resolved.append(resolved_character)

        scene_copy["resolved_characters"] = scene_resolved

        # Mapeamento rápido speaker -> character
        speaker_mapping = {}

        for item in scene_resolved:
            speaker_id = item.get("speaker_id")

            character_id = item.get("character_id")

            if speaker_id and character_id:
                speaker_mapping[speaker_id] = character_id

        scene_copy["speaker_character_mapping"] = speaker_mapping

        resolved_scenes.append(scene_copy)

    return (
        resolved_scenes,
        registry,
    )


def format_character_registry(
    registry: dict[str, Any],
) -> str:
    """
    Formata a registry estruturada em texto humano-legível.
    """

    blocks = []

    for character_id, character in registry.items():
        stable = character.get(
            "stable_appearance",
            {},
        )

        temporary = character.get(
            "temporary_appearance",
            {},
        )

        features = character.get(
            "distinctive_features",
            [],
        )

        name = character.get("name") or "UNNAMED"

        blocks.append(
            "\n".join(
                [
                    f"CHARACTER_ID: {character_id}",
                    f"NAME: {name}",
                    (
                        "SPEAKER_IDS: "
                        + ", ".join(
                            map(
                                str,
                                character.get(
                                    "speaker_ids",
                                    [],
                                ),
                            )
                        )
                    ),
                    ("STABLE_APPEARANCE: " + str(stable)),
                    ("TEMPORARY_APPEARANCE: " + str(temporary)),
                    ("DISTINCTIVE_FEATURES: " + str(features)),
                    ("VISUAL_CONFIDENCE: " + str(character.get("visual_confidence"))),
                    ("VOICE_CONFIDENCE: " + str(character.get("voice_confidence"))),
                    (
                        "IDENTITY_CONFIDENCE: "
                        + str(character.get("identity_confidence"))
                    ),
                    (
                        "SCENES: "
                        + ", ".join(
                            map(
                                str,
                                character.get(
                                    "scenes",
                                    [],
                                ),
                            )
                        )
                    ),
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)
