"""
Identity Resolver

Resolve a identidade global dos personagens combinando:

- speaker_id produzido pela diarização;
- evidência de reconhecimento de voz;
- associações visuais produzidas pelo Qwen2.5-VL;
- descritores visuais;
- continuidade temporal;
- contexto da cena.

Princípio:

    Qwen -> fornece evidência visual.
    Speaker mapping -> fornece evidência de voz.
    Identity Resolver -> decide a identidade global.

O resolver NUNCA deve simplesmente confiar no nome produzido pelo Qwen.

Status possíveis:

    CONFIRMADO
    PROVÁVEL
    AMBÍGUO
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

UNKNOWN_VALUES = {
    "",
    "unknown",
    "unk",
    "indeterminado",
    "não identificado",
    "nao identificado",
    "não observável",
    "nao observavel",
    "none",
    "null",
    "n/a",
    "na",
}


VISUAL_WEIGHTS = {
    "distinctive_features": 0.25,
    "hair": 0.20,
    "face": 0.20,
    "body": 0.15,
    "skin": 0.10,
    "clothing": 0.07,
    "accessories": 0.03,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _clean_speaker_id(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if _is_unknown(text):
        return None

    return text


# ---------------------------------------------------------------------------
# Estruturas
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
    margin: float = 0.0
    candidate_character_ids: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "character_id": self.character_id,
            "visual_score": round(self.visual_score, 4),
            "voice_score": round(self.voice_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "context_score": round(self.context_score, 4),
            "identity_score": round(self.identity_score, 4),
            "status": self.status,
            "margin": round(self.margin, 4),
            "candidate_character_ids": self.candidate_character_ids or [],
        }


# ---------------------------------------------------------------------------
# Speaker associations
# ---------------------------------------------------------------------------


def _normalize_association(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    speaker_id = (
        value.get("speaker_id") or value.get("speaker") or value.get("speakerId")
    )

    status = _normalize_text(value.get("status", "indeterminado"))

    confidence = _normalize_text(value.get("confidence", "baixa"))

    evidence = value.get("evidence", "")

    return {
        "speaker_id": _clean_speaker_id(speaker_id),
        "status": status or "indeterminado",
        "confidence": confidence or "baixa",
        "evidence": str(evidence).strip() if evidence else "",
    }


def _extract_speaker_associations(
    character: dict[str, Any],
) -> list[dict[str, Any]]:
    associations = character.get("speaker_associations", [])

    if not isinstance(associations, list):
        associations = [associations]

    normalized = []

    for item in associations:
        association = _normalize_association(item)

        if association.get("speaker_id"):
            normalized.append(association)

    # Compatibilidade com versões antigas.
    direct_speaker = character.get("speaker_id") or character.get("speaker")

    if direct_speaker:
        direct_speaker = _clean_speaker_id(direct_speaker)

        if direct_speaker and not any(
            item.get("speaker_id") == direct_speaker for item in normalized
        ):
            normalized.append(
                {
                    "speaker_id": direct_speaker,
                    "status": "indeterminado",
                    "confidence": "baixa",
                    "evidence": "",
                }
            )

    return normalized


def _extract_speaker_ids(
    character: dict[str, Any],
) -> list[str]:
    result = []

    for speaker_id in _as_list(character.get("speaker_ids", [])):
        speaker_id = _clean_speaker_id(speaker_id)

        if speaker_id and speaker_id not in result:
            result.append(speaker_id)

    for association in _extract_speaker_associations(character):
        speaker_id = association.get("speaker_id")

        if speaker_id and speaker_id not in result:
            result.append(speaker_id)

    direct = character.get("speaker_id") or character.get("speaker")

    direct = _clean_speaker_id(direct)

    if direct and direct not in result:
        result.append(direct)

    return result


# ---------------------------------------------------------------------------
# Normalização dos descritores visuais
# ---------------------------------------------------------------------------


def normalize_character_descriptor(
    character: dict[str, Any],
) -> dict[str, Any]:

    if not isinstance(character, dict):
        return {
            "character_ref": "",
            "name": None,
            "stable_appearance": {},
            "temporary_appearance": {},
            "distinctive_features": [],
            "speaker_ids": [],
            "speaker_associations": [],
            "voice_character": None,
            "voice_score": None,
            "voice_margin": None,
            "voice_status": None,
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

    associations = _extract_speaker_associations(character)

    speaker_ids = _extract_speaker_ids(character)

    return {
        "character_ref": character.get(
            "character_ref",
            character.get("ref", ""),
        ),
        "name": character.get("name"),
        "stable_appearance": stable,
        "temporary_appearance": temporary,
        "distinctive_features": _as_list(distinctive),
        "speaker_ids": speaker_ids,
        "speaker_associations": associations,
        "voice_character": character.get("voice_character"),
        "voice_score": character.get("voice_score"),
        "voice_margin": character.get("voice_margin"),
        "voice_status": character.get("voice_status"),
    }


def build_visual_signature(
    character: dict[str, Any],
) -> dict[str, Any]:

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
# Similaridade visual
# ---------------------------------------------------------------------------


def _token_similarity(
    a: Any,
    b: Any,
) -> float | None:

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

    union = tokens_a | tokens_b

    if not union:
        return None

    return len(tokens_a & tokens_b) / len(union)


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

    sig_a = build_visual_signature(a)
    sig_b = build_visual_signature(b)

    weighted_scores = []
    total_weight = 0.0

    score = _distinctive_similarity(sig_a, sig_b)

    if score is not None:
        weighted_scores.append((score, VISUAL_WEIGHTS["distinctive_features"]))
        total_weight += VISUAL_WEIGHTS["distinctive_features"]

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
        weighted_scores.append((score, VISUAL_WEIGHTS["hair"]))
        total_weight += VISUAL_WEIGHTS["hair"]

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
        weighted_scores.append((score, VISUAL_WEIGHTS["face"]))
        total_weight += VISUAL_WEIGHTS["face"]

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
        weighted_scores.append((score, VISUAL_WEIGHTS["body"]))
        total_weight += VISUAL_WEIGHTS["body"]

    score = _token_similarity(
        sig_a.get("skin_tone"),
        sig_b.get("skin_tone"),
    )

    if score is not None:
        weighted_scores.append((score, VISUAL_WEIGHTS["skin"]))
        total_weight += VISUAL_WEIGHTS["skin"]

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
        weighted_scores.append((score, VISUAL_WEIGHTS["clothing"]))
        total_weight += VISUAL_WEIGHTS["clothing"]

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
        weighted_scores.append((score, VISUAL_WEIGHTS["accessories"]))
        total_weight += VISUAL_WEIGHTS["accessories"]

    if total_weight == 0:
        return 0.0

    result = sum(score * weight for score, weight in weighted_scores) / total_weight

    return max(0.0, min(1.0, result))


# ---------------------------------------------------------------------------
# Associação visual -> speaker
# ---------------------------------------------------------------------------


def speaker_association_score(
    scene_character: dict[str, Any],
    known_character: dict[str, Any],
) -> float:

    scene_associations = _extract_speaker_associations(scene_character)

    scene_speakers = set(_extract_speaker_ids(scene_character))

    known_speakers = set(_extract_speaker_ids(known_character))

    if scene_speakers and known_speakers:
        overlap = scene_speakers & known_speakers

        if overlap:
            return 1.0

    for association in scene_associations:
        speaker_id = association.get("speaker_id")

        if not speaker_id or speaker_id not in known_speakers:
            continue

        status = association.get("status", "")

        confidence = association.get("confidence", "")

        if status in {
            "confirmado",
            "confirmed",
        }:
            return 1.0

        if confidence == "alta":
            return 0.95

        if confidence == "media":
            return 0.80

        return 0.65

    return 0.5


# ---------------------------------------------------------------------------
# Evidência temporal
# ---------------------------------------------------------------------------


def temporal_similarity(
    scene_character: dict[str, Any],
    known_character: dict[str, Any],
) -> float:

    scene_speakers = set(_extract_speaker_ids(scene_character))

    known_speakers = set(_extract_speaker_ids(known_character))

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

    scores = []

    scene_voice = _normalize_text(scene_character.get("voice_character"))

    known_voice = _normalize_text(known_character.get("voice_character"))

    known_name = _normalize_text(known_character.get("name"))

    scene_name = _normalize_text(scene_character.get("name"))

    if scene_voice and known_voice:
        scores.append(1.0 if scene_voice == known_voice else 0.0)

    if scene_voice and known_name:
        scores.append(1.0 if scene_voice == known_name else 0.0)

    if scene_name and known_name:
        scores.append(1.0 if scene_name == known_name else 0.0)

    if not scores:
        return 0.5

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Voz
# ---------------------------------------------------------------------------


def _voice_evidence_from_item(
    item: Any,
) -> list[dict[str, Any]]:

    if not isinstance(item, dict):
        return []

    result = []

    speaker_id = item.get("speaker_id") or item.get("speaker")

    speaker_id = _clean_speaker_id(speaker_id)

    character_name = (
        item.get("character_name")
        or item.get("character")
        or item.get("voice_character")
    )

    voice_score = item.get("voice_score")
    voice_margin = item.get("voice_margin")
    voice_status = item.get("voice_status")

    voice_match = item.get("voice_match")

    if isinstance(voice_match, dict):
        character_name = (
            voice_match.get("character")
            or voice_match.get("character_name")
            or character_name
        )

        voice_score = (
            voice_match.get("score")
            if voice_match.get("score") is not None
            else voice_score
        )

        voice_margin = (
            voice_match.get("margin")
            if voice_match.get("margin") is not None
            else voice_margin
        )

    if character_name:
        result.append(
            {
                "speaker_id": speaker_id,
                "character": character_name,
                "score": voice_score,
                "margin": voice_margin,
                "status": voice_status,
            }
        )

    return result


def extract_voice_evidence(
    scene: dict[str, Any],
) -> list[dict[str, Any]]:

    evidence = []

    candidate_containers = [
        scene.get("segments"),
        scene.get("dialogue_segments"),
        scene.get("speaker_segments"),
        scene.get("timeline"),
    ]

    for container in candidate_containers:
        if not isinstance(container, list):
            continue

        for item in container:
            evidence.extend(_voice_evidence_from_item(item))

    # A própria cena também pode conter evidência.
    evidence.extend(_voice_evidence_from_item(scene))

    return evidence


def voice_score_for_character(
    scene_character: dict[str, Any],
    known_character: dict[str, Any],
    voice_evidence: list[dict[str, Any]] | None = None,
) -> float:

    known_id = known_character.get("character_id")

    known_name = _normalize_text(known_character.get("name"))

    known_voice = _normalize_text(known_character.get("voice_character"))

    known_speakers = set(_extract_speaker_ids(known_character))

    scene_voice = _normalize_text(scene_character.get("voice_character"))

    if scene_voice:
        if known_name and scene_voice == known_name:
            return 1.0

        if known_voice and scene_voice == known_voice:
            return 1.0

        if known_name or known_voice:
            return 0.0

    if voice_evidence:
        scores = []

        for evidence in voice_evidence:
            speaker_id = evidence.get("speaker_id")
            character = _normalize_text(evidence.get("character"))

            if speaker_id and speaker_id not in known_speakers:
                continue

            matches_name = character and (
                character == known_name
                or character == known_voice
                or character == _normalize_text(known_id)
            )

            if not matches_name:
                continue

            score = evidence.get("score")

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None

            if score is not None:
                scores.append(max(0.0, min(1.0, score)))
            else:
                scores.append(1.0)

        if scores:
            return max(scores)

    return 0.5


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_character(
    scene_character: dict[str, Any],
    known_characters: list[dict[str, Any]],
    *,
    voice_evidence: list[dict[str, Any]] | None = None,
    visual_weight: float = 0.40,
    voice_weight: float = 0.35,
    temporal_weight: float = 0.15,
    context_weight: float = 0.10,
    confirmed_threshold: float = 0.80,
    probable_threshold: float = 0.65,
    confirmed_margin: float = 0.10,
    probable_margin: float = 0.05,
) -> IdentityMatch | None:

    if not known_characters:
        return None

    normalized_scene = normalize_character_descriptor(scene_character)

    scored = []

    for known in known_characters:
        visual = visual_similarity(
            normalized_scene,
            known,
        )

        voice = voice_score_for_character(
            normalized_scene,
            known,
            voice_evidence,
        )

        temporal = temporal_similarity(
            normalized_scene,
            known,
        )

        context = context_similarity(
            normalized_scene,
            known,
        )

        speaker_score = speaker_association_score(
            normalized_scene,
            known,
        )

        # A associação explícita speaker -> personagem é incorporada
        # ao componente temporal, sem permitir que ela ignore totalmente
        # uma forte contradição visual.
        temporal = max(
            temporal,
            speaker_score * 0.90,
        )

        total = (
            visual * visual_weight
            + voice * voice_weight
            + temporal * temporal_weight
            + context * context_weight
        )

        scored.append(
            {
                "character": known,
                "visual": visual,
                "voice": voice,
                "temporal": temporal,
                "context": context,
                "total": total,
            }
        )

    scored.sort(
        key=lambda item: item["total"],
        reverse=True,
    )

    best = scored[0]

    second_score = scored[1]["total"] if len(scored) > 1 else 0.0

    margin = best["total"] - second_score

    candidate_ids = []

    # Todos os candidatos próximos do melhor resultado.
    for item in scored:
        if best["total"] - item["total"] <= max(
            probable_margin,
            0.05,
        ):
            character_id = item["character"].get("character_id")

            if character_id:
                candidate_ids.append(character_id)

    if best["total"] >= confirmed_threshold and margin >= confirmed_margin:
        status = "CONFIRMADO"

    elif best["total"] >= probable_threshold and margin >= probable_margin:
        status = "PROVÁVEL"

    else:
        status = "AMBÍGUO"

    return IdentityMatch(
        character_id=best["character"].get(
            "character_id",
            "",
        ),
        visual_score=best["visual"],
        voice_score=best["voice"],
        temporal_score=best["temporal"],
        context_score=best["context"],
        identity_score=best["total"],
        status=status,
        margin=margin,
        candidate_character_ids=candidate_ids,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _new_character(
    character_id: str,
    normalized: dict[str, Any],
    scene_index: int,
) -> dict[str, Any]:

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
        "distinctive_features": list(
            normalized.get(
                "distinctive_features",
                [],
            )
        ),
        "speaker_ids": list(
            normalized.get(
                "speaker_ids",
                [],
            )
        ),
        "speaker_associations": list(
            normalized.get(
                "speaker_associations",
                [],
            )
        ),
        "voice_character": normalized.get("voice_character"),
        "voice_confidence": normalized.get("voice_score"),
        "visual_confidence": None,
        "identity_confidence": None,
        "identity_status": "PROVÁVEL",
        "scenes": [scene_index],
    }


def _merge_character_evidence(
    registry_item: dict[str, Any],
    normalized: dict[str, Any],
    scene_index: int,
    *,
    visual_confidence: float | None = None,
    identity_confidence: float | None = None,
    identity_status: str | None = None,
) -> None:

    # Nome só é aceito se for uma informação explícita.
    name = normalized.get("name")

    if name and _is_unknown(name):
        name = None

    if name:
        current_name = registry_item.get("name")

        if not current_name:
            registry_item["name"] = name
        elif _normalize_text(current_name) != _normalize_text(name):
            aliases = registry_item.setdefault(
                "aliases",
                [],
            )

            if name not in aliases:
                aliases.append(name)

    for speaker_id in normalized.get(
        "speaker_ids",
        [],
    ):
        if speaker_id not in registry_item["speaker_ids"]:
            registry_item["speaker_ids"].append(speaker_id)

    for association in normalized.get(
        "speaker_associations",
        [],
    ):
        if association not in registry_item["speaker_associations"]:
            registry_item["speaker_associations"].append(association)

    voice_character = normalized.get("voice_character")

    if voice_character:
        registry_item["voice_character"] = voice_character

    voice_score = normalized.get("voice_score")

    if voice_score is not None:
        old_score = registry_item.get("voice_confidence")

        if old_score is None or voice_score > old_score:
            registry_item["voice_confidence"] = voice_score

    stable = normalized.get(
        "stable_appearance",
        {},
    )

    if isinstance(stable, dict):
        for key, value in stable.items():
            if value in (None, "", []):
                continue

            if (
                key not in registry_item["stable_appearance"]
                or not registry_item["stable_appearance"][key]
            ):
                registry_item["stable_appearance"][key] = value

    temporary = normalized.get(
        "temporary_appearance",
        {},
    )

    if isinstance(temporary, dict):
        for key, value in temporary.items():
            if value in (None, "", []):
                continue

            registry_item["temporary_appearance"][key] = value

    for feature in normalized.get(
        "distinctive_features",
        [],
    ):
        if feature not in registry_item["distinctive_features"]:
            registry_item["distinctive_features"].append(feature)

    if scene_index not in registry_item["scenes"]:
        registry_item["scenes"].append(scene_index)

    if visual_confidence is not None:
        old = registry_item.get("visual_confidence")

        if old is None or visual_confidence > old:
            registry_item["visual_confidence"] = visual_confidence

    if identity_confidence is not None:
        old = registry_item.get("identity_confidence")

        if old is None or identity_confidence > old:
            registry_item["identity_confidence"] = identity_confidence

    if identity_status:
        priority = {
            "AMBÍGUO": 0,
            "PROVÁVEL": 1,
            "CONFIRMADO": 2,
        }

        current = registry_item.get(
            "identity_status",
            "AMBÍGUO",
        )

        if priority.get(
            identity_status,
            0,
        ) >= priority.get(
            current,
            0,
        ):
            registry_item["identity_status"] = identity_status


# ---------------------------------------------------------------------------
# Helpers de identidade
# ---------------------------------------------------------------------------


def _find_character_by_speaker(
    registry: dict[str, dict[str, Any]],
    speaker_ids: list[str],
) -> str | None:

    if not speaker_ids:
        return None

    speaker_ids = set(speaker_ids)

    for character_id, character in registry.items():
        known_speakers = set(
            character.get(
                "speaker_ids",
                [],
            )
        )

        if speaker_ids & known_speakers:
            return character_id

    return None


def _next_character_id(
    registry: dict[str, dict[str, Any]],
) -> str:

    number = 1

    while True:
        candidate = f"character_{number:03d}"

        if candidate not in registry:
            return candidate

        number += 1


def _get_visual_characters(
    scene: dict[str, Any],
) -> list[dict[str, Any]]:

    visual = scene.get(
        "visual_descriptors",
        {},
    )

    if isinstance(visual, list):
        return [item for item in visual if isinstance(item, dict)]

    if isinstance(visual, dict):
        characters = visual.get(
            "characters",
            [],
        )

        if isinstance(characters, list):
            return [item for item in characters if isinstance(item, dict)]

    return []


# ---------------------------------------------------------------------------
# Resolução principal
# ---------------------------------------------------------------------------


def resolve_scene_characters(
    scenes: list[dict[str, Any]],
    *,
    initial_registry: dict[str, dict[str, Any]] | None = None,
    visual_weight: float = 0.40,
    voice_weight: float = 0.35,
    temporal_weight: float = 0.15,
    context_weight: float = 0.10,
    confirmed_threshold: float = 0.80,
    probable_threshold: float = 0.65,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:

    registry = {}

    if initial_registry:
        for character_id, character in initial_registry.items():
            registry[character_id] = dict(character)

    resolved_scenes = []

    for scene_index, scene in enumerate(scenes):

        visual_characters = _get_visual_characters(scene)

        voice_evidence = extract_voice_evidence(scene)

        scene_resolved = []

        speaker_mapping = {}

        for visual_character in visual_characters:

            normalized = normalize_character_descriptor(visual_character)

            speaker_ids = normalized.get(
                "speaker_ids",
                [],
            )

            # ---------------------------------------------------------------
            # 1. Primeiro: speaker_id conhecido
            # ---------------------------------------------------------------

            direct_character_id = _find_character_by_speaker(
                registry,
                speaker_ids,
            )

            match = None

            if direct_character_id:
                known = registry[direct_character_id]

                visual = visual_similarity(
                    normalized,
                    known,
                )

                voice = voice_score_for_character(
                    normalized,
                    known,
                    voice_evidence,
                )

                temporal = temporal_similarity(
                    normalized,
                    known,
                )

                context = context_similarity(
                    normalized,
                    known,
                )

                identity_score = (
                    visual * visual_weight
                    + voice * voice_weight
                    + temporal * temporal_weight
                    + context * context_weight
                )

                # Se houver uma contradição visual muito forte,
                # não forçamos a identidade.
                if visual >= 0.15 or voice >= 0.75:
                    status = "CONFIRMADO"

                    if identity_score < probable_threshold:
                        status = "PROVÁVEL"

                    match = IdentityMatch(
                        character_id=direct_character_id,
                        visual_score=visual,
                        voice_score=voice,
                        temporal_score=temporal,
                        context_score=context,
                        identity_score=identity_score,
                        status=status,
                        margin=1.0,
                        candidate_character_ids=[direct_character_id],
                    )

            # ---------------------------------------------------------------
            # 2. Matching geral
            # ---------------------------------------------------------------

            if match is None and registry:

                match = match_character(
                    normalized,
                    list(registry.values()),
                    voice_evidence=voice_evidence,
                    visual_weight=visual_weight,
                    voice_weight=voice_weight,
                    temporal_weight=temporal_weight,
                    context_weight=context_weight,
                    confirmed_threshold=confirmed_threshold,
                    probable_threshold=probable_threshold,
                )

            # ---------------------------------------------------------------
            # 3. Decisão
            # ---------------------------------------------------------------

            character_id = None

            if match and match.status in {
                "CONFIRMADO",
                "PROVÁVEL",
            }:
                character_id = match.character_id

            elif match and match.status == "AMBÍGUO":

                # IMPORTANTE:
                # Não criamos outro personagem apenas porque a cena
                # ficou ambígua.
                #
                # Isso evita:
                #
                # cena 1 -> character_001
                # cena 2 -> character_002
                # cena 3 -> character_001
                #
                # quando na verdade é o mesmo personagem.

                candidate_ids = match.candidate_character_ids or []

                if len(candidate_ids) == 1:
                    character_id = candidate_ids[0]

            # ---------------------------------------------------------------
            # 4. Primeiro aparecimento sem candidato
            # ---------------------------------------------------------------

            if character_id is None and not registry:
                character_id = _next_character_id(registry)

                registry[character_id] = _new_character(
                    character_id,
                    normalized,
                    scene_index,
                )

                match = IdentityMatch(
                    character_id=character_id,
                    visual_score=1.0,
                    voice_score=0.5,
                    temporal_score=0.5,
                    context_score=0.5,
                    identity_score=0.5,
                    status="PROVÁVEL",
                    margin=1.0,
                    candidate_character_ids=[character_id],
                )

            # ---------------------------------------------------------------
            # 5. Se ainda não temos identidade:
            #    manter AMBÍGUO sem criar personagem artificial.
            # ---------------------------------------------------------------

            if character_id is not None:

                registry_item = registry.get(character_id)

                if registry_item is None:
                    registry_item = _new_character(
                        character_id,
                        normalized,
                        scene_index,
                    )

                    registry[character_id] = registry_item

                _merge_character_evidence(
                    registry_item,
                    normalized,
                    scene_index,
                    visual_confidence=(match.visual_score if match else None),
                    identity_confidence=(match.identity_score if match else None),
                    identity_status=(match.status if match else "PROVÁVEL"),
                )

            # ---------------------------------------------------------------
            # 6. Resultado da cena
            # ---------------------------------------------------------------

            result = {
                "scene_character_ref": normalized.get("character_ref"),
                "character_id": character_id,
                "name": (
                    registry.get(
                        character_id,
                        {},
                    ).get("name")
                    if character_id
                    else normalized.get("name")
                ),
                "status": (match.status if match else "AMBÍGUO"),
                "identity_confidence": (
                    round(
                        match.identity_score,
                        4,
                    )
                    if match
                    else 0.0
                ),
                "identity_margin": (
                    round(
                        match.margin,
                        4,
                    )
                    if match
                    else 0.0
                ),
                "visual_score": (
                    round(
                        match.visual_score,
                        4,
                    )
                    if match
                    else 0.0
                ),
                "voice_score": (
                    round(
                        match.voice_score,
                        4,
                    )
                    if match
                    else 0.0
                ),
                "temporal_score": (
                    round(
                        match.temporal_score,
                        4,
                    )
                    if match
                    else 0.0
                ),
                "context_score": (
                    round(
                        match.context_score,
                        4,
                    )
                    if match
                    else 0.0
                ),
                "speaker_ids": speaker_ids,
                "speaker_associations": normalized.get(
                    "speaker_associations",
                    [],
                ),
                "voice_evidence": voice_evidence,
                "candidate_character_ids": (
                    match.candidate_character_ids if match else []
                ),
            }

            scene_resolved.append(result)

            # ---------------------------------------------------------------
            # Speaker -> Character
            # ---------------------------------------------------------------

            if character_id:

                for speaker_id in speaker_ids:
                    speaker_mapping[speaker_id] = {
                        "character_id": character_id,
                        "character_name": registry[character_id].get("name"),
                        "status": (match.status if match else "AMBÍGUO"),
                        "confidence": (match.identity_score if match else 0.0),
                    }

        resolved_scenes.append(
            {
                "scene_id": scene.get(
                    "scene_id",
                    scene_index,
                ),
                "start": scene.get("start"),
                "end": scene.get("end"),
                "resolved_characters": scene_resolved,
                "speaker_character_mapping": speaker_mapping,
            }
        )

    return resolved_scenes, registry


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------


def format_character_registry(
    registry: dict[str, dict[str, Any]],
) -> str:

    lines = []

    for character_id, character in sorted(registry.items()):
        name = character.get("name") or "NÃO IDENTIFICADO"

        status = character.get(
            "identity_status",
            "AMBÍGUO",
        )

        confidence = character.get("identity_confidence")

        speakers = (
            ", ".join(
                character.get(
                    "speaker_ids",
                    [],
                )
            )
            or "nenhum"
        )

        scenes = (
            ", ".join(
                str(scene)
                for scene in character.get(
                    "scenes",
                    [],
                )
            )
            or "nenhuma"
        )

        lines.append(
            f"{character_id} | "
            f"{name} | "
            f"{status} | "
            f"confiança={confidence} | "
            f"speakers={speakers} | "
            f"cenas={scenes}"
        )

    return "\n".join(lines)
