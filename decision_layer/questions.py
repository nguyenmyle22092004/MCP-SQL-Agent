from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

QuestionType: TypeAlias = Literal["count", "average", "ranking", "lookup", "comparison", "other"]
ENTITY_GROUNDING = "entity_grounding"
NEEDS_CLARIFICATION = "needs_clarification"
ANSWER_WRONG = "answer_wrong"
QUESTION_TYPE = "question_type"


@dataclass(frozen=True)
class Predicate:
    name: str
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Choice:
    name: str
    choices: tuple[str, ...]
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    name: str
    features: dict[str, float] = field(default_factory=dict)
