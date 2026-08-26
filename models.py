from dataclasses import dataclass, field
from typing import Any

@dataclass
class ChatTurn:
    question: str
    answer: str

@dataclass
class SourceInfo:
    number: int
    title: str
    source: str
    source_type: str
    score: float
    content: str

@dataclass
class AskResponse:
    answer: str
    sources: list[SourceInfo] = field(default_factory=list)
    rewritten_query: str = ""
