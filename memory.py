from collections import deque
from config import MEMORY_TURNS
from models import ChatTurn

class ConversationMemory:
    def __init__(self, max_turns=MEMORY_TURNS):
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)

    def add(self, question: str, answer: str):
        self.turns.append(ChatTurn(question, answer))

    def clear(self):
        self.turns.clear()

    def as_text(self) -> str:
        if not self.turns:
            return "No previous conversation."
        return "\n\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}"
            for turn in self.turns
        )
