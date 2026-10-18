"""Intentionally explicit, independently inspectable fact memory.

This is an *oracle address binder*, not a learned language understanding model.
The scientific question is WHERE an already known fact enters a frozen model.
"""
import json
from pathlib import Path

COLORS = ('red', 'blue', 'green', 'yellow')


class FactMemory:
    def __init__(self):
        self.facts: dict[str, str] = {}
        self.events: list[tuple[str, str, str | None]] = []

    def observe(self, name: str, color: str):
        name, color = name.strip(), color.strip().lower()
        if not name or '\n' in name or len(name) > 80:
            raise ValueError('Enter a name of 1-80 characters')
        if color not in COLORS:
            raise ValueError(f'Color must be one of {COLORS}')
        old = self.facts.get(name.casefold())
        self.facts[name.casefold()] = color
        self.events.append((name, color, old))
        return old

    def get(self, name: str) -> str | None:
        return self.facts.get(name.strip().casefold())

    def save(self, filename):
        Path(filename).write_text(json.dumps(self.facts, indent=2), encoding='utf-8')

    def load(self, filename):
        obj = json.loads(Path(filename).read_text(encoding='utf-8'))
        if not isinstance(obj, dict) or not all(isinstance(n, str) and isinstance(c, str) for n, c in obj.items()):
            raise ValueError('Expected a JSON dictionary of names to colors')
        new = FactMemory()
        for n, c in obj.items():
            new.observe(n, c)
        self.facts = new.facts
        self.events = []
