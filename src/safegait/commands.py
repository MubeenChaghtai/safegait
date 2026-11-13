from __future__ import annotations

from dataclasses import dataclass

_VERBS = {
    "reach": "reach",
    "touch": "reach",
    "pick": "pick",
    "grab": "pick",
    "grasp": "pick",
}


@dataclass(frozen=True)
class Task:
    kind: str
    target: str


def parse_command(text: str, known_objects: list[str]) -> Task | None:
    words = [w.strip(".,!?") for w in text.lower().split()]
    kinds = [_VERBS[w] for w in words if w in _VERBS]
    if not kinds:
        return None

    full = [name for name in known_objects if all(p in words for p in name.split("_"))]
    if len(full) == 1:
        return Task(kinds[0], full[0])
    if len(full) > 1:
        return None

    by_noun = [name for name in known_objects if name.split("_")[-1] in words]
    return Task(kinds[0], by_noun[0]) if len(by_noun) == 1 else None
