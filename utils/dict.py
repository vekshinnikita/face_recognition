

from typing import Any


def traverse(obj, *args) -> Any:
    """Копание вглубь dict'а"""
    if obj is None:
        return None

    if len(args) == 0:
        # raise ValueError("len(args) must be > 0")
        return obj[0]

    elif len(args) == 1:
        return obj[args[0]]
    else:
        return traverse(obj[args[0]], *args[1:])