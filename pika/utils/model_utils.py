import random
from typing import Literal


def shuffle_protein(seq: str, constant_ends: int = 5) -> str:
    """Shuffle the protein sequence except for the first and last constant_ends aa."""
    mid = list(seq[constant_ends:-constant_ends])
    random.shuffle(mid)
    return seq[:constant_ends] + "".join(mid) + seq[-constant_ends:]


def get_is_real_question(response: Literal["Yes", "No"]) -> str:
    """Get a random question form on whether the protein is a real protein, with a specified response."""
    is_real_questions = [
        "Is this a real protein?",
        "Does this sequence represent a real protein?",
        "Is this sequence from an actual protein?",
        "Does this sequence belong to a true protein?",
        "Does the given sequence correspond to a genuine protein?",
    ]
    return f"{random.choice(is_real_questions)} {response}"
