import random


def shuffle_protein(seq: str) -> str:
    """Shuffle the protein sequence except for the first and last 5 aa."""
    mid = list(seq[5:-5])
    random.shuffle(mid)
    return seq[:5] + "".join(mid) + seq[-5:]
