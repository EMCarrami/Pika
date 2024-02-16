from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger


def random_split_df(df: pd.DataFrame, ratios: Sequence[float], key: str = "uniref_id", seed: int | None = None) -> None:
    """Add split column values to df, inplace."""
    if seed is not None:
        logger.info(f"numpy seed set to {seed} for creating random splits.")
        np.random.seed(seed)
    assert len(ratios) == 3 and sum(ratios) == 1, f"3 ratio values must sum to 1. {ratios} was given."
    assert key in df.columns, f"{key} is missing in df. Choose from {df.columns}"
    id_list = df[key].unique()
    val_size, test_size = int(len(id_list) * ratios[1]), int(len(df) * ratios[2])
    train_size = len(id_list) - val_size - test_size
    split_array = np.array(
        ["train" for _ in range(train_size)] + ["val" for _ in range(val_size)] + ["test" for _ in range(test_size)]
    )
    np.random.shuffle(split_array)
    split_mapper = {k: v for k, v in zip(id_list, split_array)}
    df.loc[:, "split"] = df[key].map(split_mapper)
