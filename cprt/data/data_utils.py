import csv
import os
import pickle
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


def file_path_assertions(file_path: str, exists_ok: bool, strict_extension: str | None = None) -> Tuple[str, str]:
    """
    Check validity of file_path and create parent dirs.

    :param file_path: file path to analyse/
    :param exists_ok: If False raises Exception when file exists. If True raises a warning.
    :param strict_extension: Whether to strictly check for a specific extension.

    :returns file name and file extension
    """
    base_name = os.path.basename(file_path)
    assert base_name, f"file path {file_path} should not point to a directory."
    assert "." in base_name, f"specify an extension or the file {file_path}"
    if strict_extension is not None:
        assert file_path.endswith(
            strict_extension.strip(".")
        ), f"file path must be a {strict_extension} file. {file_path} was given."

    if exists_ok:
        if os.path.isfile(file_path):
            logger.warning(f"{file_path} already exists. File will be overwritten, ensure this is intended.")
    else:
        assert not os.path.isfile(file_path), f"file {file_path} already present, provide a new file name."

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fn, ext = os.path.splitext(base_name)
    return fn, ext


def add_line_to_csv(row: Tuple[str, str, str, int, int], out_file_name: str) -> None:
    """Append a new row to csv file."""
    with open(out_file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def load_data_from_path(data_path: str) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Load .pkl dicts or .csv tables."""
    if data_path.endswith(".pkl") or data_path.endswith(".pickle"):
        with open(data_path, "rb") as f:
            data_dict: Dict[str, Any] = pickle.load(f)
            return data_dict
    elif data_path.endswith(".csv"):
        data_df: pd.DataFrame = pd.read_csv(data_path)
        return data_df
    else:
        raise ValueError("only supports .csv and .pkl/.pickle files.")


def random_split_df(df: pd.DataFrame, ratios: Sequence[float], key: str = "uniref_id") -> None:
    """Add split column values to df, inplace."""
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
