from typing import Any, Dict, Tuple

from cprt.data.cprt_datamodule import CprtDataModule
from cprt.data.data_utils import load_data_from_path, random_split_df


def creat_datamodule(
    data_dict_path: str,
    data_df_path: str,
    split_ratios: Tuple[float, float, float],
    datamodule_config: Dict[str, Any],
    only_keep_questions: bool = True,
) -> CprtDataModule:
    """Create the CprtDatamodule."""
    data_dict = load_data_from_path(data_dict_path)
    if only_keep_questions:
        data_dict = {
            k: {"sequence": v["sequence"], "info": [i for i in v["info"] if "?" in i]}  # type: ignore[misc]
            for k, v in data_dict.items()
        }
    data_df = load_data_from_path(data_df_path)
    data_df = data_df[data_df["uniprot_id"].isin(data_dict)]  # type: ignore[index]
    # TODO: to be removed after data update
    data_df["protein_length"] = data_df["uniprot_id"].apply(lambda x: len(data_dict[x]["sequence"]))
    data_df = data_df[data_df["protein_length"] < datamodule_config["max_protein_length"]]
    data_df.reset_index(drop=True, inplace=True)
    random_split_df(data_df, split_ratios)
    return CprtDataModule(data_dict, data_df, **datamodule_config)  # type: ignore[arg-type]
