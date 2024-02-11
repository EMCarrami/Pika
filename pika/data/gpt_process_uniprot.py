import os
import pickle
from os.path import isfile
from typing import Any, Dict

from pika.data.gpt_instructions import METRIC_INSTRUCTIONS, SUMMARY_INSTRUCTIONS
from pika.utils import DATA_PATH
from pika.utils.chatgpt_processor import GPTProcessor


def process_fields_with_gpt(save_dir: str = "datasets/gpt3", num_workers: int = 20) -> None:
    """
    Process all fields of uniref50_subsample_data.pkl with ChatGPT.

    :param save_dir: directory to save the results
    :param num_workers: number of parallel GPT requests.
    """
    with open(f"{DATA_PATH}/uniref50_subsample_data.pkl", "rb") as f:
        uniref_dict = pickle.load(f)
    name_suffix1 = "_r1"
    name_suffix2 = ""
    tasks = {}
    for uid, fields in uniref_dict.items():
        if not (isfile(f"{save_dir}/{uid}{name_suffix1}.pkl") or isfile(f"{save_dir}/{uid}{name_suffix2}.pkl")):
            tasks[uid] = "\n".join(
                [f"{k}: {', '.join(v)}" for k, v in fields.items() if k not in ["sequence", "uniref_id"]]
            )
    if len(tasks) > 0:
        get_gpt_summary(
            tasks, SUMMARY_INSTRUCTIONS, num_workers=num_workers, save_dir=save_dir, name_suffix=name_suffix1
        )
    tasks = {}
    for uid in uniref_dict:
        if not os.path.isfile(f"{save_dir}/{uid}{name_suffix2}.pkl"):
            assert os.path.isfile(f"{save_dir}/{uid}{name_suffix1}.pkl"), f"summary file is missing for {uid}"
            with open(f"{save_dir}/{uid}{name_suffix1}.pkl", "rb") as f:
                r1 = pickle.load(f)
            tasks[uid] = r1["choices"][0]["message"]["content"]
    if len(tasks) > 0:
        get_gpt_summary(
            tasks, METRIC_INSTRUCTIONS, num_workers=num_workers, save_dir=save_dir, name_suffix=name_suffix2
        )


def get_gpt_summary(
    data_dict: Dict[str, Any], instruction: str, num_workers: int, save_dir: str, name_suffix: str
) -> None:
    """Run parallel requests to ChatGPT for summarising protein fields."""
    gpt = GPTProcessor(model="gpt-3.5-turbo-0613", secondary_model="gpt-3.5-turbo-16k-0613")
    tasks, names = [], []
    for uid, txt in data_dict.items():
        tasks.append(
            [
                {"role": "system", "content": "You are a helpful assistant following all instructions exactly."},
                {"role": "user", "content": instruction},
                {"role": "user", "content": txt},
            ]
        )
        names.append(f"{uid}{name_suffix}")
    gpt.bulk_process(tasks, names, num_workers=num_workers, save_dir=save_dir)


if __name__ == "__main__":
    process_fields_with_gpt(num_workers=100)
