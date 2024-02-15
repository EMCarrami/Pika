import os
import pickle
from os.path import isfile
from typing import Any, Dict

from loguru import logger

from pika.data.data_utils import file_path_assertions
from pika.data.gpt_instructions import METRIC_INSTRUCTIONS, SUMMARY_INSTRUCTIONS
from pika.data.postprocess_gpt_results import postprocess_gpt_results
from pika.utils.chatgpt_processor import GPTProcessor


def create_final_dataset_with_gpt(
    path_to_debiased_data_dict: str = "dataset/uniref50_gzip_subsample_data.pkl",
    temp_dir: str = "dataset/gpt",
    out_file_path: str = "",
    num_gpt_workers: int = 100,
) -> None:
    """
    Create final Pika dataset by applying ChatGPT to filtered uniprot data.

    :param path_to_debiased_data_dict: path to the sub-sampled data_dict to be used for GPT processing
    :param temp_dir: temp_dir to store GPT responses.
    :param out_file_path: Path to output file. Must not already exist.
    :param num_gpt_workers: number of parallel GPT requests to submit. If 0 processing will be done serially.
    :return:
    """
    file_path_assertions(out_file_path, exists_ok=False)
    sq_suffix = "_r1"
    m_suffix = "_r2"
    process_fields_with_gpt(
        path_to_debiased_data_dict,
        output_dir=temp_dir,
        summary_qa_file_suffix=sq_suffix,
        metrics_file_suffix=m_suffix,
        num_workers=num_gpt_workers,
    )
    logger.info("completed GPT processing.")
    postprocess_gpt_results(
        path_to_data_dict=path_to_debiased_data_dict,
        gpt_output_dir=temp_dir,
        summary_qa_file_suffix=sq_suffix,
        metrics_file_suffix=m_suffix,
        out_file_path=out_file_path,
    )


def process_fields_with_gpt(
    path_to_debiased_data_dict: str = "dataset/uniref50_gzip_subsample_data.pkl",
    output_dir: str = "dataset/gpt",
    summary_qa_file_suffix: str = "_r1",
    metrics_file_suffix: str = "_r2",
    num_workers: int = 20,
) -> None:
    """
    Process all fields of sub-sampled entries with ChatGPT, saving each response file individually.

    :param path_to_debiased_data_dict: path to the sub-sampled data_dict to be used for GPT processing.
    :param output_dir: directory to save the results
    :param summary_qa_file_suffix: suffix to use for summary/qa responses pickle file: {uid}_{suffix}.pkl
    :param metrics_file_suffix: suffix to use for metrics responses pickle file: {uid}_{suffix}.pkl
    :param num_workers: number of parallel GPT requests.
    :return saves two files for each entry
            - {output_dir}/{uid}_r1.pkl for summary/qa
            - {output_dir}/{uid}_r2.pkl for metrics
    """
    with open(path_to_debiased_data_dict, "rb") as f:
        uniref_dict = pickle.load(f)

    tasks = {}
    for uid, fields in uniref_dict.items():
        if not (
            isfile(f"{output_dir}/{uid}{summary_qa_file_suffix}.pkl")
            or isfile(f"{output_dir}/{uid}{metrics_file_suffix}.pkl")
        ):
            tasks[uid] = "\n".join(
                [f"{k}: {', '.join(v)}" for k, v in fields.items() if k not in ["sequence", "uniref_id"]]
            )
    if len(tasks) > 0:
        get_gpt_summary(
            tasks,
            SUMMARY_INSTRUCTIONS,
            num_workers=num_workers,
            save_dir=output_dir,
            name_suffix=summary_qa_file_suffix,
        )
    tasks = {}
    for uid in uniref_dict:
        if not os.path.isfile(f"{output_dir}/{uid}{metrics_file_suffix}.pkl"):
            assert os.path.isfile(
                f"{output_dir}/{uid}{summary_qa_file_suffix}.pkl"
            ), f"summary file is missing for {uid}"
            with open(f"{output_dir}/{uid}{summary_qa_file_suffix}.pkl", "rb") as f:
                r1 = pickle.load(f)
            tasks[uid] = r1["choices"][0]["message"]["content"]
    if len(tasks) > 0:
        get_gpt_summary(
            tasks, METRIC_INSTRUCTIONS, num_workers=num_workers, save_dir=output_dir, name_suffix=metrics_file_suffix
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
