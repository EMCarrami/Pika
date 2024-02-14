import os
import pickle
import random
import re
from glob import glob
from typing import Any, Dict, List

from loguru import logger
from tqdm import tqdm

from pika.data.data_utils import file_path_assertions


def postprocess_gpt_results(
    path_to_data_dict: str,
    gpt_output_dir: str,
    summary_qa_file_suffix: str,
    metrics_file_suffix: str,
    out_file_path: str,
) -> None:
    """
    Postprocess and merge results from chatGPT into a single dict.

    For each entry gathers summary/qa and metrics responses, performs postprocessing and saves the final data dict.
    Adds additional questions for is_enzyme and is_real to the pool of qa
    :param path_to_data_dict: path to the sub-sampled data_dict used for GPT processing.
    :param gpt_output_dir: directory that was used to save gpt results
    :param summary_qa_file_suffix: suffix used for summary/qa responses pickle file: {uid}_{suffix}.pkl
    :param metrics_file_suffix: suffix used for metrics responses pickle file: {uid}_{suffix}.pkl
    :param out_file_path: Path to output file. Must not already exist.

    :return Creates the final pickled dataset file.
            Containing uniref_id, sequence, fields, summary, qa, metrics values for each uniprot_id
    """
    file_path_assertions(out_file_path, exists_ok=False)
    with open(path_to_data_dict, "rb") as f:
        uniref_dict = pickle.load(f)

    summary_qa_files = glob(f"{gpt_output_dir}/*{summary_qa_file_suffix}.pkl")
    metrics_files = glob(f"{gpt_output_dir}/*{metrics_file_suffix}.pkl")
    assert len(summary_qa_files) == len(metrics_files), "number of summary/qa and metrics files do not match."
    logger.info(f"found {len(summary_qa_files)} summary/qa files and metrics files")

    logger.info("confirming uniref_dict ids match with summary/qa and metrics files")
    all_uids = set(uniref_dict.keys())
    all_summary_ids = set([os.path.basename(i).split(summary_qa_file_suffix)[0] for i in summary_qa_files])
    all_metric_ids = set([os.path.basename(i).split(metrics_file_suffix)[0] for i in metrics_files])
    assert all_uids == all_summary_ids, (
        "summary/qa files do not match ids in uniref dict. "
        f"Missing files: {all_uids - all_summary_ids} \n Extra files: {all_summary_ids - all_uids}"
    )
    assert all_uids == all_metric_ids, (
        "metrics files do not match ids in uniref dict. "
        f"Missing files: {all_uids - all_metric_ids} \n Extra files: {all_metric_ids - all_uids}"
    )

    out = {}
    bad_entries = []
    for idx, (fl1, fl2) in tqdm(enumerate(zip(summary_qa_files, metrics_files)), total=len(summary_qa_files)):
        uid = os.path.basename(fl1).split(summary_qa_file_suffix)[0]
        with open(fl1, "rb") as f:
            summary_qa = pickle.load(f)["choices"][0]["message"]["content"].split("\n")
        with open(fl2, "rb") as f:
            metrics = pickle.load(f)["choices"][0]["message"]["content"].split("\n")
        summary_qa = merge_qa([i.strip() for i in summary_qa if len(i.strip()) > 0 and i.strip() != "QA pairs:"])
        metrics = merge_qa([i.strip() for i in metrics if len(i.strip()) > 0])
        if (
            len(summary_qa) > 1
            and summary_qa[0].lower().startswith("summary:")
            and all([len(i.split("?")) == 2 for i in summary_qa[1:]])
        ):
            info_fields = [
                f"{k}: {', '.join(v)}" for k, v in uniref_dict[uid].items() if k not in ["sequence", "uniref_id"]
            ]
            aggregate_metrics = assign_metrics(metrics)
            assert isinstance(aggregate_metrics["is_enzyme"], bool)
            summary_qa.append(get_enzyme_qa(aggregate_metrics["is_enzyme"]))
            summary_qa.append(get_is_real_question())

            aggregate_metrics["length"] = int(uniref_dict[uid]["protein size"][0].split()[0])
            aggregate_metrics["mw"] = int(uniref_dict[uid]["protein size"][1].split()[0])
            out[uid] = {
                "uniref_id": uniref_dict[uid]["uniref_id"],
                "sequence": uniref_dict[uid]["sequence"],
                "fields": info_fields,
                "summary": summary_qa[0][9:].split(". "),
                "qa": summary_qa[1:],
                "metrics": aggregate_metrics,
            }
        else:
            bad_entries.append(uid)
    logger.info(f"{len(out)} entries were processed successfully with {len(bad_entries)} bad entries: {bad_entries}")
    with open(out_file_path, "wb") as f:
        pickle.dump(out, f)


def merge_qa(in_list: List[str]) -> List[str]:
    """Merge questions into their respective answers removing the gap in lines."""
    out = []
    if len(in_list) > 0:
        new_entry = in_list[0]
        for e in in_list[1:]:
            if is_question_format_correct(e):
                out.append(fix_entry(new_entry))
                new_entry = e
            else:
                new_entry += f" {e}"
        out.append(fix_entry(new_entry))
    return out


def is_question_format_correct(q: str) -> bool:
    r"""
    Check if question follows the expected patterns.

    accepts:
    'n) ' -> r'^\d+\)\s'
    'n. ' -> r'^\d+\.\s'
    '- ' -> r'^-\s'
    """
    valid_pattern = r"^(\d+\)\s)|(\d+\.\s)|(-\s)"
    return bool(re.match(valid_pattern, q.strip())) and "?" in q


def fix_entry(ne: str) -> str:
    """Fix q/a entries by removing additional questions that GPT might have added before the answer."""
    pattern = r"^(\d+\)\s)|(\d+\.\s)|(-\s)"
    if "?" in ne:
        q = re.sub(pattern, "", ne).strip().split("?")
        return f"{q[-2]}?{q[-1]}"
    else:
        return ne


def assign_metrics(metrics: List[str]) -> Dict[str, Any]:
    """Assign standardised metric values for each entry."""
    out = {
        "localization": "none",
        "in_membrane": False,
        "in_nucleus": False,
        "in_mitochondria": False,
        "DNA_binding": False,
        "RNA_binding": False,
        "nucleic_acid_binding": False,
        "is_enzyme": False,
        "cofactor": "none",
    }
    for q in metrics:
        if len(q.split("?")) == 2:
            if "yes" in q.split("?")[1].strip().lower():
                if "membrane" in q:
                    out["in_membrane"] = True
                    out["localization"] = "membrane"
                elif "nucleus" in q:
                    out["in_nucleus"] = True
                    out["localization"] = "nucleus"
                elif "mitochondria" in q:
                    out["in_mitochondria"] = True
                    out["localization"] = "mitochondrion"
                elif "DNA" in q:
                    out["DNA_binding"] = True
                    out["nucleic_acid_binding"] = True
                elif "RNA" in q:
                    out["RNA_binding"] = True
                    out["nucleic_acid_binding"] = True
                elif "enzyme" in q:
                    out["is_enzyme"] = True
            # for co-factors
            elif "factor" in q:
                v = q.split("?")[1].strip().lower()
                if "unknown" not in v:
                    out["cofactor"] = v
    # remove none values
    out = {k: v for k, v in out.items() if v != "none"}
    return out


def get_enzyme_qa(is_enzyme: bool) -> str:
    """Get a random question form on whether the protein is an enzyme."""
    is_enzyme_questions = [
        "Is this protein an enzyme?",
        "Does this protein function as an enzyme?",
        "Can this protein act as an enzyme?",
        "Does this protein possess the characteristics of an enzyme?",
        "Can this protein be correctly termed an enzyme?",
    ]
    assert isinstance(is_enzyme, bool)
    if str(is_enzyme) == "False":
        return f"{random.choice(is_enzyme_questions)} No"
    elif str(is_enzyme) == "True":
        return f"{random.choice(is_enzyme_questions)} Yes"
    else:
        raise ValueError(f"is_enzyme must be 0 or 1. {is_enzyme} was given.")


def get_is_real_question() -> str:
    """Get a random question form on whether the protein is a real protein."""
    is_real_questions = [
        "Is this a real protein?",
        "Does this sequence represent a real protein?",
        "Is this sequence from an actual protein?",
        "Does this sequence belong to a true protein?",
        "Does the given sequence correspond to a genuine protein?",
    ]
    return f"{random.choice(is_real_questions)} yes_real"
