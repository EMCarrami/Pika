import json
import os.path
import pickle
import string
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

import wandb
from cprt.utils.chatgpt_processor import GPTProcessor

FIRST = "Please take a deep breath and with care and attention to details answer the following question.\n"
INSTRUCTIONS = {
    "cofactor": """Given two statements about cofactors of a protein,
provide a single word answer as below:
Correct: if (2) describes the cofactors and their binding ratios in (1).
Maybe: if (2) somewhat describes the cofactors in (1).
Wrong: if (2) is completely wrong. """,
    "functional domains": """Given two statements about cofactors of a protein where (1) is the ground truths,
Provide a single word answer as below:
Correct: if domains in (2) are mostly correct and could describe a similar protein as in (1).
Maybe: if (2) somewhat describes the same protein as in (1).
Wrong: if (2) is completely wrong. """,
    "catalytic activity": """Given two statements about the catalytic activity of an enzyme,
tell me if the reaction in (2) could be the same enzyme that catalyzes the reaction in (1).
Answer with Yes or No only. """,
}
MODELS = {
    "cofactor": "gpt-3.5-turbo-1106",
    "functional domains": "gpt-3.5-turbo-1106",
    "catalytic activity": "gpt-4-0125-preview",
}


def get_gpt_metrics(
    table_path: str,
    wandb_project: str | None = "",
    wandb_run_id: str | None = None,
    subsample: float | int = 1.0,
    subjects: List[str] | None = None,
    save_dir: str = "results",
) -> None:
    """Get GPT's comparison response for a results table."""
    if wandb_project is not None:
        if wandb_run_id is not None:
            wandb.init(project=wandb_project, id=wandb_run_id, resume="must")
        else:
            wandb.init(project=wandb_project)
    file_name = table_path.split("/")[-1]
    results_path = f'{save_dir}/{file_name.replace(".ckpt", ".tsv")}'
    os.makedirs(save_dir, exist_ok=True)
    assert not os.path.isfile(results_path)
    np.random.seed(0)
    df = pd.read_csv(table_path, delimiter="\t")
    if subsample != 1:
        if isinstance(subsample, float):
            assert subsample < 1
            df = df.sample(frac=subsample)
        elif isinstance(subsample, int):
            assert subsample > 1
            df = df.sample(n=subsample)

    df["request_name"] = df["subject"].str.replace(" ", "-") + "_" + df["uniprot_id"]
    subjects = list(INSTRUCTIONS.keys()) if subjects is None else subjects
    df = df[df["subject"].isin(subjects)]
    print(df["subject"].unique())

    results_map = {}
    for subject, data in df.groupby("subject"):
        subject_path = results_path.replace(".tsv", f"_{subject.replace(' ', '-')}.pkl")
        if not os.path.isfile(subject_path):
            logger.info(f"processing {subject} with {MODELS[subject]}")
            gpt = GPTProcessor(model=MODELS[subject], secondary_model=None)
            tasks, names = [], []
            for _, row in data.iterrows():
                _, _, label, pred, name = row
                names.append(name)
                tasks.append(
                    [
                        {"role": "user", "content": f"{FIRST}{INSTRUCTIONS[subject]}"},
                        {"role": "user", "content": f"(1) {label}\n(2) {pred}\n"},
                    ]
                )
            out = gpt.bulk_process(tasks, names, num_workers=100, return_dict=True, save_dir=None)
            with open(subject_path, "wb") as f:
                pickle.dump(out, f)
        else:
            with open(subject_path, "rb") as f:
                out = pickle.load(f)
        assert out is not None
        results_map.update({k: v["choices"][0]["message"]["content"] for k, v in out.items()})

    df["results"] = df["request_name"].map(results_map)
    df.to_csv(results_path, sep="\t", index=False)

    if wandb_project is not None:
        for subject, _df in df.groupby("subject"):
            responses = _df["results"].map(harmonise_gpt_results).value_counts(normalize=True).to_dict()
            wandb.log({f"{subject}/{k}": v for k, v in responses.items()})
        wandb.finish()


def harmonise_gpt_results(result: str) -> str:
    """Harmonise GPT responses to match expected format."""
    result_map = {"yes": "correct", "no": "incorrect", "wrong": "incorrect", "right": "correct"}
    result = result.translate(str.maketrans("", "", string.punctuation))
    result = result.split()[0].lower()
    return result_map.get(result, result)


def gpt_metrics_on_wandb_project(
    project: str, subjects: List[str] | None = None, allowed_runs: List[str] | None = None, subsample: float = 1.0
) -> None:
    """Get GPT metrics on all/allowed_runs of the project."""
    subjects = list(INSTRUCTIONS.keys()) if subjects is None else subjects
    api = wandb.Api()
    runs = api.runs(project)
    run_data = {}
    for r in runs:
        config = json.loads(r.json_config)
        if allowed_runs is None or r.id in allowed_runs:
            if any([any([i in m for m in r.summary.keys()]) for i in subjects]) and "checkpoint" in config:
                print(r.name)
                ckpt = config["checkpoint"]["value"]["path"]
                if "name" in config["wandb"]["value"]:
                    run_data[r.id] = f"{config['wandb']['value']['name']}_{ckpt.replace('.ckpt', '.tsv')}"
                else:
                    run_data[r.id] = ckpt.replace(".ckpt", ".tsv")

    print(len(run_data))
    for i, path in run_data.items():
        get_gpt_metrics(
            f"../../test_results/{path}",
            wandb_project="Cprt-Paper-Tests",
            wandb_run_id=i,
            subjects=subjects,
            subsample=subsample,
        )


if __name__ == "__main__":
    get_gpt_metrics("Cprt-Paper-Tests")
