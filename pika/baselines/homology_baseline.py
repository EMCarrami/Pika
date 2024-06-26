import csv
import pickle
from typing import Any, Dict

import pandas as pd
import torch
from lightning.pytorch import seed_everything

import wandb
from pika.datamodule.pika_datamodule import PikaDataModule
from pika.metrics.biochem_lite_metrics import BiochemLiteMetrics
from pika.metrics.gpt_assessment import get_gpt_metrics
from pika.utils.helpers import load_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_homology_baseline_lite(config: Dict[str, Any]) -> None:
    """Compute metrics based on the info of the closest protein sequence in training set."""
    if "seed" in config:
        seed_everything(config["seed"])
    with open(config["datamodule"]["data_dict_path"], "rb") as f:
        data_dict: Dict[str, Any] = pickle.load(f)
    with open(config["baseline"]["homology_dict_path"], "rb") as f:
        hmg_dict = pickle.load(f)

    config["datamodule"]["language_model"] = "gpt2"
    config["datamodule"]["protein_model"] = "esm2_t12_35M_UR50D"
    datamodule = PikaDataModule(**config["datamodule"])

    def response_map(uid: str, metric_name: str) -> str:
        """Get homologue response."""
        for h in hmg_dict[uid]:
            if metric_name in data_dict[h]["metrics"]:
                out: str = data_dict[h]["metrics"][metric_name]
                return out
        return "none"

    bool_map = {"False": "no", "True": "yes"}
    metric = BiochemLiteMetrics()
    df = datamodule.val_metric_dataset.split_df
    for n in df.metric.unique():
        lbl = df[df.metric == n]["value"].to_list()
        preds = df[df.metric == n]["uniprot_id"].map(lambda x: response_map(x, n)).to_list()
        preds = [bool_map[str(i)] if isinstance(i, bool) else str(i) for i in preds]
        metric.update(preds, lbl, [n] * len(lbl))

    config["wandb"]["group"] = "homology_baseline"
    config["wandb"]["name"] = f"homology_baseline_{config['seed']}"
    wandb.init(**config["wandb"], config=config)
    wandb.log({f"biochem/val_{k}": v for k, v in metric.compute().items()})
    wandb.finish()


def get_homology_baseline_react(config: Dict[str, Any]) -> None:
    """Get reAct values based on the info of the closest protein sequence in training set."""
    if "seed" in config:
        seed_everything(config["seed"])
    with open(config["datamodule"]["data_dict_path"], "rb") as f:
        data_dict: Dict[str, Any] = pickle.load(f)
    with open(config["baseline"]["homology_dict_path"], "rb") as f:
        hmg_dict = pickle.load(f)

    config["datamodule"]["language_model"] = "gpt2"
    config["datamodule"]["protein_model"] = "esm2_t12_35M_UR50D"
    datamodule = PikaDataModule(**config["datamodule"])

    def response_map(uid: str, metric_name: str) -> str:
        """Get homologue response."""
        for h in hmg_dict[uid]:
            if any([i.startswith(metric_name) for i in data_dict[h]["fields"]]):
                out: str = [i for i in data_dict[h]["fields"] if i.startswith(metric_name)][0].split(":")[-1]
                return out
        return "none"

    df: pd.DataFrame = datamodule.test_df
    test_results = []
    for n in df.subjects.unique():
        lbl = df[df.subjects == n]["ground_truth"].to_list()
        preds = df[df.subjects == n]["uniprot_id"].map(lambda x: response_map(x, n)).to_list()
        for id, expc, gen in zip(df[df.subjects == n]["uniprot_id"].to_list(), lbl, preds):
            test_results.append([id, n, expc, gen])

    col_names = ["uniprot_id", "subject", "expected_answer", "generated_response"]
    with open(config["baseline"]["out_save_path"], "a", newline="") as f:  # type: ignore[assignment]
        writer = csv.writer(f, delimiter="\t")  # type: ignore[arg-type]
        writer.writerow(col_names)
        for row in test_results:
            writer.writerow(row)

    config["wandb"]["group"] = "homology_react"
    if "name" not in config["wandb"]:
        config["wandb"]["name"] = f"homology_react_{config['seed']}"
    wandb.init(**config["wandb"], config=config)
    test_table = wandb.Table(columns=col_names)
    for v in test_results:
        test_table.add_data(*v)
    wandb.log({"Biochem-ReAct_results": test_table})
    wandb.finish()


if __name__ == "__main__":
    c = load_config("../../configs/homology_config.json")
    # get_homology_baseline_lite(c)
    # get_homology_baseline_react(c)
    # import pickle
    # # with open("split_0_closest_homologue_sim_score.pkl", "rb") as f:
    # with open("test_sim_scores.pkl", "rb") as f:
    #     scores = pickle.load(f)
    # subsample = [k for k, v in scores.items() if v < 0.36]
    get_gpt_metrics(
        table_path="../metrics/results/evo_train_domains_results.tsv",
        wandb_project="Cprt-Paper-Tests",
        wandb_run_id="s92rs3di",
        subjects=["functional domains"],
        subsample=1.0,
    )
    # get_gpt_metrics(
    #     table_path="../metrics/results/evo_train_react_results.tsv",
    #     wandb_project="Cprt-Paper-Tests",
    #     wandb_run_id="s92rs3di",
    #     subjects=["cofactor"],
    #     subsample=1.0,
    # )
