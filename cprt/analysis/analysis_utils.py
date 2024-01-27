import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd
from matplotlib.pyplot import Axes
from scipy.stats import ttest_ind, ttest_rel

import wandb

METRIC_NAMES = {
    "biochem/val_is_real_f1": "is_real F1",
    "biochem/val_is_enzyme_hard_f1": "is_enzyme F1",
    "biochem/val_kingdom_f1": "kingdom F1",
    "biochem/val_localization_f1": "localization F1",
    "biochem/val_cofactor": "cofactor Recall",
    "avg_binary_loc_f1": "binary localization\naverage F1",
    "biochem/val_is_fake_f1": "is_fake F1",
    "biochem/val_mw_error": "MW MALE",
    "metrics/val_perplexity": "perplexity",
    "metrics/val_rouge1_fmeasure": "rouge1 Score",
    "metrics/val_rouge1_precision": "rouge1 Precision",
    "metrics/val_rouge1_recall": "rouge1 Recall",
    "metrics/val_rouge2_fmeasure": "rouge2 Score",
    "metrics/val_rouge2_precision": "rouge2 Precision",
    "metrics/val_rouge2_recall": "rouge2 Recall",
    "metrics/val_rougeL_fmeasure": "rougeL Score",
    "metrics/val_rougeL_precision": "rougeL Precision",
    "metrics/val_rougeL_recall": "rougeL Recall",
    "biochem/val_in_membrane_f1": "in_membrane F1",
    "biochem/val_in_nucleus_f1": "in_nucleus F1",
    "biochem/val_in_mitochondria_f1": "in_mitochondria F1",
}
metrics_modes = {k: "min" if any([h in k for h in ["perplexity", "error"]]) else "max" for k in METRIC_NAMES}


METRICS = {
    "metric_groups": [
        [
            "biochem/val_is_real_f1",
            "biochem/val_is_enzyme_f1",
            "biochem/val_kingdom_f1",
            "biochem/val_localization_f1",
            "biochem/val_cofactor",
            "avg_binary_loc_f1",
            "biochem/val_is_fake_f1",
        ],
        ["biochem/val_mw_error"],
        [
            "metrics/val_rouge1_fmeasure",
            "metrics/val_rouge1_precision",
            "metrics/val_rouge1_recall",
            "metrics/val_rouge2_fmeasure",
            "metrics/val_rouge2_precision",
            "metrics/val_rouge2_recall",
            "metrics/val_rougeL_fmeasure",
            "metrics/val_rougeL_precision",
            "metrics/val_rougeL_recall",
        ],
        ["metrics/val_perplexity"],
    ],
    "metric_names": [
        [
            "is_real f1",
            "is_enzyme f1",
            "kingdom f1",
            "localization f1",
            "cofactor recall",
            "binary localization\naverage f1",
            "is_fake f1",
        ],
        ["mw MALE"],
        [
            "rouge1 score",
            "rouge1 precision",
            "rouge1 recall",
            "rouge2 score",
            "rouge2 precision",
            "rouge2 recall",
            "rougeL score",
            "rougeL precision",
            "rougeL recall",
        ],
        ["perplexity"],
    ],
    "metric_mode": ["max", "min", "max", "min"],
}


def get_run_data(project: str) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    """Extract all project's run data from wandb api."""
    if os.path.isfile(f"{project.replace('/', '_')}_data.pkl"):
        with open(f"{project.replace('/', '_')}_data.pkl", "rb") as f:
            out = pickle.load(f)
            data = out["data"]
            config = out["config"]
    else:
        api = wandb.Api()
        runs = api.runs(project)
        data, config = [], []
        for r in runs:
            print(r.name, end=" ")
            keys = (
                ["metrics/val_metric"]
                if "metrics/val_metric" in r.summary
                else [i for i in METRIC_NAMES.keys() if "/" in i] + ["epoch"]
            )
            data.append(pd.DataFrame(r.scan_history(keys=keys if r.summary["_step"] > 0 else None)))
            config.append(json.loads(r.json_config))
        with open(f"{project.replace('/', '_')}_data.pkl", "wb") as f:
            pickle.dump({"data": data, "config": config}, f)
    return data, config


def process_run_data(
    out: List[pd.Series], fltr_list: List[Tuple[str, str]], ordering: Tuple[str, List[str]]
) -> pd.DataFrame:
    """Process run data into a filtered dataframe."""
    df = pd.DataFrame(out).reset_index(inplace=False, drop=True)
    df["avg_binary_loc_f1"] = df[[col for col in df.columns if "_in_" in col]].mean(axis=1)
    for fltr in fltr_list:
        df = df[df[fltr[0]] == fltr[1]]
    df = df[df[ordering[0]].isin(ordering[1])]
    df[ordering[0]] = pd.Categorical(df[ordering[0]], categories=ordering[1], ordered=True)
    df.sort_values([ordering[0], "seed"], inplace=True)
    return df


def aggregate_metrics(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Aggregate metrics."""
    df = df.select_dtypes(exclude=["object"])
    return df.groupby(group_by).agg(["mean", "std"])


def add_significance_marks(
    ax: Axes, df: pd.DataFrame, var: str, group_metrics: List[str], orders: List[str], metric_mode: str
) -> None:
    """Add statistical significance between first and last bars of each plot set."""
    s1, s2 = orders[0], orders[-1]
    l, t = ax.get_ylim()
    offset = (t - l) / 30
    agg_df = aggregate_metrics(df, var)
    for i, metric in enumerate(group_metrics):
        first_layer_data = df[df[var] == s1][[metric, "seed"]]
        last_layer_data = df[df[var] == s2][[metric, "seed"]]
        if len(first_layer_data) == len(last_layer_data):
            col = "black"
            assert (first_layer_data["seed"].values == last_layer_data["seed"].values).all()
            stat, p_value = ttest_rel(
                first_layer_data[metric],
                last_layer_data[metric],
                alternative="greater" if metric_mode == "min" else "less" if metric_mode == "max" else "two-sided",
            )
        else:
            col = "red"
            stat, p_value = ttest_ind(
                first_layer_data[metric],
                last_layer_data[metric],
                alternative="greater" if metric_mode == "min" else "less" if metric_mode == "max" else "two-sided",
            )
        print(s1, s2, metric, p_value, stat)
        # Add asterisks if significant
        if p_value < 0.05:
            cnt = 3 if p_value < 0.001 else 2 if p_value < 0.01 else 1
            x1 = ax.patches[i].get_x() + ax.patches[i].get_width() / 2
            x2 = x1 + ax.patches[i].get_width() * (len(orders) - 1)
            y1 = agg_df[metric].iloc[0].sum() + offset * 0.9
            y2 = agg_df[metric].iloc[-1].sum() + offset * 0.9
            top = max(y1, y2) + offset
            # Draw lines from the first and last bars to the asterisk
            ax.plot([x1, x1, x2, x2], [y1, top, top, y2], color=col, lw=1.5)
            # Draw the asterisk
            ax.text((x1 + x2) / 2, top, cnt * "*", ha="center", va="bottom", fontsize=14, color=col)
