import string
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import f1_score
from torchmetrics import Metric


class BiochemMetrics(Metric):
    """Class to compute biochemical metrics from textual output."""

    metric_names: List[str]
    metric_values: List[float]
    kingdom_preds: List[Tuple[str, str]]
    localization_preds: List[Tuple[str, str]]

    def __init__(self, **kwargs: Any) -> None:
        super(BiochemMetrics, self).__init__(**kwargs)
        nltk.download("vader_lexicon")
        self.sia = SentimentIntensityAnalyzer()
        self.sia_yes = self.sia.polarity_scores("Yes")["compound"]
        self.sia_no = self.sia.polarity_scores("No")["compound"]
        self.add_state("metric_names", [])
        self.add_state("metric_values", [])
        self.add_state("kingdom_preds", [])
        self.add_state("localization_preds", [])

    def update(self, predictions: List[str], labels: List[str | int], metric_names: List[str]) -> None:
        """Update metric name and value states."""
        for pred, label, name in zip(predictions, labels, metric_names):
            if isinstance(label, int):
                if name in ["mw", "length"]:
                    # get the int of predicted size only if one numeric present in the pred else set to 0
                    p = [i.replace(",", "") for i in pred.split()]
                    pnum = [i.isnumeric() for i in p]
                    pred_size = abs(int(p[pnum.index(True)])) if sum(pnum) == 1 else 0
                    pred_ratio = pred_size / label
                    self.metric_values.append(abs(np.log10(pred_ratio + 0.001)))
                    self.metric_names.append(f"{name}_error")
                else:
                    assert label in [0, 1]
                    # normalise score between sia_yes and sia_no scores
                    score = (self.sia.polarity_scores(pred)["compound"] - self.sia_no) / (self.sia_yes - self.sia_no)
                    # clamp the score between 0 and 1
                    self.metric_values.append(1 - abs(label - max(0, min(1, score))))
                    self.metric_names.append(name)
                    # NEW WAY
                    pred_t = pred.translate(str.maketrans("", "", string.punctuation)).split()
                    pos = "yes" in pred_t and "no" not in pred_t
                    neg = "no" in pred_t and "yes" not in pred_t
                    if (label == 1 and pos) or (label == 0 and neg):
                        self.metric_values.append(1)
                    else:
                        self.metric_values.append(0)
                    self.metric_names.append(f"x{name}")

            elif isinstance(label, str):
                if name == "kingdom":
                    # arch(aea), bact(eria), euka(ryota), viru(ses)
                    all_kingdoms = ["arch", "bact", "euka", "viru"]
                    pk = [i in pred.lower() for i in all_kingdoms]
                    pred_kingdom = all_kingdoms[pk.index(True)] if sum(pk) == 1 else "none"
                    if label.lower()[:4] == pred_kingdom:
                        self.metric_values.append(1)
                    else:
                        self.metric_values.append(0)
                    self.metric_names.append(f"{name}_{label.lower()}")
                    self.kingdom_preds.append((label.lower()[:4], pred_kingdom))
                elif name == "localization":
                    # membr(ane), nucle(us), mitoc(hondrion)
                    all_locs = ["membr", "nucle", "mitoc"]
                    if label != "none":
                        pl = [i in pred.lower() for i in all_locs]
                        pred_loc = all_locs[pl.index(True)] if sum(pl) == 1 else "none"
                        if label[:5] == pred_loc:
                            self.metric_values.append(1)
                        else:
                            self.metric_values.append(0)
                        self.metric_names.append(f"{name}_{label}")
                        self.localization_preds.append((label[:5], pred_loc))
                elif name == "cofactor":
                    if label != "none":
                        cofactors = [i.strip().lower() for i in label.split(",")]
                        if any([i in pred.lower() for i in cofactors]):
                            self.metric_values.append(1)
                        else:
                            self.metric_values.append(0)
                        self.metric_names.append(name)
                else:
                    raise ValueError(
                        "currently only supports kingdom, cofactor and localization str metrics. "
                        f"value {label} for metric {name} was given"
                    )
            else:
                raise ValueError(f"labels must be str or int, {label} was given.")

    def compute(self) -> Dict[str, float]:
        """Compute average metrics and aggregates."""
        metric_counts: Dict[str, int] = defaultdict(int)
        metric_sum: Dict[str, float] = defaultdict(float)
        for n, v in zip(self.metric_names, self.metric_values):
            metric_counts[n] += 1
            metric_sum[n] += v
        metric_out = {}
        for n in metric_counts:
            metric_out[n] = metric_sum[n] / metric_counts[n]

        if len(self.kingdom_preds) > 0:
            tax_true, tax_pred = zip(*self.kingdom_preds)
            metric_out["aggregate_f1_taxonomy"] = f1_score(tax_true, tax_pred, average="weighted")
        if len(self.localization_preds) > 0:
            loc_true, loc_pred = zip(*self.localization_preds)
            metric_out["aggregate_f1_localization"] = f1_score(loc_true, loc_pred, average="weighted")

        is_in, is_in_0 = [], []
        xis_in, xis_in_0 = [], []
        for n, v in metric_out.items():
            if n.startswith("in_"):
                if n.endswith("_0"):
                    is_in_0.append(v)
                else:
                    is_in.append(v)
            elif n.startswith("xin_"):
                if n.endswith("_0"):
                    xis_in_0.append(v)
                else:
                    xis_in.append(v)
        if is_in:
            metric_out["aggregate_semantic_location"] = sum(is_in) / len(is_in)
        if is_in_0:
            metric_out["aggregate_semantic_location_zero_shot_prompt"] = sum(is_in_0) / len(is_in_0)
        if xis_in:
            metric_out["aggregate_yes/no_location"] = sum(xis_in) / len(xis_in)
        if xis_in_0:
            metric_out["aggregate_yes/no_location_zero_shot_prompt"] = sum(xis_in_0) / len(xis_in_0)
        return metric_out
