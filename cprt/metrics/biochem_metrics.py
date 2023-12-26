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
                else:
                    assert label in [0, 1]
                    sentiment_score = (1 + self.sia.polarity_scores(pred)["compound"]) / 2
                    self.metric_values.append(1 - abs(label - sentiment_score))
                self.metric_names.append(name)
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
            metric_out["agg_f1_taxonomy"] = f1_score(tax_true, tax_pred, average="weighted")
        if len(self.localization_preds) > 0:
            loc_true, loc_pred = zip(*self.localization_preds)
            metric_out["agg_f1_localization"] = f1_score(loc_true, loc_pred, average="weighted")

        is_in, bind = [], []
        for n, v in metric_out.items():
            if "in_" in n:
                is_in.append(v)
            elif "_binding" in n:
                bind.append(v)
        if is_in:
            metric_out["agg_semantic_loc"] = sum(is_in) / len(is_in)
        if bind:
            metric_out["agg_binding"] = sum(bind) / len(bind)
        return metric_out
