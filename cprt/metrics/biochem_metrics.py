from collections import defaultdict
from typing import Any, Dict, List

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from torchmetrics import Metric


class BiochemMetrics(Metric):
    """Class to compute biochemical metrics from textual output."""

    metric_names: List[str]
    metric_values: List[float]

    def __init__(self, **kwargs: Any) -> None:
        super(BiochemMetrics, self).__init__(**kwargs)
        nltk.download("vader_lexicon")
        self.sia = SentimentIntensityAnalyzer()
        self.add_state("metric_names", [])
        self.add_state("metric_values", [])

    def update(self, predictions: List[str], labels: List[str | int], metric_names: List[str]) -> None:
        """Update metric name and value states."""
        for pred, label, name in zip(predictions, labels, metric_names):
            if isinstance(label, int):
                sentiment_score = (1 + self.sia.polarity_scores(pred)["compound"]) / 2
                self.metric_values.append(1 - abs(label - sentiment_score))
                self.metric_names.append(name)
            elif isinstance(label, str):
                if name == "kingdom":
                    # arch(aea), bact(eria), euka(ryota), viru(ses)
                    all_kingdoms = ["arch", "bact", "euka", "viru"]
                    if label.lower()[:4] in pred.lower():
                        # divide the score by the number of predicted kingdoms
                        self.metric_values.append(1 / sum([i in pred.lower() for i in all_kingdoms]))
                    else:
                        self.metric_values.append(0)
                    self.metric_names.append(f"{name}_{label}")
                elif name == "localization":
                    # membr(ane), nucle(us), mitoc(hondrion)
                    all_locs = ["membr", "nucle", "mitoc"]
                    if label != "none":
                        if label[:5] in pred.lower() and sum([i in pred.lower() for i in all_locs]) == 1:
                            self.metric_values.append(1)
                        else:
                            self.metric_values.append(0)
                        self.metric_names.append(f"{name}_{label}")
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
        # dirty aggregation
        # TODO: clean this after deciding about metrics
        tax, loc, is_in, bind = [], [], [], []
        for n, v in metric_out.items():
            if "kingdom" in n:
                tax.append(v)
            elif "localization" in n:
                loc.append(v)
            elif "in_" in n:
                is_in.append(v)
            elif "_binding" in n:
                bind.append(v)
        if tax:
            metric_out["taxonomy"] = sum(tax) / len(tax)
        if loc:
            metric_out["localization"] = sum(loc) / len(loc)
        if is_in:
            metric_out["semantic_loc"] = sum(is_in) / len(is_in)
        if bind:
            metric_out["binding"] = sum(bind) / len(bind)
        return metric_out
