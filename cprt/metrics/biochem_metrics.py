import string
from collections import defaultdict
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.metrics import f1_score
from torchmetrics import Metric


class BiochemMetrics(Metric):
    """Class to compute biochemical metrics from textual output."""

    numeric_preds: List[Tuple[str, float]]
    taxonomy_preds: List[Tuple[str, str]]
    localization_preds: List[Tuple[str, str]]

    def __init__(self, **kwargs: Any) -> None:
        super(BiochemMetrics, self).__init__(**kwargs)
        self.add_state("numeric_preds", [])
        self.add_state("taxonomy_preds", [])
        self.add_state("localization_preds", [])

    def update(self, predictions: List[str], labels: List[str | int | bool], metric_names: List[str]) -> None:
        """Update metric name and value states."""
        for pred, label, name in zip(predictions, labels, metric_names):
            # remove punctuations from pred and make lower case
            pred = pred.translate(str.maketrans("", "", string.punctuation)).lower()
            if isinstance(label, int):
                assert label > 0, "only positive int labels are supported"
                # get the int of predicted size only if one numeric present in the pred else set to 0
                pred_ints = [int(i) for i in pred.split() if i.isnumeric()]
                pred_size = pred_ints[0] if len(pred_ints) == 1 else 0
                epsilon = 1e-3
                self.numeric_preds.append((f"{name}_error", abs(np.log10(pred_size / label + epsilon))))
            elif isinstance(label, bool):
                has_yes = "yes" in pred.split()
                has_no = "no" in pred.split()
                is_pos = has_yes and not has_no
                is_neg = has_no and not has_yes
                void_answer = not (has_yes or has_no)
                if void_answer:
                    self.numeric_preds.append((f"void_answer_rate_{name}", 1))
                else:
                    self.numeric_preds.append((f"void_answer_rate_{name}", 0))
                    if (label is True and is_pos) or (label is False and is_neg):
                        self.numeric_preds.append((name, 1))
                    else:
                        self.numeric_preds.append((name, 0))
            elif isinstance(label, str):
                label = label.lower()
                if name == "kingdom":
                    if "virus" not in label:
                        # only checking for the presence of first 4 letters
                        # arch(aea), bact(eria), euka(ryota)
                        all_kingdoms = [i for i in ["archaea", "bacteria", "eukaryota"] if i[:4] in pred]
                        pred_kingdom = all_kingdoms[0] if len(all_kingdoms) == 1 else "none"
                        if label == pred_kingdom:
                            self.numeric_preds.append((f"{name}_{label}", 1))
                        else:
                            self.numeric_preds.append((f"{name}_{label}", 0))
                        self.taxonomy_preds.append((label, pred_kingdom))
                elif name == "localization":
                    if label != "none":
                        # only checking for the presence of first 5 letters
                        # membr(ane), nucle(us), mitoc(hondrion)
                        all_locs = [i for i in ["membrane", "nucleus", "mitochondrion"] if i[:5] in pred]
                        pred_loc = all_locs[0] if len(all_locs) == 1 else "none"
                        if label == pred_loc:
                            self.numeric_preds.append((f"{name}_{label}", 1))
                        else:
                            self.numeric_preds.append((f"{name}_{label}", 0))
                        self.localization_preds.append((label, pred_loc))
                elif name == "cofactor":
                    # TODO: use chemical entity prediction and molecule matching instead of string matching
                    if label != "none":
                        # assuming cofactors are labelled as a comma seperated string
                        cofactors = [i.strip().lower() for i in label.split(",")]
                        if any([i in pred for i in cofactors]):
                            self.numeric_preds.append((name, 1))
                        else:
                            self.numeric_preds.append((name, 0))
                else:
                    raise ValueError(
                        "currently only supports kingdom, cofactor and localization str metrics. "
                        f"value {label} for metric {name} was given"
                    )
            else:
                raise ValueError(f"labels must be int, bool or str. {label} was given.")

    def compute(self) -> Dict[str, float]:
        """Compute average metrics and aggregates."""
        metrics: Dict[str, List[float]] = defaultdict(list)
        for n, v in self.numeric_preds:
            metrics[n].append(v)
        metric_out = {}
        for n in metrics:
            metric_out[n] = cast(float, np.mean(metrics[n]))
        # Aggregate metrics
        if len(self.taxonomy_preds) > 0:
            tax_true, tax_pred = zip(*self.taxonomy_preds)
            metric_out["aggregate_f1_taxonomy"] = f1_score(tax_true, tax_pred, average="weighted")
        if len(self.localization_preds) > 0:
            loc_true, loc_pred = zip(*self.localization_preds)
            metric_out["aggregate_f1_localization"] = f1_score(loc_true, loc_pred, average="weighted")
        # aggregating localization "polar questions", all starting with is_in
        is_in = []
        for n, v in metric_out.items():
            if n.startswith("in_"):
                is_in.append(v)
        if is_in:
            metric_out["average_semantic_localization"] = cast(float, np.mean(is_in))
        return metric_out
