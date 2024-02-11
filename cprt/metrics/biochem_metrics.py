import string
from collections import defaultdict
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from torchmetrics import Metric


class BiochemMetrics(Metric):
    """Class to compute biochemical metrics from textual output."""

    numeric_preds: List[Tuple[str, float]]
    class_preds: List[Tuple[str, str, str]]

    def __init__(self, **kwargs: Any) -> None:
        super(BiochemMetrics, self).__init__(**kwargs)
        self.add_state("numeric_preds", [])
        self.add_state("class_preds", [])

    def update(self, predictions: List[str], labels: List[str | int | bool], metric_names: List[str]) -> None:
        """Update metric name and value states."""
        for pred, label, name in zip(predictions, labels, metric_names):
            pred = pred.lower()
            if isinstance(label, bool):
                # remove all punctuations from pred
                pred = pred.translate(str.maketrans("", "", string.punctuation))
                has_yes = "yes" in pred.split()
                has_no = "no" in pred.split()
                # ensure exactly one class is predicted
                if has_no + has_yes == 1:
                    pred_class = "yes" if has_yes else "no"
                else:
                    pred_class = "none"
                label_class = "yes" if label == True else "no"
                self.class_preds.append((name, label_class, pred_class))
            elif isinstance(label, int):
                # remove common punctuations without affecting float values
                pred = pred.replace(",", "").replace(";", "").replace(". ", " ").replace("!", "")
                assert label > 0, f"only positive int labels are supported. {name}: {label} -> {pred}"
                # get the int of predicted size only if one numeric present in the pred else set to 0
                pred_ints = [int(i) for i in pred.split() if i.isnumeric()]
                pred_size = pred_ints[0] if len(pred_ints) == 1 and pred_ints[0] > 0 else 1
                self.numeric_preds.append((f"{name}_error", abs(np.log10(pred_size / label))))
            elif isinstance(label, str):
                label = label.lower()
                if name == "localization":
                    if label != "none":
                        # only checking for the presence of first 5 letters
                        # membr(ane), nucle(us), mitoc(hondrion)
                        all_locs = [i for i in ["membrane", "nucleus", "mitochondrion"] if i[:5] in pred]
                        pred_loc = all_locs[0] if len(all_locs) == 1 else "none"
                        self.class_preds.append((name, label, pred_loc))
                elif name == "cofactor":
                    # TODO: use chemical entity prediction and molecule matching instead of string matching
                    if label != "none":
                        # assuming cofactors are labelled as a comma seperated string
                        # Mentioning even one correct cofactor would be assumed correct answer
                        cofactors = [i.strip() for i in label.split(",")]
                        if any([i in pred for i in cofactors]):
                            self.numeric_preds.append((name, 1))
                        else:
                            self.numeric_preds.append((name, 0))
                else:
                    raise ValueError(
                        "currently only supports cofactor and localization str metrics. "
                        f"value {label} for metric {name} was given"
                    )
            else:
                raise ValueError(f"labels must be int, bool or str. {label} was given.")

    def compute(self) -> Dict[str, float]:
        """Compute average metrics and aggregates."""
        n_metrics: Dict[str, List[float]] = defaultdict(list)
        for n, v in self.numeric_preds:
            n_metrics[n].append(v)

        c_metrics: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for n, c, p in self.class_preds:
            c_metrics[n].append((c, p))

        metric_out = {}
        for n, v in n_metrics.items():
            metric_out[n] = cast(float, np.mean(v))
        for n, v in c_metrics.items():
            actual, preds = zip(*v)
            all_labels = list(set(actual))
            metric_out[f"{n}_f1"] = f1_score(actual, preds, average="macro", labels=all_labels)
            metric_out[f"{n}_balanced_accuracy"] = balanced_accuracy_score(actual, preds)
            # per class accuracies for non-yes/no questions: Taxonomy and Localization
            if "yes" not in all_labels:
                class_recall = recall_score(actual, preds, average=None, labels=all_labels)
                for c, r in zip(all_labels, class_recall):
                    metric_out[f"{n}_{c}_accuracy"] = r

        # aggregating localization "polar questions", all starting with is_in
        is_in = []
        for n, v in metric_out.items():
            if n.startswith("in_"):
                is_in.append(v)
        metric_out["average_semantic_localization"] = cast(float, np.mean(is_in))
        return metric_out
