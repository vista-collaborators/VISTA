from typing import Any, Dict, List, Sequence


def binary_auc(labels: List[int], probs: List[float]) -> float:
    n_pos = sum(1 for y in labels if int(y) == 1)
    n_neg = sum(1 for y in labels if int(y) == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted((float(p), int(y)) for y, p in zip(labels, probs))
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_in_group = sum(1 for _, y in pairs[i:j] if y == 1)
        rank_sum_pos += avg_rank * pos_in_group
        i = j
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_matrix(labels: Sequence[int], preds: Sequence[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, pred in zip(labels, preds):
        matrix[int(label)][int(pred)] += 1
    return matrix


def compute_classification_metrics(
    labels: List[int],
    preds: List[int],
    probs: List[List[float]],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    num_classes = len(class_names)
    total = max(1, len(labels))
    cm = confusion_matrix(labels, preds, num_classes)
    correct = sum(cm[i][i] for i in range(num_classes))

    per_class = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0
    for idx, name in enumerate(class_names):
        tp = cm[idx][idx]
        fp = sum(cm[row][idx] for row in range(num_classes) if row != idx)
        fn = sum(cm[idx][col] for col in range(num_classes) if col != idx)
        support = sum(cm[idx])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_f1 += f1 * support
        per_class.append(
            {
                "class_id": int(idx),
                "class_name": str(name),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )

    metrics: Dict[str, Any] = {
        "accuracy": float(correct / total),
        "macro_precision": float(macro_precision / max(1, num_classes)),
        "macro_recall": float(macro_recall / max(1, num_classes)),
        "macro_f1": float(macro_f1 / max(1, num_classes)),
        "weighted_f1": float(weighted_f1 / total),
        "confusion_matrix": cm,
        "per_class": per_class,
    }
    if num_classes == 2 and probs:
        probs_pos = [float(row[1]) for row in probs]
        metrics.update(
            {
                "auc": binary_auc(labels, probs_pos),
                "precision": per_class[1]["precision"],
                "recall": per_class[1]["recall"],
                "f1": per_class[1]["f1"],
                "specificity": per_class[0]["recall"],
            }
        )
    return metrics
