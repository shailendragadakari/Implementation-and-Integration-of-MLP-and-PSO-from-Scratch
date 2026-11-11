# Classification metrics (binary, positive class = 1).

import numpy as np

def probabilities_to_labels(probs, threshold=0.5):
    # Turn probabilities into 0/1 labels: at or above the threshold -> 1.
    return (np.asarray(probs).ravel() >= threshold).astype(int)

def confusion_counts(y_true, y_pred):
    # Count true/false positives/negatives (positive class = 1).
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    # tp: said 1, truly 1 | tn: said 0, truly 0
    # fp: said 1, truly 0 | fn: said 0, truly 1
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def classification_metrics(y_true, y_pred):
    # Accuracy, precision, recall, F1 and the four counts.
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    # precision: of everything predicted 1, how much was right? (0 if nothing predicted 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    # recall: of everything truly 1, how much did we find?
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # F1: the balance (harmonic mean) of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}