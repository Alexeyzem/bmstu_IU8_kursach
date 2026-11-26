"""Calibration of probabilities on held-out validation files."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def logits(scores):
    p = np.clip(scores, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrated_probabilities(validation_scores, validation_labels, scores):
    calibrator = LogisticRegression().fit(logits(validation_scores), validation_labels)
    return calibrator.predict_proba(logits(scores))[:, 1]


def calibration_metrics(validation_scores, validation_labels, scores, labels):
    calibrated = calibrated_probabilities(validation_scores, validation_labels, scores)
    return dict(brier=brier_score_loss(labels, scores),
                calibrated_brier=brier_score_loss(labels, calibrated),
                log_loss=log_loss(labels, scores),
                calibrated_log_loss=log_loss(labels, calibrated))
