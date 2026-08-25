from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .conformal import ConformalAbstention


def evaluate_alpha_grid(
    calibration_scores: Sequence[float],
    test_scores: Sequence[float],
    test_errors: Sequence[int],
    alphas: Iterable[float] = (0.05, 0.10, 0.20, 0.30, 0.40),
) -> pd.DataFrame:
    """
    Avalia a política para vários níveis de alpha.
    """
    rows = []

    for alpha in alphas:
        calibrator = ConformalAbstention(alpha=alpha)
        calibration = calibrator.fit(calibration_scores)

        metrics = calibrator.evaluate(
            test_scores=test_scores,
            test_errors=test_errors,
        )

        rows.append(
            {
                "alpha": alpha,
                "target_coverage": calibration.target_coverage,
                "threshold": calibration.threshold,
                "n_calibration": calibration.n_calibration,
                "quantile_rank": calibration.quantile_rank,
                "coverage": metrics.coverage,
                "selective_accuracy": metrics.selective_accuracy,
                "selective_risk": metrics.selective_risk,
                "n_accepted": metrics.n_accepted,
                "n_abstained": metrics.n_abstained,
                "accepted_errors": metrics.accepted_errors,
            }
        )

    return pd.DataFrame(rows)


def risk_coverage_curve(
    uncertainty_scores: Sequence[float],
    errors: Sequence[int],
) -> pd.DataFrame:
    """
    Calcula risco e acurácia para coberturas crescentes.

    As instâncias com menor incerteza são aceitas primeiro.
    """
    scores = np.asarray(uncertainty_scores, dtype=float)
    errors_array = np.asarray(errors, dtype=int)

    if scores.ndim != 1 or errors_array.ndim != 1:
        raise ValueError("Os vetores devem ser unidimensionais.")

    if len(scores) == 0:
        raise ValueError("Os vetores não podem estar vazios.")

    if len(scores) != len(errors_array):
        raise ValueError(
            "uncertainty_scores e errors devem ter o mesmo tamanho."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Os escores devem ser finitos.")

    if not np.all(np.isin(errors_array, [0, 1])):
        raise ValueError("errors deve conter somente 0 ou 1.")

    order = np.argsort(scores, kind="stable")
    ordered_errors = errors_array[order]

    accepted = np.arange(1, len(ordered_errors) + 1)
    cumulative_errors = np.cumsum(ordered_errors)

    coverage = accepted / len(ordered_errors)
    risk = cumulative_errors / accepted
    selective_accuracy = 1.0 - risk

    return pd.DataFrame(
        {
            "n_accepted": accepted,
            "coverage": coverage,
            "risk": risk,
            "selective_accuracy": selective_accuracy,
        }
    )


def pairwise_cluster_metrics(
    reference_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> dict[str, float]:
    """
    Métricas opcionais para comparar clusters.

    Esta função mantém o módulo de abstenção independente,
    mas permite integrar métricas do seu agrupador.
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        completeness_score,
        f1_score,
        homogeneity_score,
        v_measure_score,
    )

    reference = np.asarray(reference_labels)
    predicted = np.asarray(predicted_labels)

    if len(reference) != len(predicted):
        raise ValueError(
            "reference_labels e predicted_labels devem ter o mesmo tamanho."
        )

    result = {
        "ari": float(adjusted_rand_score(reference, predicted)),
        "ami": float(adjusted_mutual_info_score(reference, predicted)),
        "homogeneity": float(
            homogeneity_score(reference, predicted)
        ),
        "completeness": float(
            completeness_score(reference, predicted)
        ),
        "v_measure": float(
            v_measure_score(reference, predicted)
        ),
    }

    return result


def binary_pairwise_f1(
    reference_same_cluster: Sequence[int],
    predicted_same_cluster: Sequence[int],
) -> float:
    """
    F1 para equivalência entre pares de respostas.
    """
    from sklearn.metrics import f1_score

    reference = np.asarray(reference_same_cluster, dtype=int)
    predicted = np.asarray(predicted_same_cluster, dtype=int)

    if len(reference) != len(predicted):
        raise ValueError(
            "Os vetores de pares devem ter o mesmo tamanho."
        )

    return float(
        f1_score(
            reference,
            predicted,
            zero_division=0,
        )
    )