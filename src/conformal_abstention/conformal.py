from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    alpha: float
    target_coverage: float
    n_calibration: int
    quantile_rank: int
    threshold: float


@dataclass(frozen=True)
class AbstentionMetrics:
    alpha: float
    threshold: float
    n_total: int
    n_accepted: int
    n_abstained: int
    coverage: float
    selective_accuracy: float
    selective_risk: float
    accepted_errors: int


class ConformalAbstention:
    """
    Calibrador split-conformal para uma política aceitar/abster.

    O escore deve ser uma medida de não conformidade:
    valores maiores significam maior incerteza ou maior risco.
    """

    def __init__(self, alpha: float = 0.10):
        self._validate_alpha(alpha)

        self.alpha = float(alpha)
        self.threshold_: float | None = None
        self.calibration_: CalibrationResult | None = None

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha deve estar estritamente entre 0 e 1.")

    @staticmethod
    def _as_scores(values: Iterable[float]) -> np.ndarray:
        scores = np.asarray(list(values), dtype=float)

        if scores.ndim != 1:
            raise ValueError("Os escores devem ser unidimensionais.")

        if scores.size == 0:
            raise ValueError("A lista de escores não pode estar vazia.")

        if not np.all(np.isfinite(scores)):
            raise ValueError("Os escores devem ser finitos.")

        return scores

    @staticmethod
    def _as_errors(values: Iterable[int]) -> np.ndarray:
        errors = np.asarray(list(values), dtype=int)

        if errors.ndim != 1:
            raise ValueError("Os erros devem ser unidimensionais.")

        if errors.size == 0:
            raise ValueError("A lista de erros não pode estar vazia.")

        if not np.all(np.isin(errors, [0, 1])):
            raise ValueError("Os erros devem conter somente 0 ou 1.")

        return errors

    @staticmethod
    def _conformal_rank(n: int, alpha: float) -> int:
        """
        Posição 1-indexada do quantil conformado:

            ceil((n + 1) * (1 - alpha))

        A posição é limitada ao intervalo [1, n].
        """
        rank = ceil((n + 1) * (1.0 - alpha))
        return min(max(rank, 1), n)

    def fit(self, calibration_scores: Iterable[float]) -> CalibrationResult:
        """
        Calcula o limiar usando exclusivamente o conjunto de calibração.
        """
        scores = self._as_scores(calibration_scores)
        ordered_scores = np.sort(scores)

        rank = self._conformal_rank(
            n=len(ordered_scores),
            alpha=self.alpha,
        )

        threshold = float(ordered_scores[rank - 1])

        self.threshold_ = threshold
        self.calibration_ = CalibrationResult(
            alpha=self.alpha,
            target_coverage=1.0 - self.alpha,
            n_calibration=len(ordered_scores),
            quantile_rank=rank,
            threshold=threshold,
        )

        return self.calibration_

    def _check_fitted(self) -> None:
        if self.threshold_ is None:
            raise RuntimeError(
                "O calibrador não foi ajustado. "
                "Execute fit(calibration_scores) primeiro."
            )

    def predict_accept(
        self,
        scores: Iterable[float],
    ) -> np.ndarray:
        """
        Retorna True para aceitar e False para abster.
        """
        self._check_fitted()
        scores_array = self._as_scores(scores)

        return scores_array <= self.threshold_

    def predict_decision(
        self,
        scores: Iterable[float],
    ) -> np.ndarray:
        """
        Retorna 'aceitar' ou 'abster' para cada instância.
        """
        accepted = self.predict_accept(scores)

        return np.where(
            accepted,
            "aceitar",
            "abster",
        )

    def evaluate(
        self,
        test_scores: Iterable[float],
        test_errors: Iterable[int],
    ) -> AbstentionMetrics:
        """
        Avalia cobertura, risco e acurácia seletiva.

        test_errors:
            0 = resposta correta;
            1 = resposta incorreta.
        """
        self._check_fitted()

        scores = self._as_scores(test_scores)
        errors = self._as_errors(test_errors)

        if len(scores) != len(errors):
            raise ValueError(
                "test_scores e test_errors devem ter o mesmo tamanho."
            )

        accepted = scores <= self.threshold_

        n_total = len(scores)
        n_accepted = int(accepted.sum())
        n_abstained = n_total - n_accepted
        accepted_errors = int(errors[accepted].sum())

        coverage = n_accepted / n_total

        if n_accepted > 0:
            selective_risk = accepted_errors / n_accepted
            selective_accuracy = 1.0 - selective_risk
        else:
            selective_risk = float("nan")
            selective_accuracy = float("nan")

        return AbstentionMetrics(
            alpha=self.alpha,
            threshold=float(self.threshold_),
            n_total=n_total,
            n_accepted=n_accepted,
            n_abstained=n_abstained,
            coverage=coverage,
            selective_accuracy=selective_accuracy,
            selective_risk=selective_risk,
            accepted_errors=accepted_errors,
        )