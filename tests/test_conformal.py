import numpy as np
import pytest

from conformal_abstention import (
    ConformalAbstention,
    risk_coverage_curve,
)


def test_calibration_threshold():
    calibrator = ConformalAbstention(alpha=0.25)

    result = calibrator.fit(
        calibration_scores=[0.10, 0.20, 0.30, 0.40]
    )

    assert result.n_calibration == 4
    assert result.quantile_rank == 4
    assert result.threshold == pytest.approx(0.40)


def test_accept_and_abstain():
    calibrator = ConformalAbstention(alpha=0.10)
    calibrator.fit([0.10, 0.20, 0.30])

    decisions = calibrator.predict_decision(
        [0.05, 0.30, 0.80]
    )

    assert list(decisions) == [
        "aceitar",
        "aceitar",
        "abster",
    ]


def test_evaluation():
    calibrator = ConformalAbstention(alpha=0.10)
    calibrator.fit([0.10, 0.20, 0.30])

    metrics = calibrator.evaluate(
        test_scores=[0.05, 0.25, 0.80],
        test_errors=[0, 1, 1],
    )

    assert metrics.n_total == 3
    assert metrics.n_accepted == 2
    assert metrics.n_abstained == 1
    assert metrics.coverage == pytest.approx(2 / 3)
    assert metrics.selective_risk == pytest.approx(0.5)
    assert metrics.selective_accuracy == pytest.approx(0.5)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        ConformalAbstention(alpha=0.0)

    with pytest.raises(ValueError):
        ConformalAbstention(alpha=1.0)


def test_nan_score():
    calibrator = ConformalAbstention(alpha=0.10)

    with pytest.raises(ValueError):
        calibrator.fit([0.10, np.nan, 0.30])


def test_risk_coverage_curve():
    curve = risk_coverage_curve(
        uncertainty_scores=[0.8, 0.1, 0.4],
        errors=[1, 0, 1],
    )

    assert len(curve) == 3
    assert curve.iloc[0]["coverage"] == pytest.approx(1 / 3)
    assert curve.iloc[0]["risk"] == pytest.approx(0.0)