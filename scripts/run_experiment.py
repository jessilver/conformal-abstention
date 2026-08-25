from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.conformal_abstention import (
    ConformalAbstention,
    evaluate_alpha_grid,
    load_results_csv,
    risk_coverage_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibração conformada e abstenção usando H_norm."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV com os resultados do agrupador.",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Diretório para salvar os resultados.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Nível de erro usado na calibração.",
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.30, 0.40],
        help="Valores usados na análise de sensibilidade.",
    )

    return parser.parse_args()


def metrics_to_dataframe(metrics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alpha": metrics.alpha,
                "threshold": metrics.threshold,
                "n_total": metrics.n_total,
                "n_accepted": metrics.n_accepted,
                "n_abstained": metrics.n_abstained,
                "coverage": metrics.coverage,
                "selective_accuracy": metrics.selective_accuracy,
                "selective_risk": metrics.selective_risk,
                "accepted_errors": metrics.accepted_errors,
            }
        ]
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results_csv(args.input)

    calibration_df = df[df["split"] == "calibration"].copy()
    test_df = df[df["split"] == "test"].copy()

    if calibration_df.empty:
        raise ValueError(
            "O conjunto de calibração está vazio."
        )

    if test_df.empty:
        raise ValueError(
            "O conjunto de teste está vazio."
        )

    calibration_scores = calibration_df["h_norm"].to_numpy()
    test_scores = test_df["h_norm"].to_numpy()
    test_errors = test_df["error"].to_numpy()

    calibrator = ConformalAbstention(alpha=args.alpha)
    calibration = calibrator.fit(calibration_scores)

    test_df["decision"] = calibrator.predict_decision(test_scores)
    test_df["accepted"] = test_df["decision"].eq("aceitar")

    metrics = calibrator.evaluate(
        test_scores=test_scores,
        test_errors=test_errors,
    )

    alpha_grid = evaluate_alpha_grid(
        calibration_scores=calibration_scores,
        test_scores=test_scores,
        test_errors=test_errors,
        alphas=args.alphas,
    )

    curve = risk_coverage_curve(
        uncertainty_scores=test_scores,
        errors=test_errors,
    )

    calibration_df.to_csv(
        output_dir / "calibration_data.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test_decisions.csv",
        index=False,
    )

    metrics_to_dataframe(metrics).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    alpha_grid.to_csv(
        output_dir / "alpha_grid.csv",
        index=False,
    )

    curve.to_csv(
        output_dir / "risk_coverage_curve.csv",
        index=False,
    )

    print("Execução concluída.")
    print(f"Limiar: {calibration.threshold:.6f}")
    print(f"Cobertura: {metrics.coverage:.4f}")
    print(
        "Acurácia seletiva: "
        f"{metrics.selective_accuracy:.4f}"
    )
    print(
        "Risco seletivo: "
        f"{metrics.selective_risk:.4f}"
    )
    print(f"Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()