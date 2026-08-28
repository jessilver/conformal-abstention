from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conformal_abstention import (
    ConformalAbstention,
    evaluate_alpha_grid,
    load_results_csv,
    risk_coverage_curve,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "example_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
DEFAULT_ALPHA = 0.10
DEFAULT_ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibração conformada para política de abstenção "
            "usando H_norm como escore de incerteza."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Arquivo CSV de entrada contendo os resultados "
            "do agrupamento semântico."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório em que os resultados serão salvos.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Nível de erro usado para calibrar o limiar.",
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help=(
            "Valores de alpha utilizados na análise "
            "de sensibilidade."
        ),
    )

    return parser.parse_args()


def metrics_to_dataframe(metrics) -> "pd.DataFrame":
    import pandas as pd

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


def validate_paths(input_path: Path, output_dir: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            "\nArquivo de entrada não encontrado.\n"
            f"Esperado em: {input_path}\n\n"
            "Soluções possíveis:\n"
            "1. Crie o arquivo data/example_results.csv; ou\n"
            "2. Informe o arquivo manualmente com:\n"
            "   python scripts/run_experiment.py --input caminho/do/arquivo.csv"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            "O arquivo de entrada deve estar no formato CSV. "
            f"Recebido: {input_path.name}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    validate_paths(
        input_path=input_path,
        output_dir=output_dir,
    )

    print("=" * 60)
    print("EXPERIMENTO DE ABSTENÇÃO CONFORMADA")
    print("=" * 60)
    print(f"Arquivo de entrada: {input_path}")
    print(f"Diretório de saída: {output_dir}")
    print(f"Alpha principal: {args.alpha}")
    print(f"Alphas avaliados: {args.alphas}")
    print("-" * 60)

    df = load_results_csv(input_path)

    calibration_df = df[
        df["split"] == "calibration"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    if calibration_df.empty:
        raise ValueError(
            "Não existem instâncias com split='calibration'."
        )

    if test_df.empty:
        raise ValueError(
            "Não existem instâncias com split='test'."
        )

    calibration_scores = calibration_df["h_norm"].to_numpy()
    test_scores = test_df["h_norm"].to_numpy()
    test_errors = test_df["error"].to_numpy()

    calibrator = ConformalAbstention(alpha=args.alpha)

    calibration = calibrator.fit(
        calibration_scores=calibration_scores
    )

    test_df["decision"] = calibrator.predict_decision(
        test_scores
    )

    test_df["accepted"] = test_df["decision"].eq(
        "aceitar"
    )

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

    calibration_path = output_dir / "calibration_data.csv"
    decisions_path = output_dir / "test_decisions.csv"
    metrics_path = output_dir / "metrics.csv"
    alpha_grid_path = output_dir / "alpha_grid.csv"
    curve_path = output_dir / "risk_coverage_curve.csv"

    calibration_df.to_csv(
        calibration_path,
        index=False,
    )

    test_df.to_csv(
        decisions_path,
        index=False,
    )

    metrics_to_dataframe(metrics).to_csv(
        metrics_path,
        index=False,
    )

    alpha_grid.to_csv(
        alpha_grid_path,
        index=False,
    )

    curve.to_csv(
        curve_path,
        index=False,
    )

    print("Calibração concluída.")
    print(f"Número de exemplos de calibração: {calibration.n_calibration}")
    print(f"Posição do quantil: {calibration.quantile_rank}")
    print(f"Limiar calibrado: {calibration.threshold:.6f}")
    print("-" * 60)
    print("Avaliação no conjunto de teste")
    print(f"Instâncias totais: {metrics.n_total}")
    print(f"Respostas aceitas: {metrics.n_accepted}")
    print(f"Abstenções: {metrics.n_abstained}")
    print(f"Cobertura: {metrics.coverage:.4f}")
    print(f"Erros entre aceitas: {metrics.accepted_errors}")
    print(f"Risco seletivo: {metrics.selective_risk:.4f}")
    print(
        "Acurácia seletiva: "
        f"{metrics.selective_accuracy:.4f}"
    )
    print("-" * 60)
    print("Arquivos gerados:")
    print(f"- {calibration_path.name}")
    print(f"- {decisions_path.name}")
    print(f"- {metrics_path.name}")
    print(f"- {alpha_grid_path.name}")
    print(f"- {curve_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()