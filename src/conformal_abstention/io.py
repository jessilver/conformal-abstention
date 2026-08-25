from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "instance_id",
    "h_norm",
    "error",
    "split",
}


def load_results_csv(path: str | Path) -> pd.DataFrame:
    """
    Carrega o arquivo produzido pelo agrupador.

    Colunas obrigatórias:
        instance_id
        h_norm
        error
        split
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("O arquivo não possui registros.")

    if df["split"].isna().any():
        raise ValueError("A coluna 'split' possui valores ausentes.")

    if not df["split"].isin(["calibration", "test"]).all():
        raise ValueError(
            "A coluna 'split' deve conter apenas "
            "'calibration' ou 'test'."
        )

    if not df["error"].isin([0, 1]).all():
        raise ValueError(
            "A coluna 'error' deve conter apenas 0 ou 1."
        )

    if df["h_norm"].isna().any():
        raise ValueError("A coluna 'h_norm' possui valores ausentes.")

    if not df["h_norm"].map(float).apply(lambda x: abs(x) != float("inf")).all():
        raise ValueError("A coluna 'h_norm' possui valores infinitos.")

    return df