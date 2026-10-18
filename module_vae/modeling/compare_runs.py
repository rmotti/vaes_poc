"""Aplica a regra das ablações às execuções registradas no MLflow.

    uv run python -m module_vae.modeling.compare_runs --candidates abl-A-recorte,abl-B-normalizacao

Cada ablação (um nome de execução, com uma execução por semente) é comparada com a v2
(ABLATION_REFERENCE) pela média das sementes, usando só métricas de validação. A regra
está em config.py e foi fixada antes de qualquer ablação rodar.
"""

from typing import Annotated

from loguru import logger
import mlflow
import pandas as pd
import typer

from module_vae.config import (
    ABLATION_MIN_AUC_GAIN,
    ABLATION_MIN_POINTING_GAIN,
    ABLATION_REFERENCE,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
)

AUC = "metrics.val_auc"
POINTING = "metrics.val_localization_pointing_game"
CHANCE = "metrics.val_localization_box_area_fraction"
CHANCE_IN_MASK = "metrics.val_localization_box_fraction_in_mask"

app = typer.Typer()


def apply_rule(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Compara uma ablação com a referência pela média das sementes.

    Args:
        reference (pd.DataFrame): Execuções da v2, uma por semente (colunas AUC e POINTING).
        candidate (pd.DataFrame): Execuções da ablação, uma por semente.

    Returns:
        dict: Médias, diferenças, tolerâncias (um desvio-padrão da referência) e a decisão.
    """
    delta_auc = candidate[AUC].mean() - reference[AUC].mean()
    delta_pointing = candidate[POINTING].mean() - reference[POINTING].mean()
    tolerance_auc, tolerance_pointing = reference[AUC].std(), reference[POINTING].std()

    gain = delta_auc >= ABLATION_MIN_AUC_GAIN or delta_pointing >= ABLATION_MIN_POINTING_GAIN
    no_harm = delta_auc >= -tolerance_auc and delta_pointing >= -tolerance_pointing

    return {
        "n_seeds": len(candidate),
        "val_auc": candidate[AUC].mean(),
        "val_auc_std": candidate[AUC].std(),
        "delta_auc": delta_auc,
        "val_pointing": candidate[POINTING].mean(),
        "val_pointing_std": candidate[POINTING].std(),
        "delta_pointing": delta_pointing,
        "tolerance_auc": tolerance_auc,
        "tolerance_pointing": tolerance_pointing,
        "accepted": bool(gain and no_harm),
    }


def chance_level(runs: pd.DataFrame) -> float:
    """Localização de um mapa sem informação: dentro da máscara, se houver uma."""
    if CHANCE_IN_MASK in runs and runs[CHANCE_IN_MASK].notna().all():
        return runs[CHANCE_IN_MASK].mean()
    return runs[CHANCE].mean()


@app.command()
def main(
    candidates: Annotated[str, typer.Option(help="Nomes das ablações, separados por vírgula.")],
    experiment: Annotated[str, typer.Option(help="Experimento no MLflow.")] = "rsna_pa",
    output: Annotated[str, typer.Option(help="Pasta em reports/ para a tabela.")] = (
        "rsna_pa_ablations_v2"
    ),
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'"
    )
    runs = (
        runs[runs["tags.mlflow.parentRunId"].isna()] if "tags.mlflow.parentRunId" in runs else runs
    )
    by_name = runs.groupby("tags.mlflow.runName")

    reference = pd.concat([by_name.get_group(name) for name in ABLATION_REFERENCE])
    rows = {"v2 (referência)": {**apply_rule(reference, reference), "accepted": None}}
    rows["v2 (referência)"]["chance"] = chance_level(reference)
    for name in candidates.split(","):
        candidate = by_name.get_group(name.strip())
        rows[name] = {**apply_rule(reference, candidate), "chance": chance_level(candidate)}

    table = pd.DataFrame(rows).T
    # Quanto a localização fica acima do acaso de cada um (a máscara muda o acaso)
    table["pointing_above_chance"] = table["val_pointing"] - table["chance"]

    path = REPORTS_DIR / output / "ablations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index_label="run")
    logger.success(f"Regra das ablações (validação, média das sementes):\n{table.to_string()}")
    logger.info(f"Tabela salva em {path}")


if __name__ == "__main__":
    app()
