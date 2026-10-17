"""Experimentos com hiperparâmetros (Seção 8.5 do notebook).

Varia o tamanho do espaço latente e o peso β do KL, um de cada vez, e mede a AUC na
validação (o teste não é usado para escolher nada). Cada configuração é treinada uma vez
(semente 0), então diferenças pequenas (< 0,02) estão dentro da variação entre sementes.

Usa o σ salvo em models/metadata.json, então rode `make train` antes.
"""

import json

from loguru import logger
import pandas as pd
from sklearn.metrics import roc_auc_score

from module_vae.config import BETA, LATENT_DIM, METADATA_PATH, REPORTS_DIR
from module_vae.dataset import load_data
from module_vae.features import anomaly_score
from module_vae.modeling.predict import reconstruct
from module_vae.modeling.split import split_data
from module_vae.modeling.train import train_autoencoder
from module_vae.plots import apply_style, plot_ablation

LATENT_DIMS = [2, 4, 8, 16, 32]
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0]

HISTORY_PATH = REPORTS_DIR / "ablation.csv"


def main() -> None:
    """Treina cada configuração e registra a AUC de validação."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} não existe. Rode `make train` antes.")
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    sigma, epochs = metadata["sigma"], metadata["epochs"]
    logger.info(f"σ usado no score: {sigma} | épocas: {epochs}")

    splits = split_data(load_data())

    # A configuração padrão aparece nas duas varreduras; ela é treinada uma vez só.
    configs = [(d, BETA) for d in LATENT_DIMS] + [(LATENT_DIM, b) for b in BETAS if b != BETA]

    rows = []
    for latent_dim, beta in configs:
        logger.info(f"Treinando VAE com latente {latent_dim} e β {beta}...")
        model, _ = train_autoencoder(
            splits.x_train_normal,
            splits.x_val_normal,
            seed=0,
            latent_dim=latent_dim,
            beta=beta,
            epochs=epochs,
            verbose=False,
        )
        score_val = anomaly_score(splits.x_val, reconstruct(model, splits.x_val), sigma)
        rows.append(
            {
                "latent_dim": latent_dim,
                "beta": beta,
                "auc_val": roc_auc_score(splits.y_val, score_val),
            }
        )
        logger.info(f"AUC (validação): {rows[-1]['auc_val']:.3f}")

    history = pd.DataFrame(rows)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)
    logger.success(f"Resultados salvos em {HISTORY_PATH}")

    apply_style()
    plot_ablation(
        latent=history[history["beta"] == BETA].set_index("latent_dim")["auc_val"],
        beta=history[history["latent_dim"] == LATENT_DIM]
        .set_index("beta")["auc_val"]
        .sort_index(),
    )


if __name__ == "__main__":
    main()
