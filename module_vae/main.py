"""Pipeline completo: dados → treino → calibração → avaliação → comparação.

uv run python -m module_vae.main                     # igual ao notebook (5-10 min)
uv run python -m module_vae.main --epochs 2 --n-seeds 1 --no-baselines   # teste rápido
"""

import json
from typing import Annotated

from loguru import logger
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import typer

from module_vae.config import (
    DEVICE,
    EPOCHS,
    FIGURES_DIR,
    IMAGE_SIZE,
    KL_WARMUP_EPOCHS,
    LATENT_DIM,
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    SEEDS,
    THRESHOLD_PERCENTILE,
)
from module_vae.dataset import describe_data, load_data
from module_vae.features import anomaly_score
from module_vae.modeling.architecture import VAE, describe_architecture
from module_vae.modeling.baselines import compare_methods
from module_vae.modeling.calibration import calibrate_threshold, select_sigma, youden_threshold
from module_vae.modeling.evaluate import compare_thresholds, evaluate_model, simulate_queue
from module_vae.modeling.predict import predict, reconstruct
from module_vae.modeling.split import split_data
from module_vae.modeling.train import save_model, set_seed, train_autoencoder
from module_vae.plots import (
    apply_style,
    plot_gain_curve,
    plot_generated_samples,
    plot_heatmaps,
    plot_method_comparison,
    plot_sigma_selection,
    plot_test_evaluation,
    plot_training_curves,
)

app = typer.Typer()


@app.command()
def main(
    epochs: Annotated[int, typer.Option(help="Épocas de treino de cada autoencoder.")] = EPOCHS,
    n_seeds: Annotated[int, typer.Option(help="Sementes do VAE e do autoencoder comum.")] = len(
        SEEDS
    ),
    baselines: Annotated[
        bool, typer.Option(help="Compara com PCA, autoencoder comum e CNN.")
    ] = True,
):
    logger.info(f"Dispositivo: {DEVICE}")
    apply_style()
    seeds = SEEDS[:n_seeds]

    # =============================================
    # DADOS
    # =============================================

    data = load_data()
    logger.info(f"Exames por split:\n{describe_data(data)}")
    splits = split_data(data)

    # =============================================
    # TREINO DO VAE (somente exames normais)
    # =============================================

    describe_architecture(VAE(LATENT_DIM))

    logger.info(f"Treinando o VAE (semente {seeds[0]}, {epochs} épocas)...")
    vae, history = train_autoencoder(
        splits.x_train_normal, splits.x_val_normal, seed=seeds[0], epochs=epochs
    )
    plot_training_curves(history, KL_WARMUP_EPOCHS)

    # Amostras de z ~ N(0, I) passadas pelo decoder: o VAE também é um modelo gerativo.
    set_seed(123)
    with torch.no_grad():
        samples = vae.decode(torch.randn(16, LATENT_DIM).to(DEVICE)).cpu()
    plot_generated_samples(samples)

    # =============================================
    # CALIBRAÇÃO NA VALIDAÇÃO
    # Escolhe σ e o limiar sem tocar no teste
    # =============================================

    rec_val = reconstruct(vae, splits.x_val)
    sigma, auc_val_by_sigma = select_sigma(splits.x_val, rec_val, splits.y_val)
    score_val = anomaly_score(splits.x_val, rec_val, sigma)
    threshold = calibrate_threshold(score_val, splits.y_val)

    # =============================================
    # AVALIAÇÃO NO TESTE
    # =============================================

    predictions, rec_test, maps_test = predict(vae, splits.x_test, sigma, threshold)
    score_test = predictions["anomaly_score"].to_numpy()
    metrics = evaluate_model(splits.y_test, score_test, threshold)

    # A mesma curva de σ no teste (só para a figura) mostra se a escolha generaliza.
    # Com σ = 0, o score é o erro pixel a pixel ingênuo, que funciona mal (Seção 7.2).
    auc_test_by_sigma = {
        s: roc_auc_score(splits.y_test, anomaly_score(splits.x_test, rec_test, s))
        for s in auc_val_by_sigma
    }
    metrics["roc_auc_naive"] = auc_test_by_sigma[0]
    logger.info(f"AUC do erro pixel a pixel ingênuo (σ = 0): {metrics['roc_auc_naive']:.3f}")
    plot_sigma_selection(auc_val_by_sigma, auc_test_by_sigma, sigma)

    thresholds = {
        f"p{p} dos normais (sem rótulos)": float(np.percentile(score_val[splits.y_val == 0], p))
        for p in [95, 90, 80]
    }
    thresholds["Youden (com rótulos)"] = youden_threshold(score_val, splits.y_val)
    threshold_table = compare_thresholds(
        thresholds, splits.y_val, score_val, splits.y_test, score_test
    )

    queue = simulate_queue(splits.y_test, score_test)

    plot_test_evaluation(splits.y_test, score_test)
    plot_gain_curve(splits.y_test, score_test)

    # Escala de cor comum a todos os mapas de calor (inclusive os da inferência)
    vmax_map = float(np.percentile(maps_test.numpy(), 99.5))

    rng = np.random.default_rng(42)
    examples = [
        *rng.choice(np.where(splits.y_test == 0)[0], 4, replace=False),
        *rng.choice(np.where(splits.y_test == 1)[0], 4, replace=False),
    ]
    by_score = np.argsort(-score_test)
    y_test = splits.y_test
    false_pos = [i for i in by_score if y_test[i] == 0 and score_test[i] >= threshold][:4]
    false_neg = [i for i in by_score[::-1] if y_test[i] == 1 and score_test[i] < threshold][:4]

    for indices, title, name in [
        (examples, "4 exames normais e 4 com pneumonia", "heatmaps.png"),
        (false_pos, "Falsos positivos mais fortes", "false_positives.png"),
        (false_neg, "Falsos negativos mais fortes", "false_negatives.png"),
    ]:
        if indices:
            plot_heatmaps(
                splits.x_test,
                rec_test,
                maps_test,
                score_test,
                indices,
                vmax_map,
                title,
                FIGURES_DIR / name,
                labels=y_test,
            )

    # =============================================
    # SALVA MODELO E RELATÓRIOS
    # =============================================

    save_model(
        vae,
        metadata={
            "latent_dim": LATENT_DIM,
            "image_size": IMAGE_SIZE,
            "epochs": epochs,
            "seed": seeds[0],
            "sigma": sigma,
            "threshold": threshold,
            "threshold_rule": f"percentil {THRESHOLD_PERCENTILE} dos normais da validação",
            "vmax_map": vmax_map,
            "auc_val": auc_val_by_sigma[sigma],
        },
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.insert(0, "label", y_test)
    predictions.to_csv(REPORTS_DIR / "test_predictions.csv", index_label="test_index")
    threshold_table.to_csv(REPORTS_DIR / "thresholds.csv")

    # =============================================
    # COMPARAÇÃO COM ALTERNATIVAS
    # =============================================

    if baselines:
        comparison = compare_methods(splits, {seeds[0]: vae}, seeds, epochs=epochs)
        comparison.to_csv(REPORTS_DIR / "method_comparison.csv")
        plot_method_comparison(comparison)
        metrics["comparison"] = (
            comparison[["auc_test", "auc_test_std"]].round(4).to_dict(orient="index")
        )

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({**metrics, "queue_simulation": queue}, file, indent=4, ensure_ascii=False)

    logger.success(f"Pipeline executado com sucesso. Relatórios em {REPORTS_DIR}")


if __name__ == "__main__":
    app()
