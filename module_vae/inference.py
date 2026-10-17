"""Inferência com o modelo salvo por `make train`.

    uv run python -m module_vae.inference                    # 5 exames sorteados do teste
    uv run python -m module_vae.inference --image exame.png  # um raio-X qualquer

A imagem externa é convertida para tons de cinza e redimensionada para 64×64 px. O
PneumoniaMNIST já vem recortado no tórax, então exames com outro enquadramento tendem a
receber score alto (Seção 9.1 do notebook).
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
from PIL import Image
import torch
import typer

from module_vae.config import FIGURES_DIR, METADATA_PATH, MODEL_PATH
from module_vae.dataset import load_split
from module_vae.modeling.predict import load_model, predict
from module_vae.plots import apply_style, plot_heatmaps

app = typer.Typer()


def load_image(path: Path, image_size: int) -> torch.Tensor:
    """Lê uma imagem e devolve um tensor (1, 1, H, W) com pixels em [0, 1]."""
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    return torch.tensor(np.asarray(img), dtype=torch.float32)[None, None] / 255.0


@app.command()
def main(
    image: Annotated[
        Path | None, typer.Option(help="Raio-X (png/jpg) a avaliar. Sem ele, usa o teste.")
    ] = None,
    n_samples: Annotated[
        int, typer.Option(help="Exames sorteados do teste quando não há --image.")
    ] = 5,
):
    model, metadata = load_model(MODEL_PATH, METADATA_PATH)

    if image is not None:
        x, labels = load_image(image, metadata["image_size"]), None
        title = image.name
    else:
        x_test, y_test = load_split("test", metadata["image_size"])
        idx = np.random.default_rng(42).choice(len(x_test), n_samples, replace=False)
        x, labels = x_test[idx], y_test[idx]
        title = f"{n_samples} exames sorteados do teste"

    predictions, x_rec, maps = predict(model, x, metadata["sigma"], metadata["threshold"])

    # queue_position ordena só os exames avaliados nesta chamada.
    predictions["decision"] = predictions["prediction"].map(
        {1: "revisar com prioridade", 0: "dentro do padrão"}
    )
    if labels is not None:
        predictions.insert(0, "label", np.where(labels == 1, "pneumonia", "normal"))
    logger.success(f"Predições:\n{predictions}")

    apply_style()
    plot_heatmaps(
        x,
        x_rec,
        maps,
        predictions["anomaly_score"].to_numpy(),
        list(range(len(x))),
        metadata["vmax_map"],
        title,
        FIGURES_DIR / "inference.png",
        labels=labels,
    )


if __name__ == "__main__":
    app()
