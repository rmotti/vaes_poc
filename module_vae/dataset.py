"""Download e carregamento do PneumoniaMNIST (MedMNIST v2).

São 5.856 raios-X de tórax pediátricos rotulados como normal (0) ou pneumonia (1),
já recortados e redimensionados pelo MedMNIST. Os splits de treino, validação e teste
vêm prontos no arquivo .npz.
"""

from pathlib import Path

from loguru import logger
from medmnist import PneumoniaMNIST
import numpy as np
import pandas as pd
import torch

from module_vae.config import IMAGE_SIZE, RAW_DATA_DIR

SPLITS = ("train", "val", "test")


def load_split(
    split: str,
    image_size: int = IMAGE_SIZE,
    root: Path = RAW_DATA_DIR,
) -> tuple[torch.Tensor, np.ndarray]:
    """Carrega um split do PneumoniaMNIST, baixando o arquivo se necessário.

    Args:
        split (str): "train", "val" ou "test".
        image_size (int): Resolução das imagens (28, 64, 128 ou 224).
        root (Path): Pasta onde o .npz é salvo.

    Returns:
        tuple[torch.Tensor, np.ndarray]:
            - x: imagens (N, 1, H, W) em float32, com pixels em [0, 1];
            - y: rótulos (N,), 0 = normal e 1 = pneumonia.
    """
    root.mkdir(parents=True, exist_ok=True)
    ds = PneumoniaMNIST(split=split, download=True, size=image_size, root=str(root))

    # Os pixels vêm em uint8 (0 a 255). Dividir por 255 deixa cada pixel em [0, 1],
    # que é o que a BCE da loss e a sigmoide do decoder esperam.
    x = torch.tensor(ds.imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    y = ds.labels.ravel().astype(int)

    logger.info(f"Split {split} carregado: {tuple(x.shape)}")
    return x, y


def load_data(
    image_size: int = IMAGE_SIZE,
    root: Path = RAW_DATA_DIR,
) -> dict[str, tuple[torch.Tensor, np.ndarray]]:
    """Carrega os três splits do PneumoniaMNIST.

    Returns:
        dict[str, tuple[torch.Tensor, np.ndarray]]: {"train": (x, y), "val": (x, y),
            "test": (x, y)}.
    """
    logger.info("Carregando o PneumoniaMNIST...")
    return {split: load_split(split, image_size, root) for split in SPLITS}


def describe_data(data: dict[str, tuple[torch.Tensor, np.ndarray]]) -> pd.DataFrame:
    """Conta exames normais e com pneumonia em cada split.

    Args:
        data (dict): Saída de load_data.

    Returns:
        pd.DataFrame: Uma linha por split, com as contagens e o % de pneumonia.
    """
    counts = pd.DataFrame(
        {
            split: {
                "normal": int((y == 0).sum()),
                "pneumonia": int((y == 1).sum()),
                "total": len(y),
            }
            for split, (_, y) in data.items()
        }
    ).T
    counts["% pneumonia"] = (100 * counts["pneumonia"] / counts["total"]).round(1)

    return counts


if __name__ == "__main__":
    data = load_data()
    logger.success(f"PneumoniaMNIST salvo em {RAW_DATA_DIR}\n{describe_data(data)}")
