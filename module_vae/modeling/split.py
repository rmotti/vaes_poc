"""Montagem dos conjuntos: o ponto central do desenho experimental.

O VAE só vê exames normais no treino. As pneumonias do treino ficam de fora (só a CNN
supervisionada, usada como referência, as utiliza).

- Treino do VAE: os exames normais do split de treino.
- Validação: acompanhar o treino (normais), calibrar o limiar (normais) e escolher σ
  (normais + pneumonias).
- Teste: reservado para a avaliação final.
"""

from dataclasses import dataclass

from loguru import logger
import numpy as np
import torch


@dataclass
class DataSplits:
    x_train: torch.Tensor
    y_train: np.ndarray
    x_val: torch.Tensor
    y_val: np.ndarray
    x_test: torch.Tensor
    y_test: np.ndarray
    x_train_normal: torch.Tensor
    x_val_normal: torch.Tensor


def split_data(data: dict[str, tuple[torch.Tensor, np.ndarray]]) -> DataSplits:
    """Separa os exames normais usados no treino do VAE.

    Args:
        data (dict): Saída de dataset.load_data, com os splits "train", "val" e "test".

    Returns:
        DataSplits: Os três splits completos e as versões só com normais de treino e
            validação.
    """
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    x_test, y_test = data["test"]

    x_train_normal = x_train[torch.from_numpy(y_train == 0)]
    x_val_normal = x_val[torch.from_numpy(y_val == 0)]

    logger.info(f"Treino do VAE: {len(x_train_normal)} exames normais")
    logger.info(f"Validação: {len(x_val)} exames ({len(x_val_normal)} normais)")
    logger.info(f"Teste: {len(x_test)} exames ({int((y_test == 1).sum())} pneumonias)")

    return DataSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        x_train_normal=x_train_normal,
        x_val_normal=x_val_normal,
    )
