"""Montagem dos conjuntos: o ponto central do desenho experimental.

O VAE só vê exames normais no treino. Os exames alterados do treino ficam de fora (só a
CNN supervisionada, usada como referência, os utiliza).

- Treino do VAE: os exames normais do split de treino.
- Validação: acompanhar o treino (normais), calibrar o limiar (normais) e escolher σ
  (normais + positivos).
- Teste: reservado para a avaliação final.

O "positivo" é a opacidade pulmonar (pneumonia). Os exames "anormais sem opacidade" do
teste ficam à parte (x_test_other): eles medem se o VAE sinaliza alterações que não são
pneumonia, sem terem sido usados para escolher nada.
"""

from dataclasses import dataclass

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from module_vae.dataset import load_rsna


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
    x_test_other: torch.Tensor  # anormais sem opacidade
    boxes_test: pd.DataFrame  # caixas das opacidades (test_index, x, y, width, height)
    views_test: np.ndarray  # incidência (AP ou PA) de cada exame de x_test


def split_rsna(
    images: np.ndarray,
    meta: pd.DataFrame,
    boxes: pd.DataFrame,
    views: list[str],
    seed: int = 42,
) -> DataSplits:
    """Divide o RSNA em 70% treino, 15% validação e 15% teste.

    Args:
        images (np.ndarray): Imagens (N, H, W) em uint8.
        meta (pd.DataFrame): patientId, label e view de cada imagem.
        boxes (pd.DataFrame): Caixas das opacidades (patientId, x, y, width, height).
        views (list[str]): Incidências mantidas ("PA", ou "AP" e "PA").
        seed (int): Semente do sorteio.

    Returns:
        DataSplits: Conjuntos com normais (0) e opacidades (1), mais os anormais sem
            opacidade, as caixas e as incidências do teste.
    """
    # Há um exame por paciente, então separar exames é o mesmo que separar pacientes:
    # ninguém aparece em dois conjuntos. A estratificação mantém a proporção de cada
    # classe e incidência nos três conjuntos.
    strata = (meta["label"] + "_" + meta["view"]).to_numpy()
    idx = np.arange(len(meta))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=strata, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=strata[rest_idx], random_state=seed
    )

    # O filtro de incidência vem depois do sorteio: assim o teste só com PA é um pedaço
    # do teste com AP e PA, e os dois experimentos podem ser comparados.
    labels = meta["label"].to_numpy()
    in_views = meta["view"].isin(views).to_numpy()

    def select(split_idx: np.ndarray, classes: list[str]) -> np.ndarray:
        return split_idx[in_views[split_idx] & np.isin(labels[split_idx], classes)]

    def to_tensor(ix: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(images[ix]).float().unsqueeze(1) / 255.0

    train_ix = select(train_idx, ["normal", "opacity"])
    val_ix = select(val_idx, ["normal", "opacity"])
    test_ix = select(test_idx, ["normal", "opacity"])
    other_ix = select(test_idx, ["other"])

    y_train = (labels[train_ix] == "opacity").astype(int)
    y_val = (labels[val_ix] == "opacity").astype(int)
    y_test = (labels[test_ix] == "opacity").astype(int)
    x_train, x_val = to_tensor(train_ix), to_tensor(val_ix)

    # Caixas das opacidades do teste, com a posição de cada exame em x_test
    position = {pid: i for i, pid in enumerate(meta["patientId"].to_numpy()[test_ix])}
    boxes_test = boxes[boxes["patientId"].isin(position)].assign(
        test_index=lambda df: df["patientId"].map(position)
    )

    counts = pd.DataFrame(
        {
            name: pd.Series(labels[ix]).value_counts()
            for name, ix in [
                ("treino", select(train_idx, list(set(labels)))),
                ("validação", select(val_idx, list(set(labels)))),
                ("teste", select(test_idx, list(set(labels)))),
            ]
        }
    ).T
    logger.info(f"RSNA, incidências {views}:\n{counts}")
    logger.info(
        f"Caixas no teste: {len(boxes_test)} em {boxes_test['test_index'].nunique()} exames"
    )

    return DataSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=to_tensor(test_ix),
        y_test=y_test,
        x_train_normal=x_train[torch.from_numpy(y_train == 0)],
        x_val_normal=x_val[torch.from_numpy(y_val == 0)],
        x_test_other=to_tensor(other_ix),
        boxes_test=boxes_test,
        views_test=meta["view"].to_numpy()[test_ix],
    )


def load_splits(image_size: int, views: list[str]) -> DataSplits:
    """Carrega o RSNA na resolução pedida e monta os conjuntos."""
    images, meta, boxes = load_rsna(image_size)
    return split_rsna(images, meta, boxes, views)
