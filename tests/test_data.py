from pathlib import Path

import numpy as np
import pytest
import torch

from module_vae.modeling.split import split_data

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
NPZ_PATH = RAW_DATA_DIR / "pneumoniamnist_64.npz"

requires_data = pytest.mark.skipif(
    not NPZ_PATH.exists(), reason="PneumoniaMNIST não baixado; rode `make data`"
)


@requires_data
def test_splits_have_expected_sizes():
    data = np.load(NPZ_PATH)

    assert data["train_images"].shape == (4708, 64, 64)
    assert data["val_images"].shape == (524, 64, 64)
    assert data["test_images"].shape == (624, 64, 64)


@requires_data
def test_labels_are_binary():
    data = np.load(NPZ_PATH)

    for split in ["train", "val", "test"]:
        labels = data[f"{split}_labels"]
        assert set(np.unique(labels)) == {0, 1}
        assert len(labels) == len(data[f"{split}_images"])


@requires_data
def test_images_are_uint8():
    data = np.load(NPZ_PATH)

    for split in ["train", "val", "test"]:
        assert data[f"{split}_images"].dtype == np.uint8


def test_split_keeps_only_normals_for_training():
    y_train = np.array([0, 1, 0, 1, 1])
    y_val = np.array([1, 0, 0])
    # Cada imagem é preenchida com o próprio rótulo, para saber de onde veio.
    data = {
        "train": (torch.tensor(y_train, dtype=torch.float32).view(-1, 1, 1, 1), y_train),
        "val": (torch.tensor(y_val, dtype=torch.float32).view(-1, 1, 1, 1), y_val),
        "test": (torch.zeros(2, 1, 1, 1), np.array([0, 1])),
    }

    splits = split_data(data)

    assert len(splits.x_train_normal) == 2
    assert len(splits.x_val_normal) == 2
    assert (splits.x_train_normal == 0).all()
    assert (splits.x_val_normal == 0).all()
    assert len(splits.x_test) == 2
