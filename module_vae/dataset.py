"""Download e carregamento do RSNA Pneumonia Detection Challenge (Kaggle).

São 26.684 raios-X de tórax de adultos com rótulos, um por paciente, em DICOM de 1024 px.
Cada exame tem uma de três classes (normal, opacidade pulmonar ou anormal sem opacidade),
e as opacidades vêm marcadas com caixas. Os DICOMs são lidos uma vez e guardados
redimensionados em data/processed.
"""

from pathlib import Path

import kagglehub
from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from tqdm import tqdm
import typer

from module_vae.config import IMAGE_SIZE, PROCESSED_DATA_DIR, RSNA_DIR

RSNA_COMPETITION = "rsna-pneumonia-detection-challenge"
RSNA_ORIGINAL_SIZE = 1024
RSNA_CLASSES = {
    "Normal": "normal",
    "Lung Opacity": "opacity",
    "No Lung Opacity / Not Normal": "other",
}

app = typer.Typer()


def download_rsna(root: Path = RSNA_DIR) -> None:
    """Baixa os arquivos da competição do Kaggle, se ainda não estiverem em root.

    Precisa de um token da API do Kaggle (~/.kaggle/access_token ou kaggle.json) e de as
    regras da competição terem sido aceitas no site.
    """
    if (root / "stage_2_train_labels.csv").exists():
        return

    logger.info(f"Baixando {RSNA_COMPETITION} do Kaggle para {root}...")
    kagglehub.competition_download(RSNA_COMPETITION, output_dir=str(root))


def read_dicom(path: Path, image_size: int) -> tuple[np.ndarray, str]:
    """Lê um DICOM e devolve a imagem redimensionada (uint8) e a incidência (AP ou PA)."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (255 * (img - img.min()) / max(img.max() - img.min(), 1)).astype(np.uint8)
    # Em MONOCHROME1 o osso aparece escuro; invertendo, todos ficam com osso claro.
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 255 - img

    img = Image.fromarray(img).resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img), ds.ViewPosition


def rsna_cache_path(image_size: int) -> Path:
    return PROCESSED_DATA_DIR / f"rsna_{image_size}.npz"


def build_rsna_cache(image_size: int, root: Path = RSNA_DIR) -> Path:
    """Lê os 26.684 DICOMs de treino uma vez e guarda as imagens redimensionadas.

    Só o conjunto de treino da competição tem rótulos; as imagens de teste dela não são
    usadas. Os splits de treino, validação e teste da PoC saem desse conjunto.
    """
    download_rsna(root)

    # Há uma linha por caixa; a classe é a mesma em todas as linhas de um paciente.
    meta = pd.read_csv(root / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")

    images, views = [], []
    for patient_id in tqdm(meta["patientId"], desc=f"DICOM → {image_size} px"):
        img, view = read_dicom(root / "stage_2_train_images" / f"{patient_id}.dcm", image_size)
        images.append(img)
        views.append(view)

    path = rsna_cache_path(image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.stack(images),
        patient_ids=meta["patientId"].to_numpy().astype(str),
        labels=meta["class"].map(RSNA_CLASSES).to_numpy().astype(str),
        views=np.array(views),
    )
    logger.success(f"Cache do RSNA salvo em {path}")

    return path


def load_rsna(
    image_size: int = IMAGE_SIZE,
    root: Path = RSNA_DIR,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Carrega o RSNA redimensionado, criando o cache na primeira vez.

    Returns:
        tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
            - images: (N, H, W) em uint8;
            - meta: uma linha por exame, com patientId, label ("normal", "opacity" ou
              "other") e view ("AP" ou "PA"), na mesma ordem de images;
            - boxes: caixas das opacidades (patientId, x, y, width, height), já em pixels
              da imagem redimensionada.
    """
    path = rsna_cache_path(image_size)
    if not path.exists():
        build_rsna_cache(image_size, root)

    cache = np.load(path)
    meta = pd.DataFrame(
        {
            "patientId": cache["patient_ids"],
            "label": cache["labels"],
            "view": cache["views"],
        }
    )

    boxes = pd.read_csv(root / "stage_2_train_labels.csv").dropna(subset=["x"])
    scale = image_size / RSNA_ORIGINAL_SIZE
    boxes[["x", "y", "width", "height"]] *= scale

    logger.info(f"RSNA carregado: {cache['images'].shape} | {len(boxes)} caixas")
    return cache["images"], meta, boxes[["patientId", "x", "y", "width", "height"]]


@app.command()
def main():
    """Baixa o RSNA e cria o cache das imagens redimensionadas."""
    _, meta, _ = load_rsna()
    logger.success(f"Exames por classe e incidência:\n{pd.crosstab(meta.label, meta.view)}")


if __name__ == "__main__":
    app()
