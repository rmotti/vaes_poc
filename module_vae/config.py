from pathlib import Path

from loguru import logger
import torch

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = PROJ_ROOT / "models"
MODEL_PATH = MODELS_DIR / "vae.pt"
METADATA_PATH = MODELS_DIR / "metadata.json"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Hiperparâmetros (os mesmos da configuração do notebook)
IMAGE_SIZE = 64  # resolução (o MedMNIST oferece 28, 64, 128 e 224 px)
LATENT_DIM = 8  # tamanho do código latente z
BETA = 1.0  # peso do termo KL (β = 1 → VAE clássico)
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
KL_WARMUP_EPOCHS = 10  # o peso do KL cresce de 0 até β nas primeiras épocas
SEEDS = [0, 1, 2]  # o VAE e o autoencoder comum são treinados com 3 sementes

# Score de anomalia
# Larguras testadas na suavização do resíduo. A escolhida é a de maior AUC na validação.
SIGMAS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
# Limiar sem rótulos: o percentil 95 dos normais da validação aceita ~5% de falsos alarmes.
THRESHOLD_PERCENTILE = 95


def get_device() -> torch.device:
    """Escolhe a GPU disponível (CUDA ou Apple MPS) e cai para a CPU se não houver."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
