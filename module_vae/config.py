from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import torch

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RSNA_DIR = RAW_DATA_DIR / "rsna"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

# Rastreamento dos experimentos no MLflow (local e fora do git): parâmetros e métricas em
# mlflow.db, artefatos (figuras, tabelas, pesos) em mlartifacts/
MLFLOW_TRACKING_URI = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"

# Resolução das imagens. Os DICOMs do RSNA têm 1024 px; em 64 px (a resolução do
# notebook) opacidades pequenas somem.
IMAGE_SIZE = 128

# Hiperparâmetros (os mesmos da configuração do notebook)
LATENT_DIM = 8  # tamanho do código latente z
BETA = 1.0  # peso do termo KL (β = 1 → VAE clássico)
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
KL_WARMUP_EPOCHS = 10  # o peso do KL cresce de 0 até β nas primeiras épocas
SEEDS = [0, 1, 2]  # o VAE e o autoencoder comum são treinados com 3 sementes

# Score de anomalia
# Larguras testadas na suavização do resíduo, em pixels de uma imagem de 64 px. Em outras
# resoluções elas são escaladas, para cobrir as mesmas estruturas anatômicas. A escolhida
# é a de maior AUC na validação.
SIGMAS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
# Limiar sem rótulos: o percentil 95 dos normais da validação aceita ~5% de falsos alarmes.
THRESHOLD_PERCENTILE = 95


@dataclass(frozen=True)
class RunPaths:
    """Onde cada execução do pipeline guarda modelo e relatórios.

    Cada combinação de incidências vira uma pasta própria (models/rsna_pa e
    reports/rsna_pa, ou rsna_ap_pa), para os resultados não se sobrescreverem.
    """

    name: str

    @property
    def model(self) -> Path:
        return MODELS_DIR / self.name / "vae.pt"

    @property
    def metadata(self) -> Path:
        return MODELS_DIR / self.name / "metadata.json"

    @property
    def reports(self) -> Path:
        return REPORTS_DIR / self.name

    @property
    def figures(self) -> Path:
        return REPORTS_DIR / self.name / "figures"


def run_name(views: list[str]) -> str:
    """Nome da execução: "rsna_pa" ou "rsna_ap_pa"."""
    return "rsna_" + "_".join(sorted(v.lower() for v in views))


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
