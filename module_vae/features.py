"""Do resíduo de reconstrução ao score de anomalia.

O resíduo r = x − x̂ mistura duas coisas de natureza diferente:

- baixa frequência (regiões inteiras mais claras ou mais escuras que o esperado): é aqui
  que uma opacidade de pneumonia aparece;
- alta frequência (bordas de costelas, vasos): o VAE perde esse detalhe em todos os
  exames, porque reconstrói borrado. Como os normais têm mais detalhe fino, essa parte do
  erro é maior nos normais e cancela o sinal útil.

Por isso o score suaviza o resíduo com um filtro gaussiano de largura σ antes de medir o
erro (Seção 7 do notebook). Com σ = 0, o score volta a ser o erro quadrático médio comum.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF


def smooth(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Aplica um filtro gaussiano de largura sigma em imagens (N, 1, H, W)."""
    if sigma <= 0:
        return img

    kernel_size = int(4 * sigma) | 1  # tamanho de kernel ímpar ≈ 4σ
    return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)


def anomaly_map(x: torch.Tensor, x_rec: torch.Tensor, sigma: float) -> torch.Tensor:
    """Mapa de calor: resíduo suavizado e elevado ao quadrado, pixel a pixel.

    Args:
        x (torch.Tensor): Exames originais (N, 1, H, W).
        x_rec (torch.Tensor): Reconstruções do modelo, mesmo formato.
        sigma (float): Largura da suavização do resíduo, em pixels.

    Returns:
        torch.Tensor: Mapa (N, 1, H, W). Valores altos indicam onde o exame foge do
            "normal" reconstruído.
    """
    return smooth(x - x_rec, sigma) ** 2


def anomaly_score(x: torch.Tensor, x_rec: torch.Tensor, sigma: float) -> np.ndarray:
    """Score de anomalia de cada exame: a média do mapa de calor.

    Returns:
        np.ndarray: Score (N,). Quanto maior, mais o exame foge do padrão normal.
    """
    return anomaly_map(x, x_rec, sigma).mean(dim=(1, 2, 3)).numpy()


def detail_level(x: torch.Tensor, sigma: float = 1.5) -> np.ndarray:
    """Nível de detalhe fino de cada imagem: |x − suavizar(x)| médio.

    É a heurística da Seção 7.2 do notebook ("menos detalhe = pneumonia"). Só serve de
    referência: ela foi descoberta olhando os rótulos de pneumonia.
    """
    return (x - smooth(x, sigma)).abs().mean(dim=(1, 2, 3)).numpy()


def brightness(x: torch.Tensor) -> np.ndarray:
    """Brilho médio de cada imagem."""
    return x.mean(dim=(1, 2, 3)).numpy()
