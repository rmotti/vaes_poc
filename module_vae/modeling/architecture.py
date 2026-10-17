"""Arquitetura do VAE (Seção 4 do notebook) e da CNN supervisionada usada como referência.

raio-X x ─► Encoder (4 conv: 64→32→16→8→4 px) ─► μ, log σ² ─► z = μ + σ ⊙ ε
         ─► Decoder (4 deconv: 4→8→16→32→64 px) ─► reconstrução x̂

Loss = BCE(x, x̂) + β · KL( N(μ, σ²) ‖ N(0, I) )
"""

from itertools import pairwise

from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F

CHANNELS = [1, 32, 64, 128, 256]
FLAT_SIZE = 256 * 4 * 4  # saída do encoder para imagens de 64×64 px


def conv_block(c_in: int, c_out: int) -> list[nn.Module]:
    """Convolução com passo 2 (reduz a imagem pela metade) + batch norm + LeakyReLU."""
    return [
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2),
    ]


class VAE(nn.Module):
    """Autoencoder variacional convolucional para imagens 1×64×64.

    Com deterministic=True, a mesma rede vira um autoencoder comum (z = μ, sem KL), usado
    como baseline para isolar o efeito da parte variacional.
    """

    def __init__(self, latent_dim: int = 8, deterministic: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.deterministic = deterministic

        layers = []
        for c_in, c_out in pairwise(CHANNELS):
            layers += conv_block(c_in, c_out)
        self.encoder = nn.Sequential(*layers, nn.Flatten())  # → 256×4×4 = 4096
        self.fc_mu = nn.Linear(FLAT_SIZE, latent_dim)
        self.fc_logvar = nn.Linear(FLAT_SIZE, latent_dim)

        self.fc_decoder = nn.Linear(latent_dim, FLAT_SIZE)
        reversed_channels = CHANNELS[::-1]  # 256, 128, 64, 32, 1
        layers = []
        for c_in, c_out in pairwise(reversed_channels[:-1]):
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2),
            ]
        layers += [
            nn.ConvTranspose2d(reversed_channels[-2], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixels em [0, 1]
        ]
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Truque da reparametrização: z = μ + σ·ε. Fora do treino (ou no autoencoder
        # comum), usamos z = μ, o que deixa a inferência determinística.
        if self.deterministic or not self.training:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decoder(z).view(-1, 256, 4, 4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_terms(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula os dois termos da loss do VAE, um valor por imagem.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - reconstrução: BCE somada sobre os 4.096 pixels de cada imagem;
            - KL entre N(μ, σ²) e N(0, I), fórmula fechada, somada sobre as dimensões
              latentes.
    """
    reconstruction = F.binary_cross_entropy(x_rec, x, reduction="none").sum(dim=(1, 2, 3))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return reconstruction, kl


def create_supervised_cnn() -> nn.Sequential:
    """Mesmo encoder do VAE + uma camada de classificação (logit de pneumonia).

    É a referência supervisionada: treinada com normais e pneumonias, dá o "teto"
    esperado para a tarefa.
    """
    layers = []
    for c_in, c_out in pairwise(CHANNELS):
        layers += conv_block(c_in, c_out)
    return nn.Sequential(*layers, nn.Flatten(), nn.Dropout(0.3), nn.Linear(FLAT_SIZE, 1))


def describe_architecture(model: VAE) -> None:
    """Registra no log o formato do tensor depois de cada camada e o total de parâmetros."""
    model = model.cpu().eval()
    rows = []
    with torch.no_grad():
        h = torch.zeros(1, 1, 64, 64)
        rows.append(("entrada (raio-X)", tuple(h.shape[1:])))
        for layer in model.encoder:
            h = layer(h)
            if isinstance(layer, (nn.Conv2d, nn.Flatten)):
                rows.append((f"encoder · {type(layer).__name__}", tuple(h.shape[1:])))
        rows.append(("μ e log σ²", (model.latent_dim,)))
        h = model.fc_decoder(torch.zeros(1, model.latent_dim)).view(-1, 256, 4, 4)
        rows.append(("decoder · Linear + reshape", tuple(h.shape[1:])))
        for layer in model.decoder:
            h = layer(h)
            if isinstance(layer, nn.ConvTranspose2d):
                rows.append(("decoder · ConvTranspose2d", tuple(h.shape[1:])))

    n_params = sum(p.numel() for p in model.parameters())
    table = "\n".join(f"  {name:<30} {shape}" for name, shape in rows)
    logger.info(f"Arquitetura do VAE:\n{table}\n  Total de parâmetros: {n_params:,}")
