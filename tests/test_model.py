import torch

from module_vae.features import anomaly_score, smooth
from module_vae.modeling.architecture import VAE, loss_terms


def test_vae_output_shapes():
    model = VAE(latent_dim=8)
    x = torch.rand(3, 1, 64, 64)

    x_rec, mu, logvar = model(x)

    assert x_rec.shape == x.shape
    assert mu.shape == (3, 8)
    assert logvar.shape == (3, 8)
    assert x_rec.min() >= 0 and x_rec.max() <= 1


def test_vae_is_deterministic_in_eval_mode():
    model = VAE(latent_dim=8).eval()
    x = torch.rand(2, 1, 64, 64)

    with torch.no_grad():
        assert torch.equal(model(x)[0], model(x)[0])


def test_kl_is_zero_for_standard_normal():
    x = torch.full((2, 1, 64, 64), 0.5)
    mu, logvar = torch.zeros(2, 8), torch.zeros(2, 8)

    reconstruction, kl = loss_terms(x, x, mu, logvar)

    assert torch.allclose(kl, torch.zeros(2))
    assert (reconstruction > 0).all()


def test_score_without_smoothing_is_mse():
    x, x_rec = torch.rand(4, 1, 64, 64), torch.rand(4, 1, 64, 64)

    expected = ((x - x_rec) ** 2).mean(dim=(1, 2, 3)).numpy()

    assert torch.allclose(torch.tensor(anomaly_score(x, x_rec, sigma=0)), torch.tensor(expected))


def test_smoothing_keeps_constant_residual():
    # Uma diferença uniforme (região inteira mais clara) é exatamente o sinal de baixa
    # frequência que a suavização deve preservar.
    residual = torch.full((1, 1, 64, 64), 0.2)

    assert smooth(residual, sigma=3).shape == residual.shape
    assert torch.allclose(smooth(residual, sigma=3), residual)
