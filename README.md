# Triagem de raio-X de tórax com VAE — prova de conceito

Um **Autoencoder Variacional (VAE)** treinado **somente com raios-X normais** consegue apontar quais exames fogem do padrão, indicar onde está a alteração e reordenar uma fila de exames por prioridade?

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rmotti/vaes_poc/blob/main/notebooks/poc_vae_raiox.ipynb)

## Como rodar

Há duas formas, que treinam o mesmo modelo com os mesmos hiperparâmetros.

### Forma 1: notebook no Google Colab

É a versão didática, com a explicação de cada etapa.

1. Abra o notebook no Colab pelo botão acima.
2. *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)*.
3. *Executar tudo*. Leva cerca de 5–10 minutos com GPU. Os dados (PneumoniaMNIST, 64×64 px) são baixados automaticamente.

### Forma 2: pipeline local (módulo Python)

O código do notebook foi reorganizado em um pacote (`module_vae`) no padrão
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). Precisa do
[uv](https://docs.astral.sh/uv/) instalado. Usa a GPU automaticamente quando existe (CUDA ou
Apple Silicon/MPS).

```bash
make requirements   # cria o .venv e instala as dependências (Python 3.12)
make data           # baixa o PneumoniaMNIST para data/raw/
make train          # pipeline completo: treino → calibração → avaliação → comparação
make predict        # inferência com o modelo salvo em 5 exames do teste
make ablation       # (opcional) experimentos com tamanho do latente e β
make test           # testes do pacote
```

Para conferir rapidamente se tudo funciona (2 épocas, 1 semente, sem baselines), use
`make quick`. As opções do pipeline aparecem em `uv run python -m module_vae.main --help`.

Para avaliar um raio-X qualquer (png/jpg) com o modelo treinado:

```bash
uv run python -m module_vae.inference --image caminho/do/exame.png
```

O pipeline gera:

| Arquivo | Conteúdo |
|---|---|
| `models/vae.pt` | pesos do VAE |
| `models/metadata.json` | σ do score, limiar e escala do mapa de calor, todos calibrados na validação |
| `reports/metrics.json` | métricas no teste, simulação da fila e comparação de métodos |
| `reports/test_predictions.csv` | score, decisão e posição na fila de cada exame de teste |
| `reports/thresholds.csv` | sensibilidade e especificidade de cada limiar (validação × teste) |
| `reports/method_comparison.csv` | AUC do VAE e das alternativas |
| `reports/figures/*.png` | curvas de treino, escolha de σ, ROC, mapas de calor, galeria de erros, curva de ganho |

## Estrutura

```
├── Makefile               <- atalhos: make data, make train, make predict...
├── pyproject.toml         <- dependências e configuração do ruff
├── data/raw               <- PneumoniaMNIST baixado (não versionado)
├── models                 <- modelo treinado e metadados
├── notebooks
│   └── poc_vae_raiox.ipynb   <- notebook completo (forma 1)
├── reports                <- métricas e tabelas geradas pelo pipeline
│   └── figures            <- figuras geradas pelo pipeline
├── tests                  <- testes do pacote (pytest)
└── module_vae             <- código do pipeline (forma 2)
    ├── config.py          <- caminhos, hiperparâmetros e dispositivo (CPU/GPU)
    ├── dataset.py         <- download e carregamento do PneumoniaMNIST
    ├── features.py        <- do resíduo de reconstrução ao score e ao mapa de calor
    ├── plots.py           <- figuras
    ├── main.py            <- pipeline completo (make train)
    ├── inference.py       <- inferência com o modelo salvo (make predict)
    └── modeling
        ├── architecture.py   <- VAE, loss e CNN supervisionada
        ├── split.py          <- treino só com exames normais
        ├── train.py          <- loop de treino e salvamento do modelo
        ├── calibration.py    <- escolha de σ e do limiar na validação
        ├── evaluate.py       <- métricas no teste e simulação da fila
        ├── baselines.py      <- comparação com brilho, PCA, autoencoder comum e CNN
        ├── predict.py        <- reconstrução, score e posição na fila
        └── ablation.py       <- experimentos com hiperparâmetros
```

Onde cada seção do notebook foi parar:

| Seção do notebook | Módulo |
|---|---|
| 2. Configuração | `config.py` |
| 3. Dados | `dataset.py`, `modeling/split.py` |
| 4. Arquitetura | `modeling/architecture.py` |
| 5. Treinamento | `modeling/train.py` |
| 7. Score de anomalia | `features.py`, `modeling/calibration.py` |
| 8.1–8.4 Avaliação, limiar e fila | `modeling/evaluate.py` |
| 8.2 Comparação com alternativas | `modeling/baselines.py` |
| 8.5 Hiperparâmetros | `modeling/ablation.py` |
| 9. Análise de erros | figuras `false_positives.png` e `false_negatives.png` |
| 11. Salvando o modelo | `modeling/train.py` (`save_model`), `modeling/predict.py` (`load_model`) |

As seções 1 (fundamentação), 6 (exploração do espaço latente) e 10 (conclusões) são
explicação e análise visual, e continuam só no notebook. Da seção 6, o pipeline gera apenas
as amostras sintéticas (`generated_samples.png`).

## Resultados principais

AUC-ROC no conjunto de teste (624 exames). O VAE e o autoencoder comum aparecem como média ± desvio de 3 sementes. Os valores podem variar um pouco entre execuções.

| Método | Usa rótulos de pneumonia? | AUC teste |
|---|---|---|
| Brilho médio da imagem | não | 0,58 |
| PCA (8 componentes) | não | 0,73 |
| Autoencoder comum | não | 0,83 ± 0,02 |
| **VAE** | não | **0,83 ± 0,01** |
| Heurística "nível de detalhe" | sim (descoberta com rótulos) | 0,91 |
| CNN supervisionada | sim | 0,92 |

- **O erro de reconstrução comum funciona mal** (AUC de 0,62 no teste). Ele mistura dois sinais que apontam em direções opostas: o detalhe fino que o VAE perde (maior nos normais) e as opacidades (maiores na pneumonia). Comparar o exame com a reconstrução **em baixa frequência** resolve o problema.
- **O VAE empata com um autoencoder comum** de mesma arquitetura. O ganho vem do gargalo não linear, e não da parte variacional. Em troca, o VAE é mais estável entre sementes e tem um espaço latente gerativo.
- **Na triagem simulada** (filas de 100 exames com 10% de pneumonias), a posição média de uma pneumonia na fila passa de cerca de 51 para cerca de 21.
- **Limitações:** dataset pediátrico de um único hospital, resolução de 64 px, mudança de distribuição entre validação e teste, possíveis atalhos do dataset. Não é um dispositivo médico.

## Dados

PneumoniaMNIST (MedMNIST v2, Yang et al., 2023), derivado de Kermany et al. (2018), licença CC BY 4.0.
