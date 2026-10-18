# Triagem de raio-X de tórax com VAE — prova de conceito

Um **Autoencoder Variacional (VAE)** treinado **somente com raios-X normais** consegue apontar quais exames fogem do padrão, indicar onde está a alteração e reordenar uma fila de exames por prioridade?

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rmotti/vaes_poc/blob/main/notebooks/poc_vae_raiox.ipynb)

## Como rodar

### Forma 1: notebook no Google Colab

É a versão didática e a primeira versão da PoC, feita com o PneumoniaMNIST (raios-X pediátricos, 64×64 px). Explica cada etapa: fundamentação, arquitetura, treino, avaliação e análise de erros.

1. Abra o notebook no Colab pelo botão acima.
2. *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)*.
3. *Executar tudo*. Leva cerca de 5–10 minutos com GPU. Os dados são baixados automaticamente.

### Forma 2: pipeline local com o RSNA

O código do notebook foi reorganizado em um pacote (`module_vae`) no padrão
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) e aplicado ao
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge):
26.684 raios-X de adultos, com três classes (normal, opacidade pulmonar e anormal sem
opacidade) e caixas marcando as opacidades.

**Pré-requisitos:**

- [uv](https://docs.astral.sh/uv/) instalado. O pipeline usa a GPU automaticamente quando existe (CUDA ou Apple Silicon/MPS).
- Conta no Kaggle, com as regras da competição aceitas na página do RSNA.
- Token da API do Kaggle salvo em `~/.kaggle/access_token` (*Settings → API → Create New Token*).

```bash
make requirements   # cria o .venv e instala as dependências (Python 3.12)
make data           # baixa o RSNA (~3,7 GB) e guarda as imagens em 128 px
make train          # pipeline completo, só incidência PA (resultado principal)
make train_ap_pa    # o mesmo com AP e PA juntas, para mostrar o atalho AP/PA
make predict        # inferência com o modelo salvo em 5 exames do teste
make ablation       # (opcional) experimentos com tamanho do latente e β
make test           # testes do pacote
```

Para conferir rapidamente se tudo funciona (2 épocas, 1 semente, sem baselines), use
`make quick`. As opções do pipeline aparecem em `uv run python -m module_vae.main --help`.

Para avaliar um raio-X qualquer (png/jpg, frontal) com o modelo treinado:

```bash
uv run python -m module_vae.inference --image caminho/do/exame.png
```

**Desenho do experimento:**

| | |
|---|---|
| Dados | só incidência PA; split por paciente: 70% treino, 15% validação, 15% teste |
| Resolução | 128 px |
| Treino do VAE | só os exames normais do treino |
| Validação | escolha da suavização do score (σ) e do limiar de decisão |
| Teste | opacidade × normal; anormal sem opacidade × normal; mapa de calor × caixas; fila simulada |
| Comparação | brilho médio, imagem média, PCA, autoencoder comum, CNN supervisionada |

**Por que só PA:** no RSNA, 78% das opacidades são exames AP (paciente no leito), contra 18%
dos normais. Um modelo treinado com as duas incidências pode aprender a diferença AP × PA
em vez da doença. `make train_ap_pa` roda o mesmo pipeline com as duas incidências para
medir esse atalho.

Cada execução salva os resultados na sua pasta (`rsna_pa` ou `rsna_ap_pa`):

| Arquivo | Conteúdo |
|---|---|
| `models/<execução>/vae.pt` | pesos do VAE |
| `models/<execução>/metadata.json` | σ do score, limiar e escala do mapa de calor, todos calibrados na validação |
| `reports/<execução>/metrics.json` | métricas no teste, outras anomalias, localização, incidência, fila e comparação de métodos |
| `reports/<execução>/test_predictions.csv` | score, decisão e posição na fila de cada exame de teste |
| `reports/<execução>/thresholds.csv` | sensibilidade e especificidade de cada limiar (validação × teste) |
| `reports/<execução>/method_comparison.csv` | AUC do VAE e das alternativas |
| `reports/<execução>/figures/*.png` | curvas de treino, escolha de σ, ROC, mapas de calor com as caixas, galeria de erros, curva de ganho |

## Estrutura

```
├── Makefile               <- atalhos: make data, make train, make predict...
├── pyproject.toml         <- dependências e configuração do ruff
├── data
│   ├── raw/rsna           <- arquivos do Kaggle (não versionados)
│   └── processed          <- imagens redimensionadas (não versionadas)
├── models                 <- modelos treinados e metadados, uma pasta por execução
├── notebooks
│   └── poc_vae_raiox.ipynb   <- notebook completo (forma 1)
├── reports                <- métricas, tabelas e figuras, uma pasta por execução
├── tests                  <- testes do pacote (pytest)
└── module_vae             <- código do pipeline (forma 2)
    ├── config.py          <- caminhos, hiperparâmetros e dispositivo (CPU/GPU)
    ├── dataset.py         <- download do RSNA e leitura dos DICOMs
    ├── features.py        <- do resíduo de reconstrução ao score e ao mapa de calor
    ├── plots.py           <- figuras
    ├── main.py            <- pipeline completo (make train)
    ├── inference.py       <- inferência com o modelo salvo (make predict)
    └── modeling
        ├── architecture.py   <- VAE, loss e CNN supervisionada
        ├── split.py          <- split por paciente; treino só com exames normais
        ├── train.py          <- loop de treino e salvamento do modelo
        ├── calibration.py    <- escolha de σ e do limiar na validação
        ├── evaluate.py       <- métricas, outras anomalias, localização, incidência e fila
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

## Dados

- **Pipeline:** [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (RSNA, NIH, Society of Thoracic Radiology e MD.ai). O uso segue as regras da competição no Kaggle.
- **Notebook:** PneumoniaMNIST (MedMNIST v2, Yang et al., 2023), derivado de Kermany et al. (2018), licença CC BY 4.0.

Não é um dispositivo médico e não deve ser usado para diagnóstico.
