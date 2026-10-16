# Triagem de raio-X de tórax com VAE — prova de conceito

Um **Autoencoder Variacional (VAE)** treinado **somente com raios-X normais** consegue apontar quais exames fogem do padrão, indicar onde está a alteração e reordenar uma fila de exames por prioridade?

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rmotti/vaes_poc/blob/main/notebooks/poc_vae_raiox.ipynb)

## Como rodar

1. Abra o notebook no Colab pelo botão acima.
2. *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)*.
3. *Executar tudo*. Leva cerca de 5–10 minutos com GPU. Os dados (PneumoniaMNIST, 64×64 px) são baixados automaticamente.

## Estrutura

```
notebooks/poc_vae_raiox.ipynb   notebook completo: problema, fundamentação, arquitetura,
                                treino, avaliação, análise de erros e conclusões
```

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
