# Métricas: o que entra e o que não entra

Números do modelo só PA no RSNA (teste: 1.082 normais e 203 opacidades).

## A tarefa de treino é diferente da tarefa avaliada

| | O que mede | Onde aparece |
|---|---|---|
| **Loss** (reconstrução + KL) | o modelo aprendeu o "normal"? | curvas de treino (`training_curves.png`) |
| **AUC, AP, precision@k** | o score separa normais de alterados? | avaliação no teste |
| **Precision, recall, F1** | o modelo como alarme, num limiar escolhido | tabela de limiares (`thresholds.csv`) |

Nos classificadores supervisionados (Olist, CNN), a tarefa treinada e a avaliada são a mesma. No VAE, não.

## Loss

| Época | Reconstrução (treino) | Reconstrução (validação) | KL |
|---|---|---|---|
| 10 | 9.156 | 9.160 | 28,2 |
| 20 | 9.132 | 9.146 | 25,8 |
| 30 | 9.116 | 9.135 | 25,6 |

- A loss cai e se estabiliza, com treino e validação próximos: **sem overfitting**.
- O KL não vai a zero: o espaço latente está sendo usado (**sem colapso do posterior**).
- **A loss não mede a detecção:** ela é calculada só com normais. Loss menor ≠ detector melhor. No notebook, com latente 32 o modelo reconstrói melhor, mas reconstrói até as anomalias, e a AUC cai.
- O score de anomalia é parecido com uma "loss por exame" (erro de reconstrução suavizado), mas é avaliado comparando normais com alterados.

## Métricas principais: não dependem de limiar

O objetivo é **ordenar a fila**, então as métricas principais avaliam a ordenação:

- **AUC-ROC = 0,757**: chance de uma opacidade sorteada ter score maior que um normal sorteado.
- **AP (área sob a curva precisão × recall) = 0,449**, contra **0,158** do acaso (a AP do acaso é a proporção de positivos, não 0,5).
- **Topo da fila (simulação, 100 exames com 10% de positivos):** entre os 10 primeiros, **3,7** opacidades com o VAE contra **1,0** na ordem de chegada, ou seja, precision@10 de **37% × 10%**. A posição média de uma opacidade vai de **51** para **27**.

## Precision e recall: quando o modelo vira alarme

| Limiar (no teste) | Recall (sensibilidade) | Especificidade | Precision (VPP) |
|---|---|---|---|
| percentil 95 dos normais (sem rótulos, padrão) | 29% | 96% | 56% |
| percentil 90 dos normais | 41% | 91% | 47% |
| percentil 80 dos normais | 58% | 82% | 37% |
| Youden (usa rótulos, só referência) | 62% | 79% | 35% |

Para pegar mais opacidades, é preciso aceitar mais falsos alarmes. No limiar padrão: **58** opacidades encontradas, **145** perdidas, **46** falsos alarmes entre 1.082 normais. F1 = 0,38.

**Conclusão:** como alarme binário o modelo é fraco; como ordenador da fila, funciona.

## Três armadilhas

1. **A acurácia engana:** no limiar p95 ela dá **85,1%**. Um "modelo" que diz "normal" para todo mundo teria **84,2%**, porque 84% do teste é normal.
2. **A precision depende da prevalência:** no teste (15,8% de opacidades) ela é 56%. Numa fila real com 10%, os mesmos 29% de recall e 96% de especificidade dariam cerca de **43%**.
3. **O limiar não é escolhido pelo F1, como no Olist, de propósito:** otimizar o F1 exige rótulos da doença, e um detector de anomalias só conhece normais. Por isso o limiar padrão é um percentil dos normais da validação.

## Acurácia em deep learning

- **No VAE**, não existe acurácia durante o treino: ele não recebe rótulos e não classifica nada.
- **Na CNN de comparação**, daria para acompanhar a acurácia por época, mas com classes desbalanceadas a AUC é mais informativa.
- Os tutoriais mostram "curvas de loss e acurácia" porque quase sempre tratam de classificadores supervisionados.

Na apresentação: as curvas de loss entram em **implementação** ("o treino convergiu"); AUC, AP, fila e limiares entram em **avaliação**.
