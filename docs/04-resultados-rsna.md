# Resultados no RSNA

Execução de 18/10/2026: 40 épocas, 128 px, latente 8, β = 1. Relatórios completos em `reports/rsna_pa/` e `reports/rsna_ap_pa/`.

## Dataset: RSNA Pneumonia Detection Challenge (Kaggle)

- 26.684 raios-X de adultos com rótulos, um por paciente, DICOM de 1024 px (6 a 83 anos, mediana 49).
- Três classes: **normal** (8.851), **opacidade pulmonar** (6.012, com **9.555 caixas** marcando as lesões) e **anormal sem opacidade** (11.821).

**O atalho escondido:** a proporção de exames em incidência AP (paciente no leito, geralmente internado) muda muito entre as classes.

| Classe | % em AP |
|---|---|
| Normal | 18% |
| Anormal sem opacidade | 50% |
| Opacidade | **78%** |

Um modelo pode aprender "AP = doente" em vez da doença. Por isso o experimento principal usa **só PA**.

## Desenho do experimento

| | |
|---|---|
| Dados | só PA; split por paciente 70/15/15, estratificado por classe e incidência |
| Treino do VAE | 5.050 exames normais |
| Validação | 1.082 normais + 202 opacidades: escolha de σ e do limiar |
| Teste | 1.082 normais + 203 opacidades; + 892 "anormais sem opacidade" à parte |
| Comparação | brilho, imagem média, PCA, autoencoder comum (3 sementes), VAE (3 sementes), heurística de detalhe, CNN supervisionada |

## Resultado principal (só PA)

| Método | Usa rótulos? | AUC teste |
|---|---|---|
| Brilho médio | não | 0,49 |
| Imagem média | não | 0,66 |
| PCA (8 componentes) | não | 0,75 |
| Autoencoder comum | não | 0,757 ± 0,002 |
| **VAE** | não | **0,757 ± 0,002** |
| Heurística "nível de detalhe" | sim | 0,55 |
| CNN supervisionada | sim | 0,83 |

## Achados

- **O padrão do notebook se repete:** VAE empata com o autoencoder comum e fica abaixo da CNN. A distância para a CNN caiu de 0,09 para 0,07.
- **A suavização continua essencial:** erro pixel a pixel 0,69 → com σ = 10, 0,76.
- **A heurística de detalhe era viés do PneumoniaMNIST:** 0,91 no notebook, 0,55 no RSNA. Responde a pergunta que o notebook deixou em aberto (fisiologia ou viés de aquisição?).
- **Anomalias nunca vistas:** "anormal sem opacidade" × normal dá AUC **0,69**; 19,5% deles passam do limiar (contra ~5% esperado nos normais). Suporte modesto à promessa de "detectar qualquer alteração".
- **Localização:** o ponto mais quente do mapa cai dentro da caixa do radiologista em **25%** dos casos (acaso: 7%); 13,5% do "calor" fica dentro das caixas (acaso: 7,3%). Melhor que o acaso, mas o mapa acende muito fora da lesão (bordas, ombros, abdômen).
- **Gravidade:** correlação score × área da lesão = **−0,14**. O score não mede gravidade.
- **Fila:** posição média de uma opacidade 51 → 27; opacidades entre os 10 primeiros 1,0 → 3,7.
- **Alarme:** no limiar sem rótulos, 29% de sensibilidade com 96% de especificidade.

## O atalho AP/PA, medido

Mesmo pipeline, com as duas incidências:

| | Só PA | AP + PA |
|---|---|---|
| AUC do VAE no teste | 0,757 | **0,788** (parece melhor) |
| AUC só entre exames AP | — | 0,66 |
| O score separa AP de PA entre os **normais**? | — | sim, AUC 0,70 |
| Ponto mais quente dentro da caixa | 25% (acaso 7%) | **12% (acaso 13%)** |
| CNN supervisionada | 0,83 | 0,94 |

O número geral sobe, mas a localização cai ao nível do acaso e o score passa a medir em parte a incidência. A CNN também ganha 0,11 com o mesmo atalho. **Número maior não é modelo melhor:** é preciso perguntar *o que* o modelo aprendeu.

## Comparação com o notebook (PneumoniaMNIST)

| | Notebook (PneumoniaMNIST, 64 px) | Pipeline (RSNA PA, 128 px) |
|---|---|---|
| Pacientes | crianças de 1 a 5 anos, 1 hospital | adultos |
| Normais no treino | 1.214 | 5.050 |
| VAE | 0,83 ± 0,01 | 0,757 ± 0,002 |
| Autoencoder comum | 0,83 ± 0,02 | 0,757 ± 0,002 |
| CNN | 0,92 | 0,83 |
| Erro ingênuo (σ = 0) | 0,62 | 0,69 |
| Heurística de detalhe | 0,91 | 0,55 |
| Fila (posição média) | 51 → 21 | 51 → 27 |

A AUC menor no RSNA não significa modelo pior: o problema é mais difícil (adultos, mais variação) e o PneumoniaMNIST tinha atalhos que inflavam todos os métodos.

## Limitações

- Mapa de calor grosseiro e que acende fora dos pulmões; restringir o score aos pulmões (segmentação) é o próximo passo natural.
- Um único hospital/fonte (NIH); o limiar precisaria ser recalibrado em outro lugar.
- Só PA no resultado principal: o modelo não serve para exames de leito (AP) sem treino específico.
- O "V" não melhorou a detecção em relação ao autoencoder comum.
- Não é um dispositivo médico.
