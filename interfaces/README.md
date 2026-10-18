# Interfaces — demo da PoC

Site estático (HTML, CSS e JavaScript, sem etapa de build) com três telas:

1. **Fila de triagem** — a lista de trabalho de um radiologista, reordenada pelo score do VAE. Mostra o exame original, a reconstrução "normal", o mapa de calor e as caixas do laudo.
2. **Por dentro do VAE** — gerador de raios-X a partir do código *z*, mapa do espaço latente, efeito da suavização do score (σ) e interpolação entre exames.
3. **Você × VAE** — a plateia ordena 8 exames por prioridade e compara com o VAE e com o laudo.

Os dados são reais: exames do conjunto de teste do RSNA, com scores, reconstruções e mapas pré-calculados pelos modelos da v2. As três abas usam o modelo treinado só com PA (`rsna_pa_v2`); o botão **AP + PA** da fila troca para o modelo treinado com as duas incidências (`rsna_ap_pa_v2`), que mostra o atalho da incidência. O que ainda é simulado está marcado na própria tela (o gerador mexe um eixo por vez, e o QR code é ilustrativo).

## Ver localmente

Abra `index.html` no navegador, ou sirva a pasta:

```bash
npx serve interfaces
```

Cada aba tem um endereço próprio: `#queue`, `#inside` e `#game`.

## Publicar na Vercel

Importe o repositório na Vercel com **Root Directory = `interfaces`** e **Framework Preset = Other**. Não há comando de build.

## Dados

`data/` é gerada a partir dos modelos treinados, na raiz do repositório:

```bash
make export   # uv run python -m module_vae.export
```

Ela contém as imagens (`data/pa/`, `data/ap_pa/`) e o `data.js` com scores, mapas e coordenadas. Rode de novo depois de treinar outra vez: a pasta é apagada e recriada. Se o modelo AP + PA não existir, o botão fica desligado.

A pasta **está no `.gitignore`**: as imagens vêm do RSNA Pneumonia Detection Challenge, e as regras da competição no Kaggle precisam ser conferidas antes de publicar essas imagens num repositório ou site público.

## Próximos passos

- Decoder exportado para ONNX e rodando no navegador, para o gerador combinar os 8 eixos ao mesmo tempo.
- Botão "AP + PA" na fila, com o modelo que mostra o atalho da incidência.
- Votação da turma em tempo real no "Você × VAE", com QR code de verdade.
