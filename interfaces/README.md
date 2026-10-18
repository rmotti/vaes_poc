# Interfaces — demo da PoC

Site em Next.js (React + TypeScript) com três telas:

1. **Fila de triagem** — a lista de trabalho de um radiologista, reordenada pelo score do VAE. Mostra o exame original, a reconstrução "normal", o mapa de calor e as caixas do laudo.
2. **Por dentro do VAE** — gerador de raios-X a partir do código *z*, mapa do espaço latente, efeito da suavização do score (σ) e interpolação entre exames.
3. **Você × VAE** — a plateia ordena 8 exames por prioridade e compara com o VAE e com o laudo.

Os dados são reais: exames do conjunto de teste do RSNA, com scores, reconstruções e mapas pré-calculados pelos modelos da v2. As três abas usam o modelo treinado só com PA (`rsna_pa_v2`); o botão **AP + PA** da fila troca para o modelo treinado com as duas incidências (`rsna_ap_pa_v2`), que mostra o atalho da incidência. O que ainda é simulado está marcado na própria tela (o gerador mexe um eixo por vez, e o QR code é ilustrativo).

## Ver localmente

Precisa do Node.js 20 ou mais novo. Dentro de `interfaces/`:

```bash
npm install
npm run dev        # http://localhost:3000, recarrega ao editar
npm run build      # o mesmo build da Vercel (também confere os tipos)
```

Cada aba tem um endereço próprio: `/#queue`, `/#inside` e `/#game`.

## Organização

- `app/page.tsx` lê `public/data/data.json` no build e entrega para `components/Demo.tsx`, que monta as abas. A página sai pré-renderizada: o site não roda o modelo nem chama servidor.
- `components/queue/`, `components/inside/` e `components/game/`: os componentes de cada aba.
- `lib/types.ts`: o formato do `data.json`, que precisa acompanhar o `module_vae/export.py`.
- `app/globals.css`: o visual inteiro (cores em variáveis no `:root`).

## Publicar na Vercel

Importe o repositório na Vercel com **Root Directory = `interfaces`** e **Application Preset = Next.js**. Os comandos de instalação e de build ficam no padrão.

## Dados

`public/data/` é gerada a partir dos modelos treinados, na raiz do repositório:

```bash
make export   # uv run python -m module_vae.export
```

Ela contém as imagens (`pa/`, `ap_pa/`) e o `data.json` com scores, mapas e coordenadas. Ela é versionada, porque a Vercel monta o site a partir do repositório. Rode de novo depois de treinar outra vez: a pasta é apagada e recriada. Se o modelo AP + PA não existir, o botão fica desligado.

### Uso das imagens e atribuição

Os termos de uso do RSNA Pneumonia Detection Challenge permitem usar e redistribuir as imagens e as anotações, para fins acadêmicos ou não, desde que (1) ninguém tente identificar ou contatar os pacientes e (2) quem redistribui dê a atribuição abaixo. O rodapé do site já traz essa atribuição. O `data.json` não guarda o `patientId` dos exames: eles aparecem só como `EX-0000`.

- **NIH Chest X-ray Dataset**, de onde vêm as imagens: download em <https://nihcc.app.box.com/v/ChestXray-NIHCC>; o NIH Clinical Center é o fornecedor dos dados. Citação: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*. IEEE CVPR, pp. 3462–3471, 2017.
- **RSNA-STR Pneumonia Detection Challenge**, de onde vêm os rótulos e as caixas: download em <https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018>. Citação: George Shih, Carol C. Wu, Safwan S. Halabi et al. *Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia*. Radiology: Artificial Intelligence, 2019.

## Próximos passos

- Decoder exportado para ONNX e rodando no navegador, para o gerador combinar os 8 eixos ao mesmo tempo.
- Votação da turma em tempo real no "Você × VAE", com QR code de verdade.
