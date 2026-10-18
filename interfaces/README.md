# Interfaces — demo da PoC

Experiência em Next.js (React + TypeScript) com quatro áreas:

1. **Entenda o VAE** — percurso de cinco etapas: problema, arquitetura, distribuição latente, score e aplicação na fila. Inclui a ilustração interativa da reparametrização e um exemplo real de reconstrução.
2. **Fila de triagem** — comparação simultânea da chegada e do ranking do VAE, lista de trabalho e análise de cada exame. O diálogo de análise apresenta original, reconstrução e mapa lado a lado, com fórmula e explicação do score; a imagem ampliada tem zoom e navegação por teclado.
3. **Por dentro do VAE** — gerador de raios-X a partir do código *z*, mapa do espaço latente, efeito da suavização do score (σ) e interpolação entre exames.
4. **Você × VAE** — a plateia ordena 8 exames por prioridade e compara com o VAE e com o laudo.

Os exames são reais: dados do conjunto de teste do RSNA, com scores, reconstruções e mapas pré-calculados pelos modelos da v2. O modelo principal é o treinado só com PA (`rsna_pa_v2`); o botão **AP + PA** da fila troca para o modelo treinado com as duas incidências (`rsna_ap_pa_v2`), que mostra o atalho da incidência. O gerador mexe um eixo por vez, consultando sprites exportados. A distribuição interativa do percurso é uma ilustração matemática explicitamente identificada, sem representar o código de um exame. O desafio é local, sem votação em tempo real.

A fila tem busca por código, filtro de alertas, simulação com pausa/continuação e navegação entre exames. As caixas das anotações só aparecem quando os laudos estão revelados. No celular, selecionar um exame abre a imagem num diálogo dedicado. Setas esquerda/direita navegam quando o foco está no visualizador ou diálogo; no quadro com zoom, elas rolam a imagem. Escape fecha o diálogo e restaura o foco ao controle que o abriu. O zoom amplia os pixels existentes, sem aumentar a resolução.

A simulação oferece três velocidades, progresso, avanço de uma chegada por vez e acompanhamento automático da última chegada. Selecionar outro exame interrompe esse acompanhamento. As filas animam a reordenação e ligam as duas posições do exame acompanhado. O modo apresentação coloca a comparação ao lado da imagem e oculta a lista detalhada. Na bancada operacional em desktop, a lista e o visualizador têm topo e base alinhados; a lista rola por dentro da altura disponível.

Ao revelar os laudos, a análise identifica acertos, falsos alertas e falsos negativos no limiar da execução. Quando há um normal à frente de uma opacidade, a comparação oferece uma explicação com os códigos e scores reais do par. O limiar altera alertas, sem alterar a ordem por score.

Veja o [guia detalhado do projeto](../docs/07-guia-do-projeto.md) para entender o experimento, o cálculo do score e os limites dos resultados.

> ⚠️ 18/10/2026: os dados foram regenerados com os modelos da v3 (`rsna_pa_v3` e `rsna_ap_pa_v3`): normalização de intensidade e mapas restritos à máscara dos pulmões. A imagem "original" mostrada é o exame normalizado, como o modelo o vê. Ver [docs/09-resultados-v3.md](../docs/09-resultados-v3.md).

## Ver localmente

Precisa do Node.js 20 ou mais novo. Dentro de `interfaces/`:

```bash
npm install
npm run dev        # http://localhost:3000, recarrega ao editar
npm run build      # o mesmo build da Vercel (também confere os tipos)
npm run typecheck  # TypeScript sem gerar build
npm run test:ui    # testes no Chrome, após o build; desktop e celular
npm run validate:data # contrato da exportação e existência dos arquivos referenciados
npm run test:data  # testes do contrato, incluindo parâmetros e contagens diferentes
```

O percurso é a entrada padrão em `/`. Cada área tem um endereço próprio: `/#learn`, `/#queue`, `/#inside` e `/#game`.

## Organização

- `app/page.tsx` lê `public/data/data.json` no build e entrega para `components/Demo.tsx`, que monta as abas. A página sai pré-renderizada: o site não roda o modelo nem chama servidor.
- `components/learn/`, `components/queue/`, `components/inside/` e `components/game/`: os componentes de cada área.
- `lib/types.ts`: o formato do `data.json`, que precisa acompanhar o `module_vae/export.py`.
- `app/globals.css`: estilos base; `app/workspace.css`: tema e estação de trabalho; `app/experience.css`: percurso, comparação e diálogo de análise.
- `tests/demo.spec.ts` e `tests/experience.spec.ts`: fluxos de busca, simulação, modelos, laudos, navegação, jogo, aprendizado, comparação, zoom e layout.
- `tests/simulation.spec.ts`: velocidade, chegada manual, acompanhamento, modo apresentação e alinhamento da bancada operacional.
- `scripts/data-contract.mjs`: contrato da exportação, sem scores, quantidades ou versão fixos.

## Atualizar os resultados do modelo

Ao receber uma nova execução, atualize a exportação completa: `data.json`, originais, reconstruções, mapas e sprites. Trocar somente os pesos ou `reports/` não atualiza o site, pois o navegador usa `public/data/`. A geração continua sendo `make export`, com dados e pesos compatíveis disponíveis no ambiente de treino.

Depois de puxar os arquivos atualizados, dentro de `interfaces/`:

```bash
npm run validate:data
npm run test:data
npm run build
npm run test:ui
```

`dev` e `build` validam os dados automaticamente antes de iniciar. A validação confere tipos, scores finitos, métricas em escala de 0 a 1, alertas coerentes com o limiar, códigos únicos, dimensão latente versus sprites e os arquivos referenciados. Ela não substitui a avaliação experimental do novo modelo.

As próximas exportações incluem `image_size`, `original_image_size` e, quando presentes nos metadados da execução, `version`, `epochs` e `threshold_rule`. Exportações anteriores continuam aceitas. Campos ausentes aparecem como não informados; a interface não assume uma versão ou resolução antiga. Execução e data da exportação ficam visíveis na fila.

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
