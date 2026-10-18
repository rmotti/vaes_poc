// Os dados vêm de data/data.js (window.VAE_DATA), gerado por `make export` a partir dos
// modelos treinados (module_vae/export.py). Nada aqui roda o modelo: tudo foi pré-calculado.
const D = window.VAE_DATA;
const $ = (sel) => document.querySelector(sel);
const fmtScore = (s) => (s * 1000).toFixed(2);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ================= abas =================
// Cada aba tem um endereço (#queue, #inside, #game), para dar para abrir direto nela.
function openTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
}
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    openTab(btn.dataset.tab);
    history.replaceState(null, "", `#${btn.dataset.tab}`);
  });
});
if (["queue", "inside", "game"].includes(location.hash.slice(1))) openTab(location.hash.slice(1));

// ================= 1. fila de triagem =================
// Cada modelo ("pa" e, se exportado, "ap_pa") tem a própria fila, sorteada do seu teste.
const queueState = {
  run: D.main,
  arrived: [], // ordem de chegada
  order: "vae",
  reveal: false,
  selected: null,
};
let run, byId, maxScore;
const rowsEl = $("#rows");
const rowEls = new Map();

function useRun(key) {
  clearInterval(simulation);
  $("#simulate").disabled = false;
  queueState.run = key;
  run = D.runs[key];
  byId = Object.fromEntries(run.queue.map((e) => [e.id, e]));
  maxScore = Math.max(...run.queue.map((e) => e.score));
  queueState.arrived = run.queue.map((e) => e.id);
  rowEls.clear(); // os ids se repetem entre modelos, mas são exames diferentes
  $("#kpi-threshold").textContent = fmtScore(run.model.threshold);
  $("#shortcut-note").hidden = run.model.views.length < 2;
  const byView = run.model.test?.auc_by_view;
  $("#shortcut-auc").textContent = byView?.AP
    ? ` No teste inteiro, a AUC fica em ${byView.AP.toFixed(2)} entre os AP e ${byView.PA.toFixed(2)} entre os PA.`
    : "";
  renderQueue({ animate: false });
  selectExam(orderedQueue()[0].id);
}

function orderedQueue() {
  const exams = queueState.arrived.map((id) => byId[id]);
  return queueState.order === "vae" ? [...exams].sort((a, b) => b.score - a.score) : exams;
}

function labelHtml(exam) {
  return queueState.reveal
    ? `<span class="label ${exam.label}">${exam.label}</span>`
    : `<span class="label hidden">oculto</span>`;
}

function createRow(exam) {
  const li = document.createElement("li");
  li.className = "row";
  const width = (exam.score / maxScore) * 100;
  const mark = (run.model.threshold / maxScore) * 100;
  const view = run.model.views.length > 1 ? `<em class="view-tag">${exam.view}</em>` : "";
  li.innerHTML = `
    <span class="pos"></span>
    <span class="exam"><img src="${exam.img}" alt=""><span>${exam.id}</span>${view}</span>
    <span class="time">${exam.arrival}</span>
    <span class="score-cell">
      <span class="bar"><i class="${exam.flagged ? "over" : ""}" style="width:${width}%"></i><b style="left:${mark}%"></b></span>
      <span>${fmtScore(exam.score)}</span>
    </span>
    <span><span class="pill ${exam.flagged ? "flag" : "ok"}">${exam.flagged ? "Revisar" : "Padrão"}</span></span>
    <span class="truth"></span>`;
  li.addEventListener("click", () => selectExam(exam.id));
  rowEls.set(exam.id, li);
  return li;
}

function renderQueue({ animate = true } = {}) {
  const list = orderedQueue();
  const before = new Map([...rowEls].map(([id, el]) => [id, el.isConnected ? el.getBoundingClientRect().top : null]));

  rowsEl.replaceChildren(
    ...list.map((exam, i) => {
      const el = rowEls.get(exam.id) || createRow(exam);
      el.querySelector(".pos").textContent = i + 1;
      el.querySelector(".truth").innerHTML = labelHtml(exam);
      el.classList.toggle("selected", exam.id === queueState.selected);
      return el;
    })
  );

  if (animate && !reduceMotion) {
    list.forEach((exam) => {
      const el = rowEls.get(exam.id);
      const first = before.get(exam.id);
      const last = el.getBoundingClientRect().top;
      if (first == null) {
        el.animate([{ opacity: 0, transform: "translateX(-14px)" }, { opacity: 1, transform: "none" }], {
          duration: 380,
          easing: "ease-out",
        });
      } else if (Math.abs(first - last) > 1) {
        el.animate([{ transform: `translateY(${first - last}px)` }, { transform: "none" }], {
          duration: 520,
          easing: "cubic-bezier(.2,.8,.2,1)",
        });
      }
    });
  }
  updateKpis();
}

function updateKpis() {
  const arrival = queueState.arrived.map((id) => byId[id]);
  const vae = [...arrival].sort((a, b) => b.score - a.score);
  const isPos = (e) => e.label === "opacidade";
  const top5 = (list) => list.slice(0, 5).filter(isPos).length;
  const meanPos = (list) => {
    const pos = list.map((e, i) => (isPos(e) ? i + 1 : null)).filter((p) => p !== null);
    return pos.length ? (pos.reduce((a, b) => a + b, 0) / pos.length).toFixed(1) : "–";
  };
  $("#kpi-count").textContent = arrival.length;
  $("#kpi-top-arrival").textContent = top5(arrival);
  $("#kpi-top-vae").textContent = top5(vae);
  $("#kpi-pos-arrival").textContent = meanPos(arrival);
  $("#kpi-pos-vae").textContent = meanPos(vae);
}

$("#order-toggle").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-order]");
  if (!btn) return;
  queueState.order = btn.dataset.order;
  $("#order-toggle").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
  renderQueue();
});

$("#reveal-labels").addEventListener("change", (ev) => {
  queueState.reveal = ev.target.checked;
  renderQueue({ animate: false });
  if (queueState.selected) selectExam(queueState.selected);
});

$("#model-toggle").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-run]");
  if (!btn || btn.disabled) return;
  $("#model-toggle").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
  useRun(btn.dataset.run);
});
if (!D.runs.ap_pa) {
  $("#model-toggle").classList.add("disabled");
  $("#model-toggle").title = "Rode `make export` depois do treino com AP + PA";
  $('#model-toggle [data-run="ap_pa"]').disabled = true;
}

let simulation = null;
$("#simulate").addEventListener("click", () => {
  clearInterval(simulation);
  const all = run.queue.map((e) => e.id);
  queueState.arrived = [];
  renderQueue({ animate: false });
  $("#simulate").disabled = true;
  simulation = setInterval(() => {
    queueState.arrived.push(all[queueState.arrived.length]);
    renderQueue();
    if (queueState.arrived.length === all.length) {
      clearInterval(simulation);
      $("#simulate").disabled = false;
    }
  }, 650);
});

// ---------- visualizador ----------
const viewer = $(".viewer");
const viewState = { mode: "original" };

function selectExam(id) {
  queueState.selected = id;
  const exam = byId[id];
  rowEls.forEach((el, key) => el.classList.toggle("selected", key === id));

  $("#viewer-title").textContent = exam.id;
  const pill = $("#viewer-decision");
  pill.className = `pill ${exam.flagged ? "flag" : "ok"}`;
  pill.textContent = exam.flagged ? "Revisar com prioridade" : "Dentro do padrão";

  $("#img-original").src = exam.img;
  $("#img-rec").src = exam.rec;
  $("#img-heat").src = exam.heat;
  drawBoxes($("#boxes"), exam.boxes);

  $("#fact-score").textContent = fmtScore(exam.score);
  $("#fact-percentile").textContent = `${Math.round(exam.percentile * 100)}% dos normais`;
  $("#fact-view").textContent = exam.view;
  $("#fact-label").innerHTML = queueState.reveal ? `<span class="label ${exam.label}">${exam.label}</span>` : "oculto";
  updateLayers();
}

function drawBoxes(container, boxes) {
  container.replaceChildren(
    ...boxes.map(([x, y, w, h]) => {
      const div = document.createElement("div");
      div.className = "box";
      Object.assign(div.style, { left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` });
      return div;
    })
  );
}

function updateLayers() {
  const swipe = Number($("#swipe").value);
  const rec = $("#img-rec");
  viewer.classList.toggle("swipe", viewState.mode === "swipe");
  rec.style.opacity = viewState.mode === "original" ? 0 : 1;
  rec.style.clipPath = viewState.mode === "swipe" ? `inset(0 0 0 ${swipe}%)` : "none";
  $("#swipe-line").style.left = `${swipe}%`;
  $("#stage-tag").textContent = { original: "original", rec: "reconstrução (o “normal” esperado)", swipe: "original | reconstrução" }[viewState.mode];

  const heatOn = $("#show-heat").checked;
  $("#img-heat").style.opacity = heatOn ? Number($("#heat-opacity").value) / 100 : 0;
  $("#boxes").style.opacity = $("#show-boxes").checked ? 1 : 0;
}

$("#view-mode").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-mode]");
  if (!btn) return;
  viewState.mode = btn.dataset.mode;
  $("#view-mode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
  updateLayers();
});
["#swipe", "#heat-opacity", "#show-heat", "#show-boxes"].forEach((sel) => $(sel).addEventListener("input", updateLayers));

useRun(D.main);

// ================= 2. por dentro do VAE =================
// Gerador: um sprite por dimensão de z, com 9 valores de −3 a +3
const genImage = $("#gen-image");
const values = D.latent.values;
function showLatent(dim, k) {
  genImage.style.backgroundImage = `url(${D.latent.dims[dim]})`;
  genImage.style.backgroundPosition = `${(k / (values.length - 1)) * 100}% 0`;
}
const sliders = [];
for (let d = 0; d < D.runs[D.main].model.latent_dim; d++) {
  const row = document.createElement("div");
  row.className = "slider-row";
  row.innerHTML = `<label>z${d + 1}</label><input type="range" min="0" max="${values.length - 1}" value="${(values.length - 1) / 2}"><output>0.0</output>`;
  const input = row.querySelector("input");
  input.addEventListener("input", () => {
    sliders.forEach((s, j) => {
      if (j !== d) {
        s.input.value = (values.length - 1) / 2;
        s.output.textContent = "0.0";
        s.row.classList.remove("active");
      }
    });
    row.classList.add("active");
    row.querySelector("output").textContent = values[input.value].toFixed(1);
    showLatent(d, Number(input.value));
  });
  sliders.push({ row, input, output: row.querySelector("output") });
  $("#gen-sliders").append(row);
}
showLatent(0, (values.length - 1) / 2);

// Mapa do espaço latente
const CLASSES = [
  { key: "normal", name: "normal", color: "var(--normal)" },
  { key: "outra", name: "anormal sem opacidade", color: "var(--other)" },
  { key: "opacidade", name: "opacidade", color: "var(--positive)" },
];
(function drawScatter() {
  const svg = $("#scatter");
  const pts = D.latent.points;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const W = 600, H = 330, pad = 16;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const sx = (x) => pad + ((x - x0) / (x1 - x0)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((y - y0) / (y1 - y0)) * (H - 2 * pad);
  const NS = "http://www.w3.org/2000/svg";
  const groups = {};
  CLASSES.forEach((c) => {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("fill", c.color);
    g.setAttribute("fill-opacity", c.key === "opacidade" ? 0.9 : 0.55);
    svg.append(g);
    groups[c.key] = g;
  });
  pts.forEach(([x, y, cls], k) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", sx(x).toFixed(1));
    c.setAttribute("cy", sy(y).toFixed(1));
    c.setAttribute("r", cls === "opacidade" ? 2.6 : 2.1);
    c.dataset.k = k;
    groups[cls].append(c);
  });

  const tooltip = $("#tooltip");
  const thumb = tooltip.querySelector(".thumb");
  const size = 96;
  svg.addEventListener("mousemove", (ev) => {
    const k = ev.target.dataset?.k;
    if (k === undefined) return (tooltip.style.display = "none");
    const cols = D.latent.atlas_cols;
    thumb.style.backgroundImage = `url(${D.latent.atlas})`;
    thumb.style.backgroundSize = `${cols * size}px ${cols * size}px`;
    thumb.style.backgroundPosition = `${-(k % cols) * size}px ${-Math.floor(k / cols) * size}px`;
    tooltip.querySelector("span").textContent = CLASSES.find((c) => c.key === pts[k][2]).name;
    const box = svg.parentElement.getBoundingClientRect();
    tooltip.style.left = `${Math.min(ev.clientX - box.left + 12, box.width - 112)}px`;
    tooltip.style.top = `${ev.clientY - box.top - 120}px`;
    tooltip.style.display = "block";
  });
  svg.addEventListener("mouseleave", () => (tooltip.style.display = "none"));

  CLASSES.forEach((c) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.innerHTML = `<i style="background:${c.color}"></i>${c.name}`;
    chip.addEventListener("click", () => {
      chip.classList.toggle("off");
      groups[c.key].style.display = chip.classList.contains("off") ? "none" : "";
    });
    $("#legend").append(chip);
  });
})();

// Suavização (σ)
(function sigmaCard() {
  const { example, steps, chosen } = D.sigma;
  $("#sigma-original").src = example.img;
  drawBoxes($("#sigma-boxes"), example.boxes);
  const range = $("#sigma-range");
  range.max = steps.length - 1;
  const chosenIdx = steps.findIndex((s) => s.sigma === chosen);

  const svg = $("#sigma-bars");
  const W = 300, H = 70;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const lo = Math.min(...steps.map((s) => s.auc_val)) - 0.02;
  const hi = Math.max(...steps.map((s) => s.auc_val));
  const bw = W / steps.length;
  svg.innerHTML = steps
    .map((s, i) => {
      const h = 8 + ((s.auc_val - lo) / (hi - lo)) * (H - 22);
      return `<rect data-i="${i}" x="${i * bw + 3}" y="${H - 12 - h}" width="${bw - 6}" height="${h}" rx="3"></rect>
              <text x="${i * bw + bw / 2}" y="${H - 1}" text-anchor="middle" font-size="9" fill="var(--faint)">${s.sigma}${i === chosenIdx ? "★" : ""}</text>`;
    })
    .join("");

  function show(i) {
    const s = steps[i];
    $("#sigma-heat").src = s.heat;
    $("#sigma-value").textContent = s.sigma;
    $("#sigma-auc").textContent = s.auc_val.toFixed(3);
    svg.querySelectorAll("rect").forEach((r) => {
      const j = Number(r.dataset.i);
      r.setAttribute("fill", j === i ? "var(--accent)" : j === chosenIdx ? "var(--muted)" : "var(--line)");
    });
  }
  range.addEventListener("input", () => show(Number(range.value)));
  show(0);
})();

// Interpolação
(function interpCard() {
  const { from, to, sprite, steps } = D.latent.interpolation;
  $("#interp-from").src = from.img;
  $("#interp-to").src = to.img;
  $("#interp-from-label").textContent = `${from.id} · ${from.label}`;
  $("#interp-to-label").textContent = `${to.id} · ${to.label}`;
  const frame = $("#interp-frame");
  frame.style.backgroundImage = `url(${sprite})`;
  frame.style.backgroundSize = `${steps * 100}% 100%`;
  const range = $("#interp-range");
  range.max = steps - 1;
  const show = () => (frame.style.backgroundPosition = `${(Number(range.value) / (steps - 1)) * 100}% 0`);
  range.addEventListener("input", show);
  show();
})();

// ================= 3. você × VAE =================
(function game() {
  const grid = $("#game-grid");
  let picks = [];
  const cards = D.game.map((exam) => {
    const btn = document.createElement("button");
    btn.className = "game-card";
    btn.innerHTML = `<img src="${exam.img}" alt="Exame ${exam.id}"><span class="order"></span>
      <span class="reveal"><span class="label ${exam.label}">${exam.label}</span><span>VAE: <b>${fmtScore(exam.score)}</b></span></span>`;
    btn.addEventListener("click", () => {
      if (btn.classList.contains("revealed")) return;
      picks = picks.includes(exam.id) ? picks.filter((id) => id !== exam.id) : [...picks, exam.id];
      refresh();
    });
    grid.append(btn);
    return { exam, btn };
  });

  function refresh() {
    cards.forEach(({ exam, btn }) => {
      const pos = picks.indexOf(exam.id);
      btn.classList.toggle("picked", pos >= 0);
      btn.querySelector(".order").textContent = pos >= 0 ? pos + 1 : "";
    });
    $("#game-reveal").disabled = picks.length !== cards.length;
  }

  $("#game-reset").addEventListener("click", () => {
    picks = [];
    cards.forEach(({ btn }) => btn.classList.remove("revealed"));
    $("#game-results").hidden = true;
    refresh();
  });

  $("#game-reveal").addEventListener("click", () => {
    cards.forEach(({ btn }) => btn.classList.add("revealed"));
    const byGameId = Object.fromEntries(D.game.map((e) => [e.id, e]));
    const yours = picks.map((id) => byGameId[id]);
    const vae = [...D.game].sort((a, b) => b.score - a.score);
    const nPos = D.game.filter((e) => e.label === "opacidade").length;
    const hits = (list) => list.slice(0, nPos).filter((e) => e.label === "opacidade").length;
    const list = (items) => items.map((e) => `<li class="${e.label}">${e.id} · ${e.label}</li>`).join("");
    const res = $("#game-results");
    res.innerHTML = `
      <h4>Opacidades nas ${nPos} primeiras posições</h4>
      <div class="score-line"><span>Você</span><b>${hits(yours)} de ${nPos}</b></div>
      <div class="score-line"><span>VAE</span><b>${hits(vae)} de ${nPos}</b></div>
      <div class="order-lists"><div>Sua ordem<ol>${list(yours)}</ol></div><div>Ordem do VAE<ol>${list(vae)}</ol></div></div>`;
    res.hidden = false;
  });
  refresh();
})();

// QR ilustrativo (não é um QR válido): na versão final ele aponta para a página de votação.
(function fakeQr() {
  const n = 25;
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const finder = (r, c) => [[0, 0], [0, n - 7], [n - 7, 0]].some(([fr, fc]) => r >= fr && r < fr + 7 && c >= fc && c < fc + 7);
  const finderOn = (r, c) => {
    const [fr, fc] = [[0, 0], [0, n - 7], [n - 7, 0]].find(([a, b]) => r >= a && r < a + 7 && c >= b && c < b + 7);
    const [y, x] = [r - fr, c - fc];
    return y === 0 || y === 6 || x === 0 || x === 6 || (y >= 2 && y <= 4 && x >= 2 && x <= 4);
  };
  let cells = "";
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const on = finder(r, c) ? finderOn(r, c) : rand() < 0.45;
      if (on) cells += `<rect x="${c}" y="${r}" width="1" height="1"/>`;
    }
  }
  $("#qr").innerHTML = `<svg viewBox="0 0 ${n} ${n}" shape-rendering="crispEdges" fill="#0b0e13">${cells}</svg>`;
})();
