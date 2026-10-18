"use client";

import { useEffect, useMemo, useState } from "react";

import Segmented from "@/components/Segmented";
import Viewer from "@/components/queue/Viewer";
import Worklist from "@/components/queue/Worklist";
import { byScore, fmtScore } from "@/lib/format";
import type { Exam, VaeData } from "@/lib/types";

type Order = "arrival" | "vae";
const pt = (x: number) => x.toFixed(2).replace(".", ",");
const ARRIVAL_INTERVAL_MS = 650;

const isPositive = (e: Exam) => e.label === "opacidade";
const positivesInTop5 = (list: Exam[]) => list.slice(0, 5).filter(isPositive).length;
function meanPositivePosition(list: Exam[]) {
  const pos = list.flatMap((e, i) => (isPositive(e) ? [i + 1] : []));
  return pos.length ? (pos.reduce((a, b) => a + b, 0) / pos.length).toFixed(1) : "–";
}

// Cada modelo ("pa" e, se exportado, "ap_pa") tem a própria fila, sorteada do seu teste.
export default function QueueTab({ data }: { data: VaeData }) {
  const [runKey, setRunKey] = useState(data.main);
  const run = data.runs[runKey];
  const [arrived, setArrived] = useState(() => run.queue.map((e) => e.id)); // ordem de chegada
  const [order, setOrder] = useState<Order>("vae");
  const [reveal, setReveal] = useState(false);
  const [selectedId, setSelectedId] = useState(() => byScore(run.queue)[0].id);
  const [simulating, setSimulating] = useState(false);

  const byId = useMemo(() => Object.fromEntries(run.queue.map((e) => [e.id, e])), [run]);
  const arrivedExams = arrived.map((id) => byId[id]);
  const vaeOrder = byScore(arrivedExams);
  const shown = order === "vae" ? vaeOrder : arrivedExams;
  const multiView = run.model.views.length > 1;
  const test = run.model.test;

  // "Simular chegada": os exames entram um a um, e a fila se reordena a cada chegada.
  useEffect(() => {
    if (!simulating) return;
    const timer = setInterval(() => {
      setArrived((prev) => (prev.length < run.queue.length ? [...prev, run.queue[prev.length].id] : prev));
    }, ARRIVAL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [simulating, run]);

  useEffect(() => {
    if (simulating && arrived.length === run.queue.length) setSimulating(false);
  }, [simulating, arrived, run]);

  function switchRun(key: string) {
    const next = data.runs[key];
    setRunKey(key);
    setSimulating(false);
    setArrived(next.queue.map((e) => e.id));
    setSelectedId(byScore(next.queue)[0].id);
  }

  function simulate() {
    setArrived([]);
    setSimulating(true);
  }

  return (
    <>
      <div className="kpis">
        <div className="kpi">
          <span className="kpi-label">Exames na fila</span>
          <span className="kpi-value">{arrived.length}</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Opacidades entre os 5 primeiros</span>
          <span className="kpi-value">
            <span>{positivesInTop5(arrivedExams)}</span>
            <small>chegada</small>
            <span className="arrow">→</span>
            <span className="good">{positivesInTop5(vaeOrder)}</span>
            <small>VAE</small>
          </span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Posição média de uma opacidade</span>
          <span className="kpi-value">
            <span>{meanPositivePosition(arrivedExams)}</span>
            <small>chegada</small>
            <span className="arrow">→</span>
            <span className="good">{meanPositivePosition(vaeOrder)}</span>
            <small>VAE</small>
          </span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Limiar de alerta</span>
          <span className="kpi-value">{fmtScore(run.model.threshold)}</span>
          <span className="kpi-sub">percentil 95 dos exames normais</span>
        </div>
      </div>

      <div className="queue-layout">
        <div className="panel worklist">
          <div className="panel-head">
            <h2>Lista de trabalho</h2>
            <div className="controls">
              <Segmented<Order>
                label="Ordem da fila"
                value={order}
                onChange={setOrder}
                options={[
                  { value: "arrival", label: "Chegada" },
                  { value: "vae", label: "VAE" },
                ]}
              />
              <Segmented
                label="Modelo"
                value={runKey}
                onChange={switchRun}
                title={data.runs.ap_pa ? undefined : "Rode `make export` depois do treino com AP + PA"}
                options={[
                  { value: "pa", label: "Modelo só PA" },
                  { value: "ap_pa", label: "AP + PA", disabled: !data.runs.ap_pa },
                ]}
              />
              <button className="btn primary" onClick={simulate} disabled={simulating}>
                ▶ Simular chegada
              </button>
            </div>
          </div>
          <div className="table-head">
            <span>#</span>
            <span>Exame</span>
            <span>Chegada</span>
            <span>Score de anomalia</span>
            <span>Decisão</span>
            <label className="switch small">
              <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />
              <span>Laudo</span>
            </label>
          </div>
          {/* key: trocar de modelo monta uma lista nova, sem animar a troca */}
          <Worklist
            key={runKey}
            exams={shown}
            maxScore={Math.max(...run.queue.map((e) => e.score))}
            threshold={run.model.threshold}
            multiView={multiView}
            reveal={reveal}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
          {multiView && (
            <p className="note shortcut-note">
              <b>Atalho da incidência.</b> Este modelo foi treinado com AP e PA. Na incidência AP o paciente está
              deitado ou sentado, e a maioria das opacidades do RSNA é AP: o score sobe com a incidência, não só com a
              doença.
              {test?.auc_by_view.AP !== undefined &&
                ` No teste inteiro, a AUC geral é ${pt(test.roc_auc)}, mas dentro de cada incidência cai para ${pt(test.auc_by_view.AP)} (AP) e ${pt(test.auc_by_view.PA)} (PA): parte do acerto vem da incidência.`}
              {test?.auc_ap_vs_pa_normals !== undefined &&
                ` Só entre exames normais, o score separa AP de PA com AUC ${pt(test.auc_ap_vs_pa_normals)}.`}
            </p>
          )}
        </div>

        <Viewer exam={byId[selectedId]} reveal={reveal} />
      </div>
    </>
  );
}
