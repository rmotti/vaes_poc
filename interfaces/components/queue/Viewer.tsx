"use client";

import { useState } from "react";

import Boxes from "@/components/Boxes";
import LabelTag from "@/components/LabelTag";
import Segmented from "@/components/Segmented";
import { fmtScore } from "@/lib/format";
import type { Exam } from "@/lib/types";

type Mode = "original" | "rec" | "swipe";
const MODE_TAG: Record<Mode, string> = {
  original: "original",
  rec: "reconstrução (o “normal” esperado)",
  swipe: "original | reconstrução",
};

/** O exame selecionado na fila: original, reconstrução, mapa de calor e caixas do laudo. */
export default function Viewer({ exam, reveal }: { exam: Exam; reveal: boolean }) {
  const [mode, setMode] = useState<Mode>("original");
  const [swipe, setSwipe] = useState(50);
  const [heatOn, setHeatOn] = useState(true);
  const [heatOpacity, setHeatOpacity] = useState(60);
  const [boxesOn, setBoxesOn] = useState(false);

  return (
    <div className={`panel viewer ${mode === "swipe" ? "swipe" : ""}`}>
      <div className="panel-head">
        <h2>{exam.id}</h2>
        <span className={`pill ${exam.flagged ? "flag" : "ok"}`}>
          {exam.flagged ? "Revisar com prioridade" : "Dentro do padrão"}
        </span>
      </div>
      <div className="stage">
        <img src={exam.img} alt="Raio-X original" />
        <img
          src={exam.rec}
          className="layer pixel"
          alt=""
          style={{
            opacity: mode === "original" ? 0 : 1,
            clipPath: mode === "swipe" ? `inset(0 0 0 ${swipe}%)` : "none",
          }}
        />
        <img src={exam.heat} className="layer pixel" alt="" style={{ opacity: heatOn ? heatOpacity / 100 : 0 }} />
        <Boxes boxes={exam.boxes} visible={boxesOn} />
        <div className="swipe-line" style={{ left: `${swipe}%` }} />
        <span className="stage-tag">{MODE_TAG[mode]}</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={swipe}
        onChange={(e) => setSwipe(Number(e.target.value))}
        className="swipe-range"
        aria-label="Comparar original e reconstrução"
      />
      <div className="viewer-controls">
        <Segmented<Mode>
          label="Imagem"
          value={mode}
          onChange={setMode}
          options={[
            { value: "original", label: "Original" },
            { value: "rec", label: "Reconstrução" },
            { value: "swipe", label: "Comparar" },
          ]}
        />
        <label className="switch">
          <input type="checkbox" checked={heatOn} onChange={(e) => setHeatOn(e.target.checked)} />
          <span>Mapa de calor</span>
        </label>
        <input
          type="range"
          min={0}
          max={100}
          value={heatOpacity}
          onChange={(e) => setHeatOpacity(Number(e.target.value))}
          aria-label="Opacidade do mapa"
        />
        <label className="switch">
          <input type="checkbox" checked={boxesOn} onChange={(e) => setBoxesOn(e.target.checked)} />
          <span>Caixas do laudo</span>
        </label>
      </div>
      <dl className="facts">
        <div>
          <dt>Score</dt>
          <dd>{fmtScore(exam.score)}</dd>
        </div>
        <div>
          <dt>Mais anômalo que</dt>
          <dd>{Math.round(exam.percentile * 100)}% dos normais</dd>
        </div>
        <div>
          <dt>Incidência</dt>
          <dd>{exam.view}</dd>
        </div>
        <div>
          <dt>Laudo</dt>
          <dd>{reveal ? <LabelTag label={exam.label} /> : "oculto"}</dd>
        </div>
      </dl>
      <p className="hint">
        A <b>reconstrução</b> é o tórax “normal” que o VAE espera ver. O <b>mapa de calor</b> marca onde o exame foge
        disso.
      </p>
    </div>
  );
}
