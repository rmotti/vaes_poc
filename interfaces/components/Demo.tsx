"use client";

import { useEffect, useState } from "react";

import GameTab from "@/components/game/GameTab";
import InsideTab from "@/components/inside/InsideTab";
import QueueTab from "@/components/queue/QueueTab";
import type { VaeData } from "@/lib/types";

const TABS = [
  { key: "queue", label: "1 · Fila de triagem" },
  { key: "inside", label: "2 · Por dentro do VAE" },
  { key: "game", label: "3 · Você × VAE" },
] as const;
type TabKey = (typeof TABS)[number]["key"];

export default function Demo({ data }: { data: VaeData }) {
  const [tab, setTab] = useState<TabKey>("queue");

  // Cada aba tem um endereço (#queue, #inside, #game), para dar para abrir direto nela.
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    const match = TABS.find((t) => t.key === hash);
    if (match) setTab(match.key);
  }, []);

  function open(key: TabKey) {
    setTab(key);
    history.replaceState(null, "", `#${key}`);
  }

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <div>
            <h1>Triagem de raio-X com VAE</h1>
            <p>Prova de conceito · RSNA Pneumonia · exames reais do conjunto de teste</p>
          </div>
        </div>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button key={t.key} role="tab" aria-selected={tab === t.key} onClick={() => open(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* As três abas ficam montadas: trocar de aba não perde o estado de cada uma. */}
      <main>
        <section className={`tab ${tab === "queue" ? "active" : ""}`} role="tabpanel">
          <QueueTab data={data} />
        </section>
        <section className={`tab ${tab === "inside" ? "active" : ""}`} role="tabpanel">
          <InsideTab data={data} />
        </section>
        <section className={`tab ${tab === "game" ? "active" : ""}`} role="tabpanel">
          <GameTab exams={data.game} />
        </section>
      </main>

      <footer>
        <p>Prova de conceito acadêmica · não é um dispositivo médico</p>
        <p>
          Imagens: <a href="https://nihcc.app.box.com/v/ChestXray-NIHCC">NIH Chest X-ray Dataset</a>, fornecido pelo
          NIH Clinical Center (Wang et al., <i>ChestX-ray8</i>, IEEE CVPR 2017, pp. 3462–3471). Rótulos e caixas:{" "}
          <a href="https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018">
            RSNA Pneumonia Detection Challenge
          </a>{" "}
          (Shih et al.,{" "}
          <i>
            Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible
            Pneumonia
          </i>
          , Radiology: AI, 2019).
        </p>
      </footer>
    </>
  );
}
