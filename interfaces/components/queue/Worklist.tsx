"use client";

import { useLayoutEffect, useRef } from "react";

import LabelTag from "@/components/LabelTag";
import { fmtScore } from "@/lib/format";
import type { Exam } from "@/lib/types";

interface Props {
  exams: Exam[];
  maxScore: number;
  threshold: number;
  multiView: boolean;
  reveal: boolean;
  selectedId: string;
  onSelect: (id: string) => void;
}

/**
 * A lista de trabalho. Quando a ordem muda, cada linha desliza da posição antiga até a
 * nova (técnica FLIP: mede antes e depois e anima a diferença).
 */
export default function Worklist({ exams, maxScore, threshold, multiView, reveal, selectedId, onSelect }: Props) {
  const rows = useRef(new Map<string, HTMLLIElement>());
  const lastTop = useRef(new Map<string, number>());
  const mounted = useRef(false);
  const order = exams.map((e) => e.id).join();

  useLayoutEffect(() => {
    const animate = mounted.current && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const tops = new Map<string, number>();
    rows.current.forEach((el, id) => {
      const top = el.offsetTop;
      tops.set(id, top);
      if (!animate) return;
      const before = lastTop.current.get(id);
      if (before === undefined) {
        el.animate([{ opacity: 0, transform: "translateX(-14px)" }, { opacity: 1, transform: "none" }], {
          duration: 380,
          easing: "ease-out",
        });
      } else if (Math.abs(before - top) > 1) {
        el.animate([{ transform: `translateY(${before - top}px)` }, { transform: "none" }], {
          duration: 520,
          easing: "cubic-bezier(.2,.8,.2,1)",
        });
      }
    });
    lastTop.current = tops;
    mounted.current = true;
  }, [order]);

  const mark = (threshold / maxScore) * 100;
  return (
    <ol className="rows">
      {exams.map((exam, i) => (
        <li
          key={exam.id}
          ref={(el) => {
            if (el) rows.current.set(exam.id, el);
            else rows.current.delete(exam.id);
          }}
          className={`row ${exam.id === selectedId ? "selected" : ""}`}
          onClick={() => onSelect(exam.id)}
        >
          <span className="pos">{i + 1}</span>
          <span className="exam">
            <img src={exam.img} alt="" />
            <span>{exam.id}</span>
            {multiView && <em className="view-tag">{exam.view}</em>}
          </span>
          <span className="time">{exam.arrival}</span>
          <span className="score-cell">
            <span className="bar">
              <i className={exam.flagged ? "over" : ""} style={{ width: `${(exam.score / maxScore) * 100}%` }} />
              <b style={{ left: `${mark}%` }} />
            </span>
            <span>{fmtScore(exam.score)}</span>
          </span>
          <span>
            <span className={`pill ${exam.flagged ? "flag" : "ok"}`}>{exam.flagged ? "Revisar" : "Padrão"}</span>
          </span>
          <span className="truth">
            <LabelTag label={exam.label} hidden={!reveal} />
          </span>
        </li>
      ))}
    </ol>
  );
}
