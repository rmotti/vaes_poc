import { useMemo } from "react";

const N = 25;
const FINDERS = [
  [0, 0],
  [0, N - 7],
  [N - 7, 0],
];

/** QR ilustrativo (não é um QR válido): na versão final ele aponta para a página de votação. */
export default function FakeQr() {
  const cells = useMemo(() => {
    let seed = 7;
    const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const out: [number, number][] = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const finder = FINDERS.find(([fr, fc]) => r >= fr && r < fr + 7 && c >= fc && c < fc + 7);
        let on;
        if (finder) {
          const [y, x] = [r - finder[0], c - finder[1]];
          on = y === 0 || y === 6 || x === 0 || x === 6 || (y >= 2 && y <= 4 && x >= 2 && x <= 4);
        } else {
          on = rand() < 0.45;
        }
        if (on) out.push([r, c]);
      }
    }
    return out;
  }, []);

  return (
    <div className="qr" aria-hidden="true">
      <svg viewBox={`0 0 ${N} ${N}`} shapeRendering="crispEdges" fill="#0b0e13">
        {cells.map(([r, c]) => (
          <rect key={`${r}-${c}`} x={c} y={r} width={1} height={1} />
        ))}
      </svg>
    </div>
  );
}
