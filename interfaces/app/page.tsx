import { readFile } from "node:fs/promises";
import path from "node:path";

import Demo from "@/components/Demo";
import type { VaeData } from "@/lib/types";

// Os dados são lidos no build: a página sai pronta, sem nenhuma chamada ao servidor.
async function loadData(): Promise<VaeData | null> {
  try {
    const file = path.join(process.cwd(), "public", "data", "data.json");
    return JSON.parse(await readFile(file, "utf-8"));
  } catch {
    return null;
  }
}

export default async function Page() {
  const data = await loadData();
  if (!data) {
    return (
      <main className="missing">
        <h1>Sem dados da demo</h1>
        <p>
          Rode <code>make export</code> na raiz do repositório para gerar <code>interfaces/public/data</code>.
        </p>
      </main>
    );
  }
  return <Demo data={data} />;
}
