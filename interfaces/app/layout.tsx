import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Triagem com VAE",
  description: "Prova de conceito: triagem de raio-X de tórax com um autoencoder variacional",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
