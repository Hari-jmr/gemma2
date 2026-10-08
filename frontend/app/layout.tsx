import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "PDF RAG", description: "Ask questions about your PDF with source references." };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
