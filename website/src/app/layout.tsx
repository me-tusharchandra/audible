import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { CursorTrail } from "@/components/CursorTrail";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Audible — An always-on voice assistant that learns when not to listen",
  description:
    "A self-improving OpenEnv environment that teaches a 24.6M-parameter mobileBERT to gate ambient utterances per user. 4× false-wake reduction in one round of adaptive curriculum.",
  metadataBase: new URL("https://audible.example"),
  openGraph: {
    title: "Audible — Self-improving ambient voice gating",
    description:
      "OpenEnv environment + mobileBERT classifier that learns when not to act on ambient utterances. Built for Meta OpenEnv Hackathon 2026.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} antialiased`}
    >
      <body className="min-h-screen bg-zinc-950 text-zinc-50 font-sans">
        <CursorTrail />
        {children}
      </body>
    </html>
  );
}
