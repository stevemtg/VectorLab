import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vector Lab — Activation Steering Workspace",
  description: "Explore representation engineering with your local models: real residual extraction, additive activation steering, and streaming inference.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
