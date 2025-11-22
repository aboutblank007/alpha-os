import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppShell } from "@/components/AppShell";
import { SkipNav } from "@/components/ui/SkipNav";

export const metadata: Metadata = {
  title: "AlphaOS | Professional Trading Interface",
  description: "Advanced quantitative trading journal and analytics platform.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#0f172a",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-background text-foreground">
        <SkipNav />
        <AppShell>
          {children}
        </AppShell>
      </body>
    </html>
  );
}
