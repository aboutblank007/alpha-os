import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
// import { AppShell } from "@/components/AppShell"; // We might want AppShell here or in specific pages?
// Actually, AppShell contains the Dock, so it should probably be in layout if it persists everywhere.
// But the Landing page usually doesn't have the Dock. 
// So let's keep RootLayout clean and put AppShell in dashboard layout or conditionally.
// However, the prompt implies "AlphaOS | Institutional Trading" generic layout.
import { AppShell } from "@/components/AppShell"; // Using AppShell globally for now as per previous logic, or maybe conditional?
// Actually, let's look at the file. The previous version didn't import AppShell in layout.tsx.
// It seems the user wants a "Core Layout Rebuild".
// Let's wrapping everything in a basic div for now, and let pages decide, OR wrap in AppShell if it handles logic.
// But `AppShell` in `src/components/AppShell.tsx` (viewed in step 119) renders QuantumDock and CommandHeader.
// The landing page shouldn't have those.
// So, for RootLayout, we just provide the Font and CSS.
// The Dashboard page or a Dashboard Layout should use AppShell.
// But wait, the standard Next.js App Router pattern is to have a layout.tsx in src/app/dashboard/layout.tsx for that.
// Since I don't see a src/app/dashboard/layout.tsx, I will create one or simple wrap Dashboard page.
// However, the immediate fix is REMOVING SkipNav.

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
    title: "AlphaOS | Institutional Trading",
    description: "Enterprise-grade quantitative trading platform.",
};

export const viewport: Viewport = {
    width: "device-width",
    initialScale: 1,
    maximumScale: 1,
    themeColor: "#050505", // Updated to new base color
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className={`dark ${inter.variable}`}>
            <body className="antialiased bg-bg-base text-text-primary font-sans min-h-screen flex flex-col">
                {children}
            </body>
        </html>
    );
}
