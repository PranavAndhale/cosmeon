import type { Metadata } from "next";
import "./globals.css";

// Fonts (Inter + Roboto Mono) are loaded via @import in globals.css
// This avoids next/font/google which is incompatible with static export

export const metadata: Metadata = {
  title: "COSMEON | Climate Prediction Software Engine",
  description: "High-performance Geospatial SaaS and Climate-Tech platform.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
