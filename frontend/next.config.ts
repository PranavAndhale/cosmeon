import type { NextConfig } from "next";

const isExport = process.env.NEXT_BUILD_MODE === "export";

const nextConfig: NextConfig = isExport
  ? {
      // PRODUCTION: Static HTML export for deployment (Render, etc.)
      output: "export",
    }
  : {
      // LOCAL DEV: Proxy /api/* to FastAPI backend on port 8000
      async rewrites() {
        return [
          {
            source: "/api/:path*",
            destination: "http://localhost:8000/api/:path*",
          },
        ];
      },
    };

export default nextConfig;
