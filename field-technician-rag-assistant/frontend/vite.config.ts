import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The browser only ever talks to /api; Vite proxies it to the backend (compose: http://backend:8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000", changeOrigin: true } },
  },
});
