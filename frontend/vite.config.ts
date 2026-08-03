import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Path aliases mirror the src/ folder structure so imports read like
// "@/services/predictionService" instead of "../../../services/predictionService".
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@components": path.resolve(__dirname, "./src/components"),
      "@pages": path.resolve(__dirname, "./src/pages"),
      "@hooks": path.resolve(__dirname, "./src/hooks"),
      "@services": path.resolve(__dirname, "./src/services"),
      "@types": path.resolve(__dirname, "./src/types"),
      "@context": path.resolve(__dirname, "./src/context"),
      "@utils": path.resolve(__dirname, "./src/utils"),
      "@constants": path.resolve(__dirname, "./src/constants"),
      "@animations": path.resolve(__dirname, "./src/animations"),
    },
  },
  server: {
    port: 5173,
  },
  build: {
    // Manual chunking keeps vendor code (React, charting, animation libs)
    // in a separate cache-friendly bundle from app code that changes often.
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom", "react-router-dom"],
          query: ["@tanstack/react-query"],
          motion: ["framer-motion"],
        },
      },
    },
  },
});
