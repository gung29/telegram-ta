import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
    },
    server: {
      port: 5173,
      host: "0.0.0.0",
    },
    build: {
      outDir: "dist",
      assetsDir: "assets",
    },
    define: {
      "process.env.GEMINI_API_KEY": JSON.stringify(env.VITE_GEMINI_API_KEY || env.GEMINI_API_KEY || ""),
    },
  };
});
