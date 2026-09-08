import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API runs separately; proxying keeps the app same-origin in dev so
    // uploads and polling need no CORS configuration.
    proxy: {
      "/runs": "http://127.0.0.1:8000",
      "/subjects": "http://127.0.0.1:8000",
      "/fixtures": "http://127.0.0.1:8000",
      "/uploads": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
