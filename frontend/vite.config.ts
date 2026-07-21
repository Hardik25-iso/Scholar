import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// No proxy: the browser (Windows side) calls the FastAPI server directly at
// localhost:8000 — CORS for localhost:5173 is already allowed by the backend.
export default defineConfig({
  plugins: [react()],
});
