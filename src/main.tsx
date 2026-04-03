import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.tsx";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./index.css";

const storedTheme = localStorage.getItem("theme");
const shouldUseDark = storedTheme === "dark";
document.documentElement.classList.toggle("dark", shouldUseDark);

const renderBootError = (message: string) => {
  document.body.innerHTML = `
    <div style="min-height:100vh;display:grid;place-items:center;background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;padding:24px;">
      <div style="max-width:720px;border:1px solid #334155;border-radius:12px;padding:20px;background:#111827;">
        <h1 style="margin:0 0 12px;font-size:24px;">Frontend render error</h1>
        <p style="margin:0;line-height:1.5;">${message}</p>
        <p style="margin-top:12px;opacity:.8;">Failure occurred before dashboard data fetch started.</p>
      </div>
    </div>
  `;
};

const rootElement = document.getElementById("root");

if (!rootElement) {
  renderBootError("Root element #root was not found in index.html.");
  throw new Error("Root element #root was not found in index.html.");
}

try {
  createRoot(rootElement).render(
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </BrowserRouter>,
  );
} catch (error) {
  const message = error instanceof Error ? error.message : "Unknown boot error.";
  renderBootError(message);
  throw error;
}
