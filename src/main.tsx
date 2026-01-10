import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.tsx";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./index.css";

const storedTheme = localStorage.getItem("theme");
const shouldUseDark = storedTheme === "dark";
document.documentElement.classList.toggle("dark", shouldUseDark);

createRoot(document.getElementById("root")!).render(
  <BrowserRouter basename={import.meta.env.BASE_URL}>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </BrowserRouter>,
);
