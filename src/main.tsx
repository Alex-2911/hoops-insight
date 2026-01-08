import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

const storedTheme = localStorage.getItem("theme");
const shouldUseDark = storedTheme === "dark";
document.documentElement.classList.toggle("dark", shouldUseDark);

createRoot(document.getElementById("root")!).render(<App />);
