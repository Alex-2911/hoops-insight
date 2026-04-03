import { Route, Routes } from "react-router-dom";
import Layout from "./components/layout/Layout";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

export default function App() {
  return (
    <Layout>
      <div className="sr-only" aria-live="polite">
        APP LOADED
      </div>
      <Routes>
        <Route index element={<Index />} />
        <Route path="/" element={<Index />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Layout>
  );
}
