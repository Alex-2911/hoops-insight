import { defineConfig, loadEnv, type PreviewServer, type ViteDevServer } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import type { IncomingMessage, ServerResponse } from "http";
import { componentTagger } from "lovable-tagger";
import { agentJsonHeaders, buildAgentResponse, methodNotAllowedBody } from "./api/agent_core.mjs";

const sendAgentJson = (res: ServerResponse, status: number, payload: Record<string, unknown>) => {
  res.statusCode = status;
  for (const [name, value] of Object.entries(agentJsonHeaders)) {
    res.setHeader(name, value);
  }
  res.end(JSON.stringify(payload));
};

const readJsonBody = async (req: IncomingMessage) => {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw.trim() ? JSON.parse(raw) : {};
};

const registerAgentMiddleware = (server: ViteDevServer | PreviewServer) => {
  server.middlewares.use("/api/agent", async (req, res) => {
    for (const [name, value] of Object.entries(agentJsonHeaders)) {
      res.setHeader(name, value);
    }

    if (req.method === "OPTIONS") {
      res.statusCode = 204;
      res.end();
      return;
    }

    if (req.method !== "POST") {
      res.setHeader("Allow", "POST, OPTIONS");
      sendAgentJson(res, 405, methodNotAllowedBody);
      return;
    }

    try {
      const body = await readJsonBody(req);
      const result = await buildAgentResponse(body);
      sendAgentJson(res, result.status, result.body);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Agent backend failed.";
      sendAgentJson(res, 503, {
        answer: "",
        used_sources: [],
        warnings: [message],
        error: message,
      });
    }
  });
};

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  for (const key of ["HOOPS_AGENT_API_URL", "OPENAI_API_KEY", "HOOPS_AGENT_MODEL", "HOOPS_AGENT_ALLOWED_ORIGIN"]) {
    if (process.env[key] === undefined && env[key] !== undefined) {
      process.env[key] = env[key];
    }
  }

  return {
    base: process.env.BASE_URL ?? "/hoops-insight/",
    server: {
      host: "::",
      port: 5173,
    },
    plugins: [
      react(),
      {
        name: "hoops-agent-api",
        configureServer: registerAgentMiddleware,
        configurePreviewServer: registerAgentMiddleware,
      },
      mode === "development" && componentTagger(),
    ].filter(Boolean),
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
});
