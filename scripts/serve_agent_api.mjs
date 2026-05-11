import http from "http";
import "./load_local_env.mjs";
import { agentJsonHeaders, buildAgentResponse, methodNotAllowedBody } from "../api/agent_core.mjs";

const port = Number.parseInt(process.env.PORT || process.env.AGENT_API_PORT || "5173", 10);

const sendJson = (response, status, payload) => {
  response.writeHead(status, agentJsonHeaders);
  response.end(JSON.stringify(payload));
};

const readBody = async (request) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw.trim()) return {};
  return JSON.parse(raw);
};

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || `localhost:${port}`}`);

  if (url.pathname !== "/api/agent") {
    return sendJson(response, 404, {
      answer: "",
      used_sources: [],
      warnings: ["Only /api/agent is served by this local test server."],
      error: "Not found.",
    });
  }

  if (request.method === "OPTIONS") {
    response.writeHead(204, agentJsonHeaders);
    response.end();
    return;
  }

  if (request.method !== "POST") {
    response.writeHead(405, { ...agentJsonHeaders, Allow: "POST, OPTIONS" });
    response.end(JSON.stringify(methodNotAllowedBody));
    return;
  }

  try {
    const body = await readBody(request);
    const result = await buildAgentResponse(body);
    sendJson(response, result.status, result.body);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Agent backend failed.";
    sendJson(response, 503, {
      answer: "",
      used_sources: [],
      warnings: [message],
      error: message,
    });
  }
});

server.listen(port, () => {
  console.log(`Hoops agent API listening on http://localhost:${port}/api/agent`);
  console.log(`Test with: curl -i -X POST http://localhost:${port}/api/agent -H "Content-Type: application/json" -d '{"question":"who is playing today?","capability":"read_only","context":{}}'`);
});
