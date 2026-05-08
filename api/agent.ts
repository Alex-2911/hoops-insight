import { agentJsonHeaders, buildAgentResponse, methodNotAllowedBody } from "./agent_core.mjs";

type ApiRequest = {
  method?: string;
  body?: unknown;
};

type ApiResponse = {
  status: (statusCode: number) => ApiResponse;
  setHeader: (name: string, value: string) => unknown;
  send: (payload: string) => void;
  end?: () => void;
};

const sendJson = (response: ApiResponse, status: number, payload: Record<string, unknown>) => {
  response.status(status);
  for (const [name, value] of Object.entries(agentJsonHeaders)) {
    response.setHeader(name, value);
  }
  response.send(JSON.stringify(payload));
};

export default async function handler(request: ApiRequest, response: ApiResponse) {
  for (const [name, value] of Object.entries(agentJsonHeaders)) {
    response.setHeader(name, value);
  }

  if (request.method === "OPTIONS") {
    response.status(204);
    response.end?.();
    return;
  }

  if (request.method !== "POST") {
    response.setHeader("Allow", "POST, OPTIONS");
    return sendJson(response, 405, methodNotAllowedBody);
  }

  try {
    const result = await buildAgentResponse((request.body || {}) as Record<string, unknown>);
    return sendJson(response, result.status, result.body);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Agent backend failed.";
    return sendJson(response, 503, {
      answer: "",
      used_sources: [],
      warnings: [message],
      error: message,
    });
  }
}
