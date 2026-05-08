type AgentMessage = {
  role?: string;
  content?: string;
};

type AgentRequest = {
  message?: string;
  messages?: AgentMessage[];
  context?: Record<string, unknown>;
};

type ApiRequest = {
  method?: string;
  body?: unknown;
};

type ApiResponse = {
  status: (statusCode: number) => ApiResponse;
  setHeader: (name: string, value: string) => unknown;
  send: (payload: string) => void;
};

type OpenAIContent = { text?: string };
type OpenAIOutputItem = { content?: OpenAIContent[] };
type OpenAIResponse = {
  output_text?: string;
  output?: OpenAIOutputItem[];
  error?: { message?: string };
};

type AgentBackendResponse = {
  reply?: string;
  message?: string;
  content?: string;
  error?: string;
};

const json = (response: ApiResponse, status: number, payload: Record<string, unknown>) => {
  response.status(status);
  response.setHeader("Content-Type", "application/json");
  response.send(JSON.stringify(payload));
};

const buildSystemPrompt = (context: Record<string, unknown> | undefined) => `You are the Hoops Insight dashboard agent.
Use only the dashboard context supplied by the app. Be concise, call out stale or missing data, and do not present betting advice as guaranteed.
If the user asks whether bots are ready, distinguish UI readiness, data freshness, backend availability, and live betting readiness.
Dashboard context JSON:
${JSON.stringify(context ?? {}, null, 2)}`;

const callOpenAI = async (body: AgentRequest) => {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not configured and HOOPS_AGENT_API_URL is not set.");
  }

  const model = process.env.HOOPS_AGENT_MODEL || "gpt-4.1-mini";
  const messages = Array.isArray(body.messages) ? body.messages.slice(-8) : [];
  const input = [
    { role: "system", content: buildSystemPrompt(body.context) },
    ...messages.map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: String(message.content ?? ""),
    })),
  ];

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, input }),
  });
  const data = (await response.json().catch(() => ({}))) as OpenAIResponse;
  if (!response.ok) {
    const error = data?.error?.message || `OpenAI request failed with ${response.status}`;
    throw new Error(error);
  }

  const reply =
    data.output_text ||
    data.output
      ?.flatMap((item) => item.content ?? [])
      ?.map((content) => content.text ?? "")
      ?.join("\n")
      ?.trim();

  return reply || "No response text was returned by the model.";
};

const proxyAgent = async (body: AgentRequest) => {
  const upstreamUrl = process.env.HOOPS_AGENT_API_URL;
  if (!upstreamUrl) return null;

  const response = await fetch(upstreamUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await response.json().catch(() => ({}))) as AgentBackendResponse;
  if (!response.ok) {
    throw new Error(data?.error || `Upstream agent returned ${response.status}`);
  }
  return data?.reply || data?.message || data?.content || "Upstream agent returned no reply.";
};

export default async function handler(request: ApiRequest, response: ApiResponse) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return json(response, 405, { error: "Method not allowed. Use POST." });
  }

  const body = (request.body || {}) as AgentRequest;
  if (!body.message || typeof body.message !== "string") {
    return json(response, 400, { error: "Missing required string field: message." });
  }

  try {
    const proxiedReply = await proxyAgent(body);
    const reply = proxiedReply ?? (await callOpenAI(body));
    return json(response, 200, { reply });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Agent backend failed.";
    return json(response, 503, { error: message });
  }
}
