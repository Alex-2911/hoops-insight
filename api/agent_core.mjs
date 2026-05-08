const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Access-Control-Allow-Origin": process.env.HOOPS_AGENT_ALLOWED_ORIGIN || "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const READ_ONLY_CAPABILITY = "read_only";

const normalizeMessages = (messages) => (Array.isArray(messages) ? messages : [])
  .slice(-8)
  .map((message) => ({
    role: message?.role === "assistant" ? "assistant" : "user",
    content: String(message?.content ?? ""),
  }))
  .filter((message) => message.content.trim().length > 0);

const normalizeBody = (body = {}) => {
  const question = String(body.question ?? body.message ?? "").trim();
  const capability = String(body.capability ?? READ_ONLY_CAPABILITY).trim() || READ_ONLY_CAPABILITY;
  const context = body.context && typeof body.context === "object" ? body.context : {};
  const messages = normalizeMessages(body.messages);

  return { question, capability, context, messages };
};

const makeReadinessAnswer = ({ question, context }) => {
  const asOfDate = context?.as_of_date ?? context?.asOfDate ?? "unknown";
  const sources = context?.sources && typeof context.sources === "object" ? Object.values(context.sources) : [];
  const sourceList = sources.length > 0 ? sources.map(String) : ["dashboard context"];

  return {
    answer: [
      "Agent backend is reachable and accepting POST requests, but no LLM provider is configured yet.",
      "Set HOOPS_AGENT_API_URL to proxy an existing read-only agent service, or set OPENAI_API_KEY for the bundled serverless agent.",
      `I received your question: “${question || "(empty question)"}”.`,
      `Dashboard as_of_date from the supplied context: ${asOfDate}.`,
      "I cannot place bets, run shell commands, or guarantee betting outcomes.",
    ].join("\n"),
    used_sources: sourceList,
    warnings: [
      "OPENAI_API_KEY and HOOPS_AGENT_API_URL are not configured; returning readiness/mock response.",
      "Answer is limited to supplied dashboard context and configuration state.",
    ],
  };
};

const buildSystemPrompt = ({ context, capability }) => `You are the Hoops Insight dashboard agent.
Capability: ${capability}. You are read-only: do not place bets, do not instruct users to place bets, and do not claim certainty.
Use only the dashboard context supplied by the app. Be concise and call out stale or missing data.
If asked who is playing today, answer only from the supplied context; if not present, say the backend needs fresh Basketball_prediction data.
Return useful plain text.
Dashboard context JSON:
${JSON.stringify(context ?? {}, null, 2)}`;

const callOpenAI = async ({ question, capability, context, messages }) => {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;

  const model = process.env.HOOPS_AGENT_MODEL || "gpt-4.1-mini";
  const input = [
    { role: "system", content: buildSystemPrompt({ context, capability }) },
    ...messages,
    { role: "user", content: question },
  ];

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, input }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = data?.error?.message || `OpenAI request failed with ${response.status}`;
    throw new Error(error);
  }

  const answer =
    data.output_text ||
    data.output
      ?.flatMap((item) => item.content ?? [])
      ?.map((content) => content.text ?? "")
      ?.join("\n")
      ?.trim();

  return {
    answer: answer || "No response text was returned by the model.",
    used_sources: ["dashboard_context", "openai_responses_api"],
    warnings: [],
  };
};

const proxyAgent = async (payload) => {
  const upstreamUrl = process.env.HOOPS_AGENT_API_URL;
  if (!upstreamUrl) return null;

  const response = await fetch(upstreamUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || data?.message || `Upstream agent returned ${response.status}`);
  }

  return {
    answer: String(data.answer ?? data.reply ?? data.message ?? data.content ?? "Upstream agent returned no answer."),
    used_sources: Array.isArray(data.used_sources) ? data.used_sources : [],
    warnings: Array.isArray(data.warnings) ? data.warnings : [],
  };
};

export const agentJsonHeaders = JSON_HEADERS;

export const buildAgentResponse = async (rawBody = {}) => {
  const payload = normalizeBody(rawBody);

  if (!payload.question) {
    return {
      status: 400,
      body: {
        answer: "",
        used_sources: [],
        warnings: ["Missing required string field: question."],
        error: "Missing required string field: question.",
      },
    };
  }

  if (payload.capability !== READ_ONLY_CAPABILITY) {
    return {
      status: 400,
      body: {
        answer: "",
        used_sources: [],
        warnings: ["Only read_only capability is supported."],
        error: "Unsupported capability. Use read_only.",
      },
    };
  }

  const answer = (await proxyAgent(payload)) ?? (await callOpenAI(payload)) ?? makeReadinessAnswer(payload);
  return { status: 200, body: answer };
};

export const methodNotAllowedBody = {
  answer: "",
  used_sources: [],
  warnings: ["Use POST with a JSON body containing question, capability, context, and optional messages."],
  error: "Method not allowed. Use POST.",
};
