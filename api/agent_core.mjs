const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Access-Control-Allow-Origin": process.env.HOOPS_AGENT_ALLOWED_ORIGIN || "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const READ_ONLY_CAPABILITY = "read_only";
const OPENAI_TRANSIENT_STATUSES = new Set([500, 502, 503]);
const OPENAI_MAX_5XX_RETRIES = 2;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const openAiRetryDelayMs = (attempt) => {
  const configured = Number.parseInt(process.env.HOOPS_AGENT_OPENAI_RETRY_BASE_MS ?? "250", 10);
  const baseMs = Number.isFinite(configured) && configured >= 0 ? configured : 250;
  return baseMs * attempt;
};

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
  const decisionContext = context?.today_decision_context && typeof context.today_decision_context === "object"
    ? context.today_decision_context
    : null;
  const decisionLabels = decisionContext?.decision_labels && typeof decisionContext.decision_labels === "object"
    ? decisionContext.decision_labels
    : null;
  const decisionLines = decisionContext
    ? [
        "",
        "Supplied daily decision context:",
        `Main decision: ${decisionLabels?.main_decision ?? decisionContext.engine_state ?? "unknown"}.`,
        `Canonical: ${decisionLabels?.canonical_label ?? "unknown"}.`,
        decisionContext.price_adjusted_warning ? `Price-adjusted warning: ${decisionContext.price_adjusted_warning}` : null,
        decisionLabels?.no_bet_reason ? `No-bet reason: ${decisionLabels.no_bet_reason}.` : null,
        decisionLabels?.steadivus_note ? `Steadivus: ${decisionLabels.steadivus_note}` : null,
      ].filter(Boolean)
    : [];

  return {
    answer: [
      "Agent backend is reachable and accepting POST requests, but no LLM provider is configured yet.",
      "Set HOOPS_AGENT_API_URL to proxy an existing read-only agent service, or set OPENAI_API_KEY for the bundled Agents SDK workflow.",
      `I received your question: “${question || "(empty question)"}”.`,
      `Dashboard as_of_date from the supplied context: ${asOfDate}.`,
      "I cannot place bets, run shell commands, or guarantee betting outcomes.",
      ...decisionLines,
    ].join("\n"),
    used_sources: sourceList,
    warnings: [
      "OPENAI_API_KEY and HOOPS_AGENT_API_URL are not configured; returning readiness/mock response.",
      "Answer is limited to supplied dashboard context and configuration state.",
    ],
  };
};

const buildDecisionContextPrefix = (context) => {
  const decisionContext = context?.today_decision_context && typeof context.today_decision_context === "object"
    ? context.today_decision_context
    : null;
  const decisionLabels = decisionContext?.decision_labels && typeof decisionContext.decision_labels === "object"
    ? decisionContext.decision_labels
    : null;
  if (!decisionLabels) return "";
  const canonicalLabel = String(
    decisionLabels.canonical_label ?? (decisionLabels.canonical ? "present" : "none"),
  ).replace(/^Canonical:\s*/i, "");

  return [
    `Main decision: ${decisionLabels.main_decision ?? decisionContext.engine_state ?? "unknown"}.`,
    `Canonical: ${canonicalLabel}.`,
    `Setup profitability: ${decisionLabels.setup_profitability ?? "historical context only"}.`,
    `Near-miss/watch: ${decisionLabels.near_miss || decisionLabels.vibe_live_watch ? "watch-only" : "none"}.`,
    decisionLabels.no_bet_reason ? `No-bet reason: ${decisionLabels.no_bet_reason}.` : null,
    decisionLabels.steadivus_note ? `Steadivus: ${decisionLabels.steadivus_note}` : null,
  ].filter(Boolean).join("\n");
};

const buildWorkflowInput = ({ question, context, messages }) => [
  `User question: ${question}`,
  `Dashboard context: ${JSON.stringify(context, null, 2)}`,
  `Recent messages: ${JSON.stringify(messages ?? [], null, 2)}`,
].join("\n\n");

const extractOpenAIStatus = (error) => {
  const candidates = [
    error?.status,
    error?.statusCode,
    error?.response?.status,
    error?.cause?.status,
    error?.cause?.statusCode,
  ];
  for (const candidate of candidates) {
    const parsed = Number.parseInt(String(candidate ?? ""), 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  const message = error instanceof Error ? error.message : String(error ?? "");
  const match = message.match(/\b(500|502|503)\b/);
  return match ? Number.parseInt(match[1], 10) : null;
};

const callBundledWorkflow = async (payload, workflowRunner) => {
  if (!process.env.OPENAI_API_KEY || process.env.HOOPS_AGENT_API_URL) return null;

  const runWorkflow = workflowRunner ?? (await import("./hoops_insight_betting_agent_workflow.mjs")).runWorkflow;
  let lastTransientStatus = null;

  for (let attempt = 0; attempt <= OPENAI_MAX_5XX_RETRIES; attempt += 1) {
    if (attempt > 0) {
      await sleep(openAiRetryDelayMs(attempt));
    }

    try {
      const { output_text } = await runWorkflow({
        input_as_text: buildWorkflowInput(payload),
      });
      const decisionPrefix = buildDecisionContextPrefix(payload.context);
      const workflowAnswer = String(output_text || "No response text was returned by the Hoops Insight Betting Agent workflow.");

      return {
        answer: decisionPrefix ? `${decisionPrefix}\n\n${workflowAnswer}` : workflowAnswer,
        used_sources: ["dashboard context", "Hoops Insight Betting Agent workflow"],
        warnings: [],
      };
    } catch (error) {
      const status = extractOpenAIStatus(error);
      if (status && OPENAI_TRANSIENT_STATUSES.has(status)) {
        lastTransientStatus = status;
        continue;
      }
      throw error;
    }
  }

  return {
    answer:
      "Agent backend reached OpenAI, but the provider returned a temporary 5xx error. Dashboard context is available, but no model answer was generated.",
    used_sources: ["dashboard context"],
    warnings: [`OpenAI provider error: ${lastTransientStatus ?? "5xx"}`],
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

export const buildAgentResponse = async (rawBody = {}, options = {}) => {
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

  const answer = (await proxyAgent(payload))
    ?? (await callBundledWorkflow(payload, options.runWorkflow))
    ?? makeReadinessAnswer(payload);
  return { status: 200, body: answer };
};

export const methodNotAllowedBody = {
  answer: "",
  used_sources: [],
  warnings: ["Use POST with a JSON body containing question, capability, context, and optional messages."],
  error: "Method not allowed. Use POST.",
};
