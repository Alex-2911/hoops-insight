import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { agentJsonHeaders, buildAgentResponse, methodNotAllowedBody } from "../api/agent_core.mjs";
import { HOOPS_INSIGHT_BETTING_AGENT_INSTRUCTIONS } from "../api/hoops_insight_betting_agent_workflow.mjs";

describe("agent backend response contract", () => {
  it("returns readiness JSON for read-only questions when no provider is configured", async () => {
    const oldOpenAIKey = process.env.OPENAI_API_KEY;
    const oldAgentUrl = process.env.HOOPS_AGENT_API_URL;
    delete process.env.OPENAI_API_KEY;
    delete process.env.HOOPS_AGENT_API_URL;

    try {
      const result = await buildAgentResponse({
        question: "who is playing today?",
        capability: "read_only",
        context: {
          as_of_date: "2026-05-08",
          sources: { combined: "combined_latest.csv" },
        },
      });

      assert.equal(result.status, 200);
      assert.equal(typeof result.body.answer, "string");
      assert.match(result.body.answer, /Agent backend is reachable/);
      assert.match(result.body.answer, /who is playing today/);
      assert.deepEqual(result.body.used_sources, ["combined_latest.csv"]);
      assert.ok(Array.isArray(result.body.warnings));
      assert.ok(result.body.warnings.length > 0);
    } finally {
      if (oldOpenAIKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = oldOpenAIKey;
      if (oldAgentUrl === undefined) delete process.env.HOOPS_AGENT_API_URL;
      else process.env.HOOPS_AGENT_API_URL = oldAgentUrl;
    }
  });

  it("keeps the full Hoops Insight betting-agent instruction hierarchy", () => {
    const requiredPhrases = [
      "canonical model signals separate from setup-profitability candidates",
      "Stage 1 snapshot as the source of truth for canonical_signal",
      "setup_profitability_scan is historical support only",
      "script11_watchlist_history is watchlist context",
      "actual_bets_manual.csv is the source for real placed bets",
      "robustness/engine_state NO_BET is the main decision",
      "Market-gap blocks are serious warnings",
      "Steadivus good skips are discipline wins",
      "Never place bets",
      "Never say guaranteed",
    ];

    const instructions = HOOPS_INSIGHT_BETTING_AGENT_INSTRUCTIONS.toLowerCase();
    for (const phrase of requiredPhrases) {
      assert.ok(instructions.includes(phrase.toLowerCase()), `Missing instruction phrase: ${phrase}`);
    }
  });

  it("calls the bundled Agents SDK workflow when OPENAI_API_KEY is set and no upstream URL is configured", async () => {
    const oldOpenAIKey = process.env.OPENAI_API_KEY;
    const oldAgentUrl = process.env.HOOPS_AGENT_API_URL;
    process.env.OPENAI_API_KEY = "test-key";
    delete process.env.HOOPS_AGENT_API_URL;

    try {
      let workflowInput = null;
      const result = await buildAgentResponse(
        {
          question: "who has the best edge?",
          capability: "read_only",
          context: { as_of_date: "2026-05-08", cards: [{ team: "NYK" }] },
          messages: [{ role: "assistant", content: "Previous answer" }],
        },
        {
          runWorkflow: async ({ input_as_text }) => {
            workflowInput = input_as_text;
            return { output_text: "Workflow answer" };
          },
        },
      );

      assert.equal(result.status, 200);
      assert.equal(result.body.answer, "Workflow answer");
      assert.deepEqual(result.body.used_sources, ["dashboard context", "Hoops Insight Betting Agent workflow"]);
      assert.deepEqual(result.body.warnings, []);
      assert.match(workflowInput, /User question: who has the best edge\?/);
      assert.match(workflowInput, /Dashboard context:/);
      assert.match(workflowInput, /Recent messages:/);
      assert.match(workflowInput, /Previous answer/);
    } finally {
      if (oldOpenAIKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = oldOpenAIKey;
      if (oldAgentUrl === undefined) delete process.env.HOOPS_AGENT_API_URL;
      else process.env.HOOPS_AGENT_API_URL = oldAgentUrl;
    }
  });

  it("rejects missing questions with JSON error shape", async () => {
    const result = await buildAgentResponse({ capability: "read_only", context: {} });

    assert.equal(result.status, 400);
    assert.equal(result.body.answer, "");
    assert.deepEqual(result.body.used_sources, []);
    assert.deepEqual(result.body.warnings, ["Missing required string field: question."]);
    assert.equal(result.body.error, "Missing required string field: question.");
  });

  it("rejects non-read-only capabilities", async () => {
    const result = await buildAgentResponse({ question: "place a bet", capability: "write" });

    assert.equal(result.status, 400);
    assert.equal(result.body.answer, "");
    assert.deepEqual(result.body.used_sources, []);
    assert.deepEqual(result.body.warnings, ["Only read_only capability is supported."]);
    assert.equal(result.body.error, "Unsupported capability. Use read_only.");
  });

  it("defines CORS and JSON method-not-allowed response", () => {
    assert.equal(agentJsonHeaders["Access-Control-Allow-Methods"], "POST, OPTIONS");
    assert.equal(methodNotAllowedBody.answer, "");
    assert.deepEqual(methodNotAllowedBody.used_sources, []);
    assert.match(methodNotAllowedBody.error, /Method not allowed/);
  });
});
