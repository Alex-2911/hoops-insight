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
      "Historical ROI Attack: a separate discretionary category",
      "decision_class=discretionary_historical_roi_attack",
      "not canonical, not Stage 1 model-approved",
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

  it("keeps broad positive ROI as watch-only when current CLE price economics are negative", async () => {
    const oldOpenAIKey = process.env.OPENAI_API_KEY;
    const oldAgentUrl = process.env.HOOPS_AGENT_API_URL;
    delete process.env.OPENAI_API_KEY;
    delete process.env.HOOPS_AGENT_API_URL;

    try {
      const result = await buildAgentResponse({
        question: "explain today's decision",
        capability: "read_only",
        context: {
          as_of_date: "2026-05-10",
          sources: { today_games: "today_games.json" },
          today_decision_context: {
            run_date: "2026-05-11",
            engine_state: "NO_BET",
            canonical_model_signals: { label: "Canonical: none", canonical_count: 0 },
            live_candidates: [
              {
                game: "CLE vs DET",
                odds_1: 1.6,
                prob_used: 0.575,
                live_ev_per_100: -7.95,
                kelly: -0.132,
                stake: 0,
                break_even_probability: 0.625,
                edge_vs_break_even: -0.05,
                current_price_supported: false,
              },
            ],
            ev_exception_summary: {
              classification: "historical_support_only",
              is_betting_signal: false,
              summary: {
                n: 71,
                wins: 44,
                win_rate: 0.62,
                avg_odds: 2.13,
                profit_100_flat: 1976,
                roi_pct: 27.8,
              },
            },
            price_adjusted_warning:
              "Broad historical EV-exception group is profitable at avg odds 2.13, but today's price 1.60 requires 62.5% break-even. Current prob_used is 57.5%; Kelly is -0.132. Treat as watch-only, not a bet.",
            decision_labels: {
              main_decision: "NO_BET",
              canonical: false,
              canonical_label: "Canonical: none",
              setup_profitability: "historical support only",
              near_miss: true,
              vibe_live_watch: true,
              no_bet_reason: "negative current EV and negative Kelly despite broad historical support",
              steadivus_note: "Good skip / no forced action.",
            },
          },
        },
      });

      assert.equal(result.status, 200);
      assert.equal(typeof result.body.answer, "string");
      assert.match(result.body.answer, /Main decision: NO_BET/);
      assert.match(result.body.answer, /Canonical: none/);
      assert.match(result.body.answer, /watch-only, not a bet/);
      assert.match(result.body.answer, /negative current EV and negative Kelly/);
      assert.match(result.body.answer, /Good skip/);
      assert.deepEqual(result.body.used_sources, ["today_games.json"]);
      assert.ok(Array.isArray(result.body.warnings));
    } finally {
      if (oldOpenAIKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = oldOpenAIKey;
      if (oldAgentUrl === undefined) delete process.env.HOOPS_AGENT_API_URL;
      else process.env.HOOPS_AGENT_API_URL = oldAgentUrl;
    }
  });

  it("retries OpenAI 5xx responses and returns normal JSON fallback after repeated 502s", async () => {
    const oldOpenAIKey = process.env.OPENAI_API_KEY;
    const oldAgentUrl = process.env.HOOPS_AGENT_API_URL;
    const oldRetryBaseMs = process.env.HOOPS_AGENT_OPENAI_RETRY_BASE_MS;
    process.env.OPENAI_API_KEY = "test-key";
    process.env.HOOPS_AGENT_OPENAI_RETRY_BASE_MS = "0";
    delete process.env.HOOPS_AGENT_API_URL;
    let calls = 0;

    try {
      const result = await buildAgentResponse(
        {
          question: "explain today's decision",
          capability: "read_only",
          context: {
            as_of_date: "2026-05-10",
            sources: { today_games: "today_games.json" },
          },
        },
        {
          runWorkflow: async () => {
            calls += 1;
            const error = new Error("OpenAI request failed with 502");
            error.status = 502;
            throw error;
          },
        },
      );

      assert.equal(result.status, 200);
      assert.equal(calls, 3);
      assert.equal(
        result.body.answer,
        "Agent backend reached OpenAI, but the provider returned a temporary 5xx error. Dashboard context is available, but no model answer was generated.",
      );
      assert.deepEqual(result.body.used_sources, ["dashboard context"]);
      assert.deepEqual(result.body.warnings, ["OpenAI provider error: 502"]);
      assert.equal(result.body.error, undefined);
    } finally {
      if (oldOpenAIKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = oldOpenAIKey;
      if (oldAgentUrl === undefined) delete process.env.HOOPS_AGENT_API_URL;
      else process.env.HOOPS_AGENT_API_URL = oldAgentUrl;
      if (oldRetryBaseMs === undefined) delete process.env.HOOPS_AGENT_OPENAI_RETRY_BASE_MS;
      else process.env.HOOPS_AGENT_OPENAI_RETRY_BASE_MS = oldRetryBaseMs;
    }
  });

  it("prepends deterministic daily decision labels to OpenAI answers", async () => {
    const oldOpenAIKey = process.env.OPENAI_API_KEY;
    const oldAgentUrl = process.env.HOOPS_AGENT_API_URL;
    process.env.OPENAI_API_KEY = "test-key";
    delete process.env.HOOPS_AGENT_API_URL;

    try {
      const result = await buildAgentResponse(
        {
          question: "Should we bet?",
          capability: "read_only",
          context: {
            today_decision_context: {
              engine_state: "NO_BET",
              decision_labels: {
                main_decision: "NO_BET",
                canonical: false,
                canonical_label: "Canonical: none",
                setup_profitability: "historical support only",
                near_miss: true,
                vibe_live_watch: true,
                no_bet_reason: "negative current EV and negative Kelly despite broad historical support",
                steadivus_note: "Good skip / no forced action.",
              },
            },
          },
        },
        { runWorkflow: async () => ({ output_text: "Model detail follows." }) },
      );

      assert.equal(result.status, 200);
      assert.match(result.body.answer, /^Main decision: NO_BET\.\nCanonical: none\./);
      assert.match(result.body.answer, /Setup profitability: historical support only/);
      assert.match(result.body.answer, /Near-miss\/watch: watch-only/);
      assert.match(result.body.answer, /Steadivus: Good skip/);
      assert.match(result.body.answer, /Model detail follows/);
      assert.deepEqual(result.body.used_sources, ["dashboard context", "Hoops Insight Betting Agent workflow"]);
    } finally {
      if (oldOpenAIKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = oldOpenAIKey;
      if (oldAgentUrl === undefined) delete process.env.HOOPS_AGENT_API_URL;
      else process.env.HOOPS_AGENT_API_URL = oldAgentUrl;
    }
  });

  it("keeps the CLE EV-exception profitability question as a golden no-bet eval", async () => {
    const oldOpenAIKey = process.env.OPENAI_API_KEY;
    const oldAgentUrl = process.env.HOOPS_AGENT_API_URL;
    process.env.OPENAI_API_KEY = "test-key";
    delete process.env.HOOPS_AGENT_API_URL;

    try {
      let workflowInput = "";
      const result = await buildAgentResponse(
        {
          question: "Why is CLE vs DET not a bet if the EV-exception setup is profitable?",
          capability: "read_only",
          context: {
            current_date: "2026-05-11",
            today_decision_context: {
              run_date: "2026-05-11",
              engine_state: "NO_BET",
              live_strategy: "none",
              canonical_model_signals: { label: "Canonical: none", canonical_count: 0 },
              live_candidates: [
                {
                  game: "CLE vs DET",
                  home_team: "CLE",
                  away_team: "DET",
                  odds_1: 1.6,
                  odds_2: 2.4,
                  raw_probability: 0.44,
                  prob_used: 0.575,
                  live_ev_per_100: -7.95,
                  kelly: -0.132,
                  stake: 0,
                  blocked_by: "min_ev",
                  canonical_signal: false,
                  candidate_type: "LOW_PRICE_NEGATIVE_EV",
                  break_even_probability: 0.625,
                  edge_vs_break_even: -0.05,
                  current_price_supported: false,
                },
              ],
              ev_exception_summary: {
                classification: "historical_support_only",
                is_betting_signal: false,
                recommendation_label: "watch-only",
                summary: {
                  n: 71,
                  wins: 44,
                  win_rate: 0.62,
                  avg_odds: 2.13,
                  profit_100_flat: 1976,
                  roi_pct: 27.8,
                },
                price_adjusted: {
                  current_odds: 1.6,
                  current_prob_used: 0.575,
                  current_ev_eur_per_100: -7.95,
                  current_kelly: -0.132,
                  current_stake_eur: 0,
                  break_even_probability: 0.625,
                  current_prob_minus_break_even: -0.05,
                  supports_play: false,
                  classification: "watch-only",
                },
              },
              price_adjusted_warning:
                "Broad historical EV-exception group is profitable at avg odds 2.13, but today's price 1.60 requires 62.5% break-even. Current prob_used is 57.5%; Kelly is -0.132. Treat as watch-only, not a bet.",
              decision_labels: {
                main_decision: "NO_BET",
                canonical: false,
                canonical_label: "Canonical: none",
                setup_profitability: "historical support only",
                near_miss: true,
                vibe_live_watch: true,
                no_bet_reason: "current price does not clear break-even; EV/Kelly negative",
                steadivus_note: "Good skip / no forced action.",
              },
            },
          },
        },
        {
          runWorkflow: async ({ input_as_text }) => {
            workflowInput = input_as_text;
            return {
              output_text:
                "CLE vs DET is watch-only / near-miss. The broad historical ROI is supportive context only and does not override today's price. Odds 1.60 require 62.5% break-even, while prob_used is 57.5%. EV -7.95 and Kelly -0.132 are negative, so there is no stake and no bet.",
            };
          },
        },
      );

      assert.equal(result.status, 200);
      assert.match(workflowInput, /Why is CLE vs DET not a bet/);
      assert.match(workflowInput, /"game": "CLE vs DET"/);
      assert.match(workflowInput, /"odds_1": 1\.6/);
      assert.match(workflowInput, /"prob_used": 0\.575/);
      assert.match(workflowInput, /"live_ev_per_100": -7\.95/);
      assert.match(workflowInput, /"kelly": -0\.132/);
      assert.match(workflowInput, /"break_even_probability": 0\.625/);
      assert.match(workflowInput, /"classification": "historical_support_only"/);

      const answer = result.body.answer;
      assert.match(answer, /Main decision: NO_BET/);
      assert.match(answer, /Canonical: none/);
      assert.match(answer, /Setup profitability: historical support only/);
      assert.match(answer, /CLE vs DET is watch-only \/ near-miss/);
      assert.match(answer, /broad historical ROI .*does not override today's price/i);
      assert.match(answer, /Odds 1\.60 require 62\.5% break-even/);
      assert.match(answer, /prob_used is 57\.5%/);
      assert.match(answer, /EV -7\.95 and Kelly -0\.132 are negative/);
      assert.match(answer, /no stake and no bet/);
      assert.match(answer, /Steadivus: Good skip \/ no forced action/);
      assert.doesNotMatch(answer, /\bcanonical bet\b/i);
      assert.doesNotMatch(answer, /\brecommend(?:ed|s)? a bet\b/i);
      assert.doesNotMatch(answer, /\bbroad historical ROI is enough\b/i);
      assert.deepEqual(result.body.used_sources, ["dashboard context", "Hoops Insight Betting Agent workflow"]);
      assert.deepEqual(result.body.warnings, []);
    } finally {
      if (oldOpenAIKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = oldOpenAIKey;
      if (oldAgentUrl === undefined) delete process.env.HOOPS_AGENT_API_URL;
      else process.env.HOOPS_AGENT_API_URL = oldAgentUrl;
    }
  });

  it("keeps Historical ROI Attack separate from canonical model bets", async () => {
    const oldOpenAIKey = process.env.OPENAI_API_KEY;
    const oldAgentUrl = process.env.HOOPS_AGENT_API_URL;
    process.env.OPENAI_API_KEY = "test-key";
    delete process.env.HOOPS_AGENT_API_URL;

    try {
      let workflowInput = "";
      const result = await buildAgentResponse(
        {
          question: "How should we classify today's DET vs CLE historical ROI attack?",
          capability: "read_only",
          context: {
            current_date: "2026-05-13",
            historical_roi_attack: {
              game: "DET vs CLE",
              decision_class: "discretionary_historical_roi_attack",
              canonical_signal: false,
              stage1_bucket: "RAW_UNDERDOG_TEMPTATION",
              official_setup_scan_candidate_count: 0,
              script11_confirms_bet: false,
              prob_used: null,
              odds: 1.56,
              break_even_probability: 0.641,
              broad_bucket: { n: 100, roi_pct: 1.17 },
              price_strict_bucket: { n: 36, roi_pct: 11.83 },
              hwr_filtered_bucket: { n: 24, roi_pct: 13.46, win_rate: 0.75 },
              stake_type: "small_fixed",
              reason: "price_strict_and_hwr_filtered_history_profitable",
              risk_note: "not canonical; official model did not approve",
            },
          },
        },
        {
          runWorkflow: async ({ input_as_text }) => {
            workflowInput = input_as_text;
            return {
              output_text:
                "Historical ROI attack: DET vs CLE is discretionary, small stake only, not canonical, and not Stage 1 model-approved. The price-strict and HWR-filtered buckets are profitable and the HWR-filtered 75.00% win rate clears the 64.10% break-even at odds 1.56. This should be tracked separately with decision_class=discretionary_historical_roi_attack and canonical_signal=false.",
            };
          },
        },
      );

      assert.equal(result.status, 200);
      assert.match(workflowInput, /"decision_class": "discretionary_historical_roi_attack"/);
      assert.match(workflowInput, /"canonical_signal": false/);
      assert.match(workflowInput, /"stage1_bucket": "RAW_UNDERDOG_TEMPTATION"/);
      assert.match(workflowInput, /"official_setup_scan_candidate_count": 0/);
      assert.match(workflowInput, /"break_even_probability": 0\.641/);
      assert.match(workflowInput, /"price_strict_bucket"/);
      assert.match(workflowInput, /"hwr_filtered_bucket"/);

      const answer = result.body.answer;
      assert.match(answer, /Historical ROI attack/i);
      assert.match(answer, /discretionary/);
      assert.match(answer, /small stake only/);
      assert.match(answer, /not canonical/);
      assert.match(answer, /not Stage 1 model-approved/);
      assert.match(answer, /75\.00% win rate clears the 64\.10% break-even/);
      assert.match(answer, /decision_class=discretionary_historical_roi_attack/);
      assert.match(answer, /canonical_signal=false/);
      assert.doesNotMatch(answer, /\bcanonical model bet\b/i);
      assert.doesNotMatch(answer, /\block\b/i);
      assert.doesNotMatch(answer, /\bguaranteed\b/i);
      assert.doesNotMatch(answer, /\bautomatic bet\b/i);
      assert.deepEqual(result.body.used_sources, ["dashboard context", "Hoops Insight Betting Agent workflow"]);
      assert.deepEqual(result.body.warnings, []);
    } finally {
      if (oldOpenAIKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = oldOpenAIKey;
      if (oldAgentUrl === undefined) delete process.env.HOOPS_AGENT_API_URL;
      else process.env.HOOPS_AGENT_API_URL = oldAgentUrl;
    }
  });

  it("defines CORS and JSON method-not-allowed response", () => {
    assert.equal(agentJsonHeaders["Access-Control-Allow-Methods"], "POST, OPTIONS");
    assert.equal(methodNotAllowedBody.answer, "");
    assert.deepEqual(methodNotAllowedBody.used_sources, []);
    assert.match(methodNotAllowedBody.error, /Method not allowed/);
  });
});
