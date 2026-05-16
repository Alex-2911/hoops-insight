import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { StatCard } from "@/components/cards/StatCard";
import type {
  DashboardPayload,
  DashboardState,
  LastRunPayload,
  LocalMatchedGameRow,
  SummaryPayload,
  TablesPayload,
  TodayGamesPayload,
} from "@/data/dashboardTypes";
import { Target, TrendingUp, Activity, BarChart3, Info } from "lucide-react";
import { fmtCurrencyEUR, fmtNumber, fmtPercent, formatSigned } from "@/lib/format";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

type AgentMessage = {
  role: "user" | "assistant";
  content: string;
};

type AgentResponse = {
  answer?: string;
  reply?: string;
  error?: string;
  used_sources?: string[];
  warnings?: string[];
};

type TodayDecisionCandidate = {
  game: string;
  home_team: string;
  away_team: string;
  odds_1: number | null;
  odds_2: number | null;
  raw_probability: number | null;
  prob_used: number | null;
  live_ev_per_100: number | null;
  kelly: number | null;
  stake: number | null;
  blocked_by: string;
  canonical_signal: boolean;
  candidate_type: string;
  break_even_probability: number | null;
  edge_vs_break_even: number | null;
  current_price_supported: boolean;
};

type ActualBetRow = {
  bet_id: string;
  bet_date: string;
  game_date: string;
  league: string;
  home_team: string;
  away_team: string;
  selection: string;
  market: string;
  bet_type: string;
  line: string;
  odds: string;
  stake_eur: string;
  payout_eur: string;
  pnl_eur: string;
  status: string;
  bookmaker: string;
  source: string;
  canonical_decision: string;
  stage2_label: string;
  model_note: string;
  user_note: string;
  screenshot_ref: string;
};

type AgentLearningCase = Record<string, unknown>;

const Index = () => {
  const [activeTab, setActiveTab] = useState<"play-today" | "overview" | "actual-bets" | "agent-chat">("play-today");
  const [payload, setPayload] = useState<DashboardPayload | null>(null);
  const [dashboardState, setDashboardState] = useState<DashboardState | null>(null);
  const [localMatchedLatestRows, setLocalMatchedLatestRows] = useState<LocalMatchedGameRow[]>([]);
  const [tablesFallback, setTablesFallback] = useState<TablesPayload | null>(null);
  const [todayGames, setTodayGames] = useState<TodayGamesPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [actualBetsRows, setActualBetsRows] = useState<ActualBetRow[]>([]);
  const [agentLearningCases, setAgentLearningCases] = useState<AgentLearningCase[]>([]);
  const [fetchStarted, setFetchStarted] = useState(false);
  const [agentMessages, setAgentMessages] = useState<AgentMessage[]>([
    {
      role: "assistant",
      content:
        "Ask about dashboard freshness, current filters, simulated matches, or manual bets. I will use the loaded dashboard context only.",
    },
  ]);
  const [agentInput, setAgentInput] = useState("");
  const [agentStatus, setAgentStatus] = useState<"idle" | "sending" | "error">("idle");
  const [agentError, setAgentError] = useState<string | null>(null);
  const agentScrollRef = useRef<HTMLDivElement | null>(null);
  const agentInputRef = useRef<HTMLInputElement | null>(null);
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  const staleMessage = dashboardState?.last_update_utc
    ? `Last update: ${dashboardState.last_update_utc}`
    : null;
  const agentEndpoint = import.meta.env.VITE_HOOPS_AGENT_API_URL ?? "/api/agent";


  const parseLocalMatchedCsv = (csvText: string): LocalMatchedGameRow[] => {
    const trimmed = csvText.trim();
    if (!trimmed) return [];
    const lines = trimmed.split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) return [];

    const delimiter = (() => {
      const headerLine = lines[0];
      const candidates: Array<{ delimiter: string; count: number }> = [
        { delimiter: "\t", count: headerLine.split("\t").length - 1 },
        { delimiter: ";", count: headerLine.split(";").length - 1 },
        { delimiter: ",", count: headerLine.split(",").length - 1 },
      ];
      return candidates.sort((a, b) => b.count - a.count)[0]?.delimiter ?? ",";
    })();

    const headers = lines[0]
      .split(delimiter)
      .map((header) => header.replace(/^\uFEFF/, "").trim());

    const normalizeHeader = (header: string) =>
      header
        .trim()
        .toLowerCase()
        .replace(/[^\w]+/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "");

    const headerAliases: Record<string, string> = {
      closing_home_odds: "odds_1",
      odds: "odds_1",
      odds_1: "odds_1",
      ev_per_100: "ev_eur_per_100",
      ev_eur_per_100: "ev_eur_per_100",
      ev_eur_per100: "ev_eur_per_100",
      ev_per100: "ev_eur_per_100",
      p_l: "pnl",
      pl: "pnl",
      pnl: "pnl",
    };

    const headerKeys = headers.map((header) => {
      const normalized = normalizeHeader(header);
      if (normalized.includes("ev") && normalized.includes("per_100")) {
        return "ev_eur_per_100";
      }
      return headerAliases[normalized] ?? normalized;
    });

    const headerIndex = headerKeys.reduce<Record<string, number>>((acc, key, index) => {
      if (!(key in acc)) acc[key] = index;
      return acc;
    }, {});

    const readString = (columns: string[], key: string) => {
      const index = headerIndex[key];
      return index !== undefined ? columns[index]?.trim() ?? "" : "";
    };
    const readNumber = (columns: string[], key: string) => {
      const raw = readString(columns, key);
      const parsed = Number.parseFloat(raw);
      return Number.isFinite(parsed) ? parsed : Number.NaN;
    };

    return lines.slice(1).reduce<LocalMatchedGameRow[]>((acc, line) => {
      const columns = line.split(delimiter);
      const date = readString(columns, "date");
      const homeTeam = readString(columns, "home_team");
      const awayTeam = readString(columns, "away_team");
      if (!date || !homeTeam || !awayTeam) return acc;

      acc.push({
        date,
        home_team: homeTeam,
        away_team: awayTeam,
        home_win_rate: readNumber(columns, "home_win_rate"),
        prob_iso: readNumber(columns, "prob_iso"),
        prob_used: readNumber(columns, "prob_used"),
        odds_1: readNumber(columns, "odds_1"),
        ev_eur_per_100: readNumber(columns, "ev_eur_per_100"),
        win: readNumber(columns, "win"),
        pnl: readNumber(columns, "pnl"),
      });
      return acc;
    }, []);
  };

  const parseActualBetsCsv = (csvText: string): ActualBetRow[] => {
    const trimmed = csvText.trim();
    if (!trimmed) return [];
    const lines = trimmed.split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) return [];
    const headers = lines[0].split(",").map((header) => header.replace(/^\uFEFF/, "").trim());
    const rows = lines.slice(1).map((line) => line.split(","));
    const sanitizeCsvCell = (value: string) => {
      const trimmed = value.trim();
      const withoutWrappingQuotes =
        trimmed.startsWith("\"") && trimmed.endsWith("\"") && trimmed.length >= 2
          ? trimmed.slice(1, -1)
          : trimmed;
      return withoutWrappingQuotes.replace(/""/g, "\"");
    };
    return rows.map((columns) =>
      headers.reduce<Record<string, string>>((acc, header, index) => {
        acc[header] = sanitizeCsvCell(columns[index] ?? "");
        return acc;
      }, {}) as ActualBetRow,
    );
  };

  const readRecordString = (row: Record<string, unknown> | null | undefined, key: string) => {
    const value = row?.[key];
    return value === undefined || value === null ? "" : String(value);
  };

  const readRecordNumber = (row: Record<string, unknown> | null | undefined, key: string) => {
    const value = row?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value !== "string") return null;
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const isCanonicalSignal = (value: unknown) => {
    if (value === true) return true;
    if (typeof value === "number") return value > 0;
    if (typeof value !== "string") return false;
    return ["1", "true", "yes", "canonical", "stage1"].includes(value.trim().toLowerCase());
  };

  useEffect(() => {
    let alive = true;

    const load = async () => {
      setFetchStarted(true);
      try {
        const cacheBust = new URLSearchParams(window.location.search).get("v") ?? String(Date.now());
        const dataUrl = (file: string) => `${baseUrl}data/${file}?v=${encodeURIComponent(cacheBust)}`;
        const [summaryRes, tablesRes, lastRunRes, dashboardStateRes, todayGamesRes] = await Promise.all([
          fetch(dataUrl("summary.json")),
          fetch(dataUrl("tables.json")),
          fetch(dataUrl("last_run.json")),
          fetch(dataUrl("dashboard_state.json")),
          fetch(dataUrl("today_games.json")),
        ]);

        if (!summaryRes.ok || !tablesRes.ok) {
          throw new Error("Failed to load dashboard data.");
        }

        const summaryJson = (await summaryRes.json()) as SummaryPayload;
        const tablesJson = (await tablesRes.json()) as TablesPayload;
        const lastRunJson = lastRunRes.ok ? ((await lastRunRes.json()) as LastRunPayload) : null;
        const dashboardStateJson = dashboardStateRes.ok ? ((await dashboardStateRes.json()) as DashboardState) : null;
        const todayGamesJson = todayGamesRes.ok ? ((await todayGamesRes.json()) as TodayGamesPayload) : null;

        const normalizedPayload = {
          as_of_date: summaryJson.as_of_date ?? summaryJson.asOfDate ?? "—",
          window: {
            size: 200,
            start: summaryJson.window_start ?? null,
            end: summaryJson.window_end ?? summaryJson.as_of_date ?? null,
            games_count: summaryJson.summary_stats?.total_games ?? 0,
          },
          active_filters_effective: summaryJson.active_filters_human ?? summaryJson.active_filters ?? "No active filters.",
          summary: summaryJson,
          tables: tablesJson,
          last_run: lastRunJson,
        } as DashboardPayload;
        const fallbackState = {
          as_of_date: normalizedPayload.as_of_date,
          window_size: normalizedPayload.window.size,
          window_start: normalizedPayload.window.start,
          window_end: normalizedPayload.window.end,
          active_filters_text: summaryJson.active_filters ?? normalizedPayload.active_filters_effective,
          params_used_label: summaryJson.params_used_type ?? "metrics_snapshot",
          params_source_label: summaryJson.source?.metrics_snapshot_file ?? "metrics_snapshot",
          last_update_utc: summaryJson.last_run ?? new Date().toISOString(),
          sources: {
            combined: summaryJson.source?.combined_file ?? "combined_latest.csv",
            local_matched: summaryJson.source?.local_matched_file ?? "local_matched_games_latest.csv",
            bet_log: summaryJson.source?.bet_log_file ?? "bet_log_flat_live.csv",
          },
          strategy_matches_window: tablesJson.local_matched_games_count ?? 0,
          data_consistency_status: "ok",
          data_consistency_issues: [],
        } as DashboardState;
        const normalizedState = dashboardStateJson ?? fallbackState;

        if (alive) {
          setPayload(normalizedPayload);
          setDashboardState(normalizedState);
          setTablesFallback(tablesJson);
          setTodayGames(todayGamesJson);
        }

        const localMatchedJsonRes = await fetch(dataUrl("local_matched_games_latest.json"));
        if (alive && localMatchedJsonRes.ok) {
          const localMatchedJson = (await localMatchedJsonRes.json()) as { rows?: LocalMatchedGameRow[] };
          if (Array.isArray(localMatchedJson.rows) && localMatchedJson.rows.length > 0) {
            setLocalMatchedLatestRows(localMatchedJson.rows);
          }
        } else {
          const localMatchedSource = normalizedState.sources?.local_matched ?? "local_matched_games_latest.csv";
          const localMatchedRes = await fetch(dataUrl(localMatchedSource));
          if (alive && localMatchedRes.ok) {
            const localMatchedText = await localMatchedRes.text();
            const parsedRows = parseLocalMatchedCsv(localMatchedText);
            if (parsedRows.length > 0) setLocalMatchedLatestRows(parsedRows);
          }
        }

        const actualBetsRes = await fetch(dataUrl("actual_bets_manual.csv"));
        if (alive && actualBetsRes.ok) {
          const actualBetsText = await actualBetsRes.text();
          setActualBetsRows(parseActualBetsCsv(actualBetsText));
        }

        const learningCasesRes = await fetch(dataUrl("agent_learning_cases.jsonl"));
        if (alive && learningCasesRes.ok) {
          const learningCasesText = await learningCasesRes.text();
          const parsedCases = learningCasesText
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean)
            .map((line) => {
              try {
                return JSON.parse(line) as AgentLearningCase;
              } catch {
                return null;
              }
            })
            .filter((row): row is AgentLearningCase => row !== null);
          setAgentLearningCases(parsedCases);
        }

      } catch (err) {
        if (alive) {
          setLoadError(err instanceof Error ? err.message : "Failed to load data.");
        }
      } finally {
        if (alive) setIsLoading(false);
      }
    };

    load();
    return () => {
      alive = false;
    };
  }, [baseUrl]);

  const summary = payload?.summary ?? null;

  const tables = useMemo(() => {
    if (!payload?.tables) {
      return tablesFallback ?? null;
    }
    if (!tablesFallback) {
      return payload.tables;
    }
    const pickArray = <T,>(primary?: T[], fallback?: T[]) =>
      primary && primary.length > 0 ? primary : fallback ?? [];
    return {
      ...payload.tables,
      ...tablesFallback,
      historical_stats: pickArray(payload.tables.historical_stats, tablesFallback.historical_stats),
      accuracy_threshold_stats: pickArray(
        payload.tables.accuracy_threshold_stats,
        tablesFallback.accuracy_threshold_stats,
      ),
      home_win_rates_last20: pickArray(payload.tables.home_win_rates_last20, tablesFallback.home_win_rates_last20),
      home_win_rates_window: pickArray(payload.tables.home_win_rates_window, tablesFallback.home_win_rates_window),
      bankroll_history: pickArray(payload.tables.bankroll_history, tablesFallback.bankroll_history),
      settled_bets_rows: pickArray(payload.tables.settled_bets_rows, tablesFallback.settled_bets_rows),
      local_matched_games_rows: pickArray(payload.tables.local_matched_games_rows, tablesFallback.local_matched_games_rows),
      calibration_metrics: tablesFallback.calibration_metrics ?? payload.tables.calibration_metrics,
      bet_log_summary: tablesFallback.bet_log_summary ?? payload.tables.bet_log_summary,
      bankroll_ytd_2026: tablesFallback.bankroll_ytd_2026 ?? payload.tables.bankroll_ytd_2026,
      settled_bets_summary: tablesFallback.settled_bets_summary ?? payload.tables.settled_bets_summary,
      local_matched_games_count: tablesFallback.local_matched_games_count ?? payload.tables.local_matched_games_count,
      local_matched_games_profit_sum_table:
        tablesFallback.local_matched_games_profit_sum_table ?? payload.tables.local_matched_games_profit_sum_table,
      local_matched_games_note: tablesFallback.local_matched_games_note ?? payload.tables.local_matched_games_note,
      local_matched_games_mismatch:
        tablesFallback.local_matched_games_mismatch ?? payload.tables.local_matched_games_mismatch,
    };
  }, [payload?.tables, tablesFallback]);

  const windowInfo = payload?.window ?? {
    size: 0,
    start: "—",
    end: "—",
    games_count: 0,
  };

  const summaryStats = useMemo(
    () =>
      summary?.summary_stats ??
      (summary?.model?.calibration
        ? {
            total_games: summary.model.calibration.fittedGames ?? windowInfo.games_count ?? 0,
            overall_accuracy: summary.model.calibration.actualWinPct ?? 0,
            as_of_date: summary.model.calibration.asOfDate ?? "—",
          }
        : {
            total_games: 0,
            overall_accuracy: 0,
            as_of_date: "—",
          }),
    [summary?.model?.calibration, summary?.summary_stats, windowInfo.games_count],
  );

  const calibrationMetrics =
    tables?.calibration_metrics ??
    summary?.model?.calibration ?? {
      asOfDate: "—",
      brierBefore: 0,
      brierAfter: 0,
      logLossBefore: 0,
      logLossAfter: 0,
      fittedGames: 0,
      ece: 0,
      calibrationSlope: 0,
      calibrationIntercept: 0,
      avgPredictedProb: 0,
      baseRate: 0,
      actualWinPct: 0,
      windowSize: 0,
    };

  const homeWinRatesLast20 = useMemo(
    () =>
      tables?.home_win_rates_last20 && tables.home_win_rates_last20.length > 0
        ? tables.home_win_rates_last20
        : summary?.model?.homeWinRatesLast20 ?? [],
    [summary?.model?.homeWinRatesLast20, tables?.home_win_rates_last20],
  );
  const homeWinRatesWindow = useMemo(() => tables?.home_win_rates_window ?? [], [tables?.home_win_rates_window]);
  const localMatchedGamesRows = tables?.local_matched_games_rows ?? [];
  const localMatchedRowsDisplay = localMatchedLatestRows.length > 0 ? localMatchedLatestRows : localMatchedGamesRows;
  const settledBetsRows = useMemo(() => tables?.settled_bets_rows ?? [], [tables?.settled_bets_rows]);

  const START_BANKROLL_REAL = 1000;
  const START_BANKROLL_SIM = 1000;

  const settledBetsSummary = useMemo(() => {
    if (!settledBetsRows.length) {
      return {
        count: 0,
        wins: 0,
        profit_eur: 0,
        roi_pct: 0,
        avg_odds: 0,
        total_stake: 0,
      };
    }
    const wins = settledBetsRows.filter((row) => row.win === 1).length;
    const profit = settledBetsRows.reduce((acc, row) => acc + (row.pnl ?? 0), 0);
    const totalStake = settledBetsRows.reduce((acc, row) => acc + (row.stake ?? 0), 0);
    const avgOdds = settledBetsRows.reduce((acc, row) => acc + (row.odds ?? 0), 0) / settledBetsRows.length;
    return {
      count: settledBetsRows.length,
      wins,
      profit_eur: profit,
      roi_pct: totalStake > 0 ? (profit / totalStake) * 100 : 0,
      avg_odds: avgOdds,
      total_stake: totalStake,
    };
  }, [settledBetsRows]);

  const realBankroll = START_BANKROLL_REAL + settledBetsSummary.profit_eur;
  const settledWinRatePct =
    settledBetsSummary.count > 0 ? (settledBetsSummary.wins / settledBetsSummary.count) * 100 : 0;

  const localMatchedProfitSum =
    tables?.local_matched_games_profit_sum_table ?? localMatchedGamesRows.reduce((acc, row) => acc + (row.pnl ?? 0), 0);

  const localMatchedCountFromTables = tables?.local_matched_games_count ?? 0;
  const localMatchedCountFromSummary = summary?.strategy_filter_stats?.matched_games_count ?? 0;
  const localMatchedCountFromState = dashboardState?.strategy_matches_window ?? 0;
  const localMatchedRowsCount = localMatchedGamesRows.length;

  const localMatchedTotalCount = Math.max(
    localMatchedCountFromTables,
    localMatchedCountFromSummary,
    localMatchedCountFromState,
    localMatchedRowsCount,
  );

  const localMatchedWins = localMatchedGamesRows.filter((row) => row.win === 1).length;

  const kpis = useMemo(
    () =>
      summary?.kpis ?? {
        total_bets: 0,
        win_rate: 0,
        roi_pct: 0,
        avg_ev_per_100: 0,
        avg_profit_per_bet_eur: 0,
        max_drawdown_eur: null,
        max_drawdown_pct: null,
      },
    [summary?.kpis],
  );

  // NOTE: in your payload, "strategy_summary" may only contain sharpeStyle.
  // "strategy" contains the full simulated strategy block (roiPct, totalProfitEur, etc).
  const strategySummary = useMemo(
    () =>
      summary?.strategy_summary ?? {
        totalBets: 0,
        totalProfitEur: 0,
        roiPct: 0,
        avgEvPer100: 0,
        winRate: 0,
        sharpeStyle: null,
        profitMetricsAvailable: false,
        asOfDate: "—",
      },
    [summary?.strategy_summary],
  );

  const renderMetricTitle = (label: string, tooltipContent: React.ReactNode) => (
    <span className="inline-flex items-center gap-2">
      <span>{label}</span>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label={`${label} info`}
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            <Info className="h-4 w-4" />
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          align="start"
          className="max-w-[300px] whitespace-pre-line border-slate-800 bg-slate-900 text-slate-100"
        >
          <div className="space-y-2 text-sm leading-snug">{tooltipContent}</div>
        </TooltipContent>
      </Tooltip>
    </span>
  );

  const strategyParams = summary?.strategy_params ?? {
    source: "missing",
    params: {},
    params_used: {},
    active_filters: "No active filters.",
  };

  const pickNumber = (...values: Array<number | null | undefined>) =>
    values.find((value) => typeof value === "number" && Number.isFinite(value)) ?? null;

  const pickPositiveNumber = (...values: Array<number | null | undefined>) =>
    values.find((value) => typeof value === "number" && Number.isFinite(value) && value > 0) ?? null;

  const formatPercentFromMaybeRatio = (value: number | null, decimals = 2) => {
    if (value === null) return "—";
    const percentValue = Math.abs(value) <= 1 ? value * 100 : value;
    return fmtPercent(percentValue, decimals);
  };

  const parseNumeric = (value: string) => {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const localMatchedSourceLabel = dashboardState?.sources?.local_matched ?? "local_matched_games_latest.csv";
  const betLogFlatSource = dashboardState?.sources?.bet_log ?? "bet_log_flat_live.csv";
  const combinedSource = dashboardState?.sources?.combined ?? "combined_latest.csv";

  const summaryAsOfDate = dashboardState?.as_of_date ?? payload?.as_of_date ?? summary?.as_of_date ?? "—";

  const activeParams = dashboardState?.active_params;

  const readActiveParam = (key: keyof NonNullable<DashboardState["active_params"]>) => {
    const value = activeParams?.[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };

  const activeParamKeys: Array<keyof NonNullable<DashboardState["active_params"]>> = [
    "home_win_rate_min",
    "odds_min",
    "odds_max",
    "prob_threshold",
    "min_ev",
    "window_size",
  ];
  const activeParamsComplete = activeParamKeys.every((key) => readActiveParam(key) !== null);

  const homeWinRateMin = readActiveParam("home_win_rate_min");
  const oddsMin = readActiveParam("odds_min");
  const oddsMax = readActiveParam("odds_max");
  const probThreshold = readActiveParam("prob_threshold");
  const minEv = readActiveParam("min_ev");
  const windowSizeFromParams = readActiveParam("window_size");
  const activeParamsEconomicallyMeaningful =
    homeWinRateMin !== null &&
    homeWinRateMin >= 0 &&
    homeWinRateMin <= 1 &&
    oddsMin !== null &&
    oddsMin >= 1 &&
    oddsMin <= 20 &&
    oddsMax !== null &&
    oddsMax >= 1 &&
    oddsMax <= 20 &&
    oddsMax >= oddsMin &&
    probThreshold !== null &&
    probThreshold >= 0 &&
    probThreshold <= 1 &&
    minEv !== null;
  const strategyStatusTrustworthy = activeParamsComplete && activeParamsEconomicallyMeaningful;

  const overallAccuracyValue = pickNumber(summaryStats.overall_accuracy, calibrationMetrics.actualWinPct);
  const overallAccuracyPct = formatPercentFromMaybeRatio(overallAccuracyValue, 2);

  const windowSize =
    windowSizeFromParams || dashboardState?.window_size || windowInfo.size || calibrationMetrics.windowSize || summaryStats.total_games || 200;

  const windowStartLabel = dashboardState?.window_start ?? windowInfo.start ?? summary?.window_start ?? "—";
  const windowEndLabel = dashboardState?.window_end ?? windowInfo.end ?? summary?.window_end ?? summaryAsOfDate ?? "—";

  const windowGamesLabel =
    pickPositiveNumber(windowInfo.games_count, summaryStats.total_games, calibrationMetrics.fittedGames, windowSize) ??
    windowSize;

  const activeFiltersEffective =
    dashboardState?.active_filters_text ??
    payload?.active_filters_effective ??
    strategyParams.active_filters ??
    "No active filters.";

  const paramsUsedLabel =
    dashboardState?.params_used_label ??
    payload?.params_used_label ??
    strategyParams.params_used_label ??
    "Historical";

  const paramsSourceLabel = dashboardState?.params_source_label ?? "strategy_params.json";
  const dataConsistencyStatus = dashboardState?.data_consistency_status ?? "ok";
  const dataConsistencyIssues = useMemo(
    () => dashboardState?.data_consistency_issues ?? [],
    [dashboardState?.data_consistency_issues],
  );
  const normalizedConsistencyIssues = dataConsistencyIssues.map((issue) => issue.toLowerCase());
  const hasBetLogStaleIssue = normalizedConsistencyIssues.some(
    (issue) => issue.includes("bet_log") && issue.includes("stale"),
  );
  const hasNonBetLogConsistencyIssue = normalizedConsistencyIssues.some(
    (issue) => !(issue.includes("bet_log") && issue.includes("stale")),
  );
  const legacyBetLogOnlyIssue = hasBetLogStaleIssue && !hasNonBetLogConsistencyIssue;
  const strategyParamsParseStatus = dashboardState?.strategy_params_parse_status ?? "ok";
  const strategyParamsParseError = dashboardState?.strategy_params_parse_error;
  const fallbackUsed = Boolean(dashboardState?.fallback_used);

  const fallbackDetailsLabel =
    homeWinRateMin !== null && oddsMin !== null && oddsMax !== null && probThreshold !== null && minEv !== null
      ? `Fallback (HW ≥ ${fmtNumber(homeWinRateMin, 2)} • odds ${fmtNumber(oddsMin, 2)}–${fmtNumber(oddsMax, 2)} • p ≥ ${fmtNumber(probThreshold, 2)} • EV ≥ ${fmtNumber(minEv, 2)})`
      : "Fallback (threshold details unavailable)";
  const noBetModeActive = [dashboardState?.params_source_type, dashboardState?.params_used, dashboardState?.fallback_reason]
    .filter((value): value is string => typeof value === "string")
    .some((value) => value.toUpperCase().includes("NO_BET"));

  const paramsUsedDisplay = fallbackUsed
    ? noBetModeActive
      ? "No-bet mode active (stability gate)"
      : fallbackDetailsLabel
    : `Params used: ${dashboardState?.params_used ?? paramsUsedLabel}`;

  const oddsRangeLabel =
    oddsMin !== null && oddsMax !== null ? `${fmtNumber(oddsMin, 2)}–${fmtNumber(oddsMax, 2)}` : null;
  const activeFiltersUnavailableLabel =
    "Strategy filters unavailable (dashboard_state.json active_params missing or invalid)";

  const thresholdsLabel =
    strategyStatusTrustworthy
      ? `HW ≥ ${fmtNumber(homeWinRateMin as number, 2)} | odds ${fmtNumber(oddsMin as number, 2)}–${fmtNumber(oddsMax as number, 2)} | p ≥ ${fmtNumber(probThreshold as number, 2)} | EV ≥ ${fmtNumber(minEv as number, 2)}`
      : activeFiltersEffective;

  const activeFiltersDisplay = strategyStatusTrustworthy
    ? `${thresholdsLabel} | window ${windowSize} (${windowStartLabel} → ${windowEndLabel})`
    : `${activeFiltersUnavailableLabel} | source text: ${activeFiltersEffective}`;

  const localMatchedGamesRowsSorted = useMemo(() => {
    return [...localMatchedRowsDisplay].sort((a, b) => b.date.localeCompare(a.date));
  }, [localMatchedRowsDisplay]);

  const localMatchedWindowRange = useMemo(() => {
    if (localMatchedRowsDisplay.length === 0) return null;

    const parseDate = (value: string) => {
      const parsed = Date.parse(value);
      return Number.isNaN(parsed) ? null : parsed;
    };

    let minDate = localMatchedRowsDisplay[0].date;
    let maxDate = localMatchedRowsDisplay[0].date;
    let minTime = parseDate(minDate);
    let maxTime = parseDate(maxDate);

    for (const row of localMatchedRowsDisplay.slice(1)) {
      const current = row.date;
      const currentTime = parseDate(current);

      if (currentTime !== null && minTime !== null) {
        if (currentTime < minTime) {
          minTime = currentTime;
          minDate = current;
        }
      } else if (current < minDate) {
        minDate = current;
        minTime = parseDate(current);
      }

      if (currentTime !== null && maxTime !== null) {
        if (currentTime > maxTime) {
          maxTime = currentTime;
          maxDate = current;
        }
      } else if (current > maxDate) {
        maxDate = current;
        maxTime = parseDate(current);
      }
    }

    return { start: minDate, end: maxDate };
  }, [localMatchedRowsDisplay]);

  const localMatchedCountDisplay = localMatchedTotalCount;
  const localMatchedWinsDisplay = localMatchedWins;
  const localMatchedProfitSumDisplay = localMatchedProfitSum;

  const localMatchedWinRateBase = localMatchedRowsCount > 0 ? localMatchedRowsCount : localMatchedTotalCount;
  const localMatchedWinRateDisplay = localMatchedWinRateBase > 0 ? (localMatchedWinsDisplay / localMatchedWinRateBase) * 100 : 0;

  const localMatchedWindowStartLabel = localMatchedWindowRange?.start ?? windowStartLabel;
  const localMatchedWindowEndLabel = localMatchedWindowRange?.end ?? windowEndLabel;

  const localMatchedCountBreakdown =
    localMatchedRowsCount > 0 && localMatchedTotalCount > localMatchedRowsCount
      ? `${localMatchedRowsCount} of ${localMatchedTotalCount}`
      : `${localMatchedTotalCount}`;

  const localMatchedDisplayWins = localMatchedRowsDisplay.filter((row) => row.win === 1).length;
  const localMatchedDisplayProfitSum = localMatchedRowsDisplay.reduce((acc, row) => acc + (row.pnl ?? 0), 0);
  const localMatchedDisplayCount = localMatchedRowsDisplay.length;

  const simulatedBankroll = START_BANKROLL_SIM + localMatchedProfitSumDisplay;

  // ✅ FIX: ROI should never depend on strategySummary.profitMetricsAvailable (often missing).
  // Prefer summary.kpis.roi_pct; fallback to summary.strategy.roiPct; fallback to 0.
  const strategyRoiPctValue =
    typeof summary?.kpis?.roi_pct === "number" && Number.isFinite(summary.kpis.roi_pct)
      ? summary.kpis.roi_pct
      : typeof summary?.strategy?.roiPct === "number" && Number.isFinite(summary.strategy.roiPct)
        ? summary.strategy.roiPct
        : 0;

  const strategyRoiDisplay = fmtPercent(strategyRoiPctValue, 2);

  const strategySharpeValue =
    typeof strategySummary.sharpeStyle === "number" && Number.isFinite(strategySummary.sharpeStyle)
      ? strategySummary.sharpeStyle
      : null;

  const strategySharpeDisplay = strategySharpeValue !== null ? fmtNumber(strategySharpeValue, 2) : "—";

  const strategyMaxDrawdownDisplay =
    kpis.max_drawdown_eur !== null ? fmtCurrencyEUR(kpis.max_drawdown_eur as number, 0) : "—";

  const liveStrategyLabel = "none";
  const formatSourceLabel = (value: string | null | undefined) => {
    if (!value || typeof value !== "string") return "—";
    const normalized = value.trim();
    if (!normalized) return "—";
    const parts = normalized.split(/[\\/]/).filter(Boolean);
    return parts.length > 0 ? parts[parts.length - 1] : normalized;
  };
  const historicalFilterSource =
    dashboardState?.metrics_snapshot_source_file ??
    dashboardState?.strategy_params_source_file ??
    dashboardState?.params_source_label ??
    "strategy_params.json";
  const historicalFilterSourceDisplay = formatSourceLabel(historicalFilterSource);
  const paramsSourceDisplay = formatSourceLabel(paramsSourceLabel);

  const currentDate = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const availableTodayGames = useMemo(() => todayGames?.games ?? [], [todayGames?.games]);
  const liveCandidateRows = useMemo(() => todayGames?.qualifying_bets ?? [], [todayGames?.qualifying_bets]);
  const evExceptionProfitability = todayGames?.ev_exception_profitability ?? null;
  const upcomingGameChecks = todayGames?.upcoming_game_checks?.rows ?? [];
  const upcomingGameChecksBasis = todayGames?.upcoming_game_checks?.basis ?? null;
  const upcomingHistoryMeta = upcomingGameChecksBasis?.history_meta as Record<string, unknown> | undefined;
  const positiveStakeRows = useMemo(
    () => liveCandidateRows.filter((row) => (parseNumeric(row.stake_eur ?? "") ?? 0) > 0),
    [liveCandidateRows],
  );
  const availableGamesDate = todayGames?.as_of_date ?? currentDate;
  const liveCandidateCount = liveCandidateRows.length;
  const positiveStakeCount = positiveStakeRows.length;

  const todayShortlist = payload?.last_run?.today_shortlist;
  const todayQualifyingGamesCountRaw = payload?.last_run?.qualifying_games_today;
  const todayQualifyingGamesCount =
    liveCandidateCount > 0
      ? liveCandidateCount
      : typeof todayQualifyingGamesCountRaw === "number" && Number.isFinite(todayQualifyingGamesCountRaw)
      ? todayQualifyingGamesCountRaw
      : Array.isArray(todayShortlist)
        ? todayShortlist.length
        : 0;
  const hasTodayQualifyingGames = todayQualifyingGamesCount > 0;

  const dashboardDataAgeDays = useMemo(() => {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(summaryAsOfDate) || !/^\d{4}-\d{2}-\d{2}$/.test(currentDate)) {
      return null;
    }

    const asOfTime = Date.parse(`${summaryAsOfDate}T00:00:00Z`);
    const currentTime = Date.parse(`${currentDate}T00:00:00Z`);
    if (!Number.isFinite(asOfTime) || !Number.isFinite(currentTime)) return null;

    return Math.max(0, Math.round((currentTime - asOfTime) / 86_400_000));
  }, [currentDate, summaryAsOfDate]);

  const liveHomeWinRateRows = useMemo(
    () =>
      availableTodayGames
        .filter((game) => typeof game.home_win_rate === "number" && Number.isFinite(game.home_win_rate))
        .map((game) => ({
          team: game.home_team ?? "—",
          homeWinRate: game.home_win_rate ?? 0,
          homeWins: game.home_wins ?? "—",
          homeGames: game.home_games ?? "—",
          windowGames: game.last20_games ?? "—",
        })),
    [availableTodayGames],
  );

  const homeWinRatesDiagnosticRows = useMemo(() => {
    if (homeWinRatesWindow.length > 0) {
      return [...homeWinRatesWindow]
        .filter((row) => row.homeWinRate > 0.5)
        .sort((a, b) => b.homeWinRate - a.homeWinRate);
    }
    if (liveHomeWinRateRows.length > 0) {
      return liveHomeWinRateRows;
    }
    return [...homeWinRatesLast20]
      .filter((row) => row.homeWinRate > 0.5)
      .sort((a, b) => b.homeWinRate - a.homeWinRate)
      .map((row) => ({
        team: row.team,
        homeWinRate: row.homeWinRate,
        homeWins: row.homeWins,
        homeGames: row.totalHomeGames,
        windowGames: row.totalLast20Games,
      }));
  }, [homeWinRatesLast20, homeWinRatesWindow, liveHomeWinRateRows]);

  const actualBetsRowsSorted = useMemo(
    () => [...actualBetsRows].sort((a, b) => `${b.bet_date}${b.bet_id}`.localeCompare(`${a.bet_date}${a.bet_id}`)),
    [actualBetsRows],
  );

  const todayDecisionContext = useMemo(() => {
    const evSummary = evExceptionProfitability?.summary ?? null;
    const priceAdjusted = evExceptionProfitability?.price_adjusted ?? null;
    const broadHistoricalSupport =
      typeof evSummary?.roi_pct === "number" && evSummary.roi_pct > 0 && typeof evSummary?.n === "number" && evSummary.n > 0;

    const candidates: TodayDecisionCandidate[] = liveCandidateRows.map((row) => {
      const odds1 = readRecordNumber(row, "odds_1");
      const probUsed = readRecordNumber(row, "prob_used");
      const kelly = readRecordNumber(row, "kelly_full");
      const stake = readRecordNumber(row, "stake_eur");
      const breakEvenProbability = odds1 && odds1 > 0 ? 1 / odds1 : null;
      const edgeVsBreakEven =
        probUsed !== null && breakEvenProbability !== null ? probUsed - breakEvenProbability : null;
      const liveEv =
        readRecordNumber(row, "EV_€_per_100") ??
        readRecordNumber(row, "EV_live_€_per_100") ??
        readRecordNumber(row, "EV_live_€_per100");

      return {
        game: `${readRecordString(row, "home_team") || "—"} vs ${readRecordString(row, "away_team") || "—"}`,
        home_team: readRecordString(row, "home_team"),
        away_team: readRecordString(row, "away_team"),
        odds_1: odds1,
        odds_2: readRecordNumber(row, "odds_2"),
        raw_probability: readRecordNumber(row, "home_team_prob"),
        prob_used: probUsed,
        live_ev_per_100: liveEv,
        kelly,
        stake,
        blocked_by: readRecordString(row, "blocked_by"),
        canonical_signal: isCanonicalSignal(row.canonical_signal),
        candidate_type: readRecordString(row, "stage2_candidate_type") || readRecordString(row, "candidate_type") || "live_candidate",
        break_even_probability: breakEvenProbability,
        edge_vs_break_even: edgeVsBreakEven,
        current_price_supported:
          edgeVsBreakEven !== null &&
          edgeVsBreakEven >= 0 &&
          kelly !== null &&
          kelly > 0 &&
          (liveEv === null || liveEv >= 0),
      };
    });

    const anyCandidateNegativeCurrentPrice = candidates.some(
      (candidate) =>
        candidate.current_price_supported === false &&
        (candidate.edge_vs_break_even === null || candidate.edge_vs_break_even < 0 || (candidate.kelly ?? 0) <= 0),
    );
    const canonicalCount = todayGames?.canonical_model_signals?.canonical_count ?? 0;
    const canonicalActive =
      canonicalCount > 0 ||
      candidates.some((candidate) => candidate.canonical_signal) ||
      (todayGames?.canonical_model_signals?.canonical ?? []).some((row) => isCanonicalSignal(row.canonical_signal ?? true));
    const engineState = todayGames?.engine_state ?? todayGames?.canonical_model_signals?.engine_state ?? null;
    const positiveStakeBets = candidates.filter((candidate) => (candidate.stake ?? 0) > 0);
    const localProfitabilityCases = todayGames?.local_profitability_rule?.cases ?? [];
    const localProfitabilityCase =
      localProfitabilityCases.find((ruleCase) => ruleCase.agent_label === "discretionary_local_profitability_confirmed") ??
      localProfitabilityCases.find((ruleCase) => ruleCase.local_profitable_candidate) ??
      localProfitabilityCases[0] ??
      null;
    const localProfitabilityLabel =
      typeof localProfitabilityCase?.agent_label === "string" ? localProfitabilityCase.agent_label : null;
    const localProfitabilityDecision =
      typeof localProfitabilityCase?.agent_decision === "string" ? localProfitabilityCase.agent_decision : null;
    const localProfitableCandidate = Boolean(localProfitabilityCase?.local_profitable_candidate);
    const robustStabilityPassed = Boolean(localProfitabilityCase?.robust_stability_passed);
    const historicalRoiAttackStatus =
      typeof localProfitabilityCase?.historical_roi_attack_status === "string"
        ? localProfitabilityCase.historical_roi_attack_status
        : null;
    const latestManualBet = actualBetsRowsSorted[0] ?? null;
    const manualSettled = actualBetsRows.filter((row) => ["Won", "Lost", "Void", "Cashout"].includes(row.status));
    const manualStake = actualBetsRows.reduce((acc, row) => acc + (parseNumeric(row.stake_eur) ?? 0), 0);
    const manualPnl = actualBetsRows.reduce((acc, row) => acc + (parseNumeric(row.pnl_eur) ?? 0), 0);

    return {
      run_date: todayGames?.as_of_date ?? currentDate,
      engine_state: engineState,
      live_strategy: liveStrategyLabel,
      canonical_model_signals: todayGames?.canonical_model_signals ?? null,
      games_available: availableTodayGames.length,
      live_candidates: candidates,
      positive_stake_bets: positiveStakeBets,
      setup_profitability_summary: todayGames?.setup_profitability?.summary ?? null,
      ev_exception_summary: evExceptionProfitability
        ? {
            label: evExceptionProfitability.label ?? null,
            classification: evExceptionProfitability.classification ?? "historical_support_only",
            is_betting_signal: evExceptionProfitability.is_betting_signal ?? false,
            recommendation_label: evExceptionProfitability.recommendation_label ?? "watch-only",
            criteria: evExceptionProfitability.criteria ?? null,
            summary: evSummary,
            price_adjusted: priceAdjusted,
        }
        : null,
      historical_roi_attack_scans: todayGames?.historical_roi_attack_scans ?? [],
      local_profitability_rule: todayGames?.local_profitability_rule ?? null,
      local_profitability_case: localProfitabilityCase,
      local_strategy_evaluation_window: todayGames?.local_strategy_evaluation_window ?? null,
      price_adjusted_warning: evExceptionProfitability?.warning ?? null,
      hwr_source_label:
        evExceptionProfitability?.price_adjusted?.hwr_source_label ??
        availableTodayGames.find((game) => game.hwr_source_label)?.hwr_source_label ??
        null,
      hwr_window_label:
        evExceptionProfitability?.price_adjusted?.hwr_window_label ??
        availableTodayGames.find((game) => game.hwr_window_label)?.hwr_window_label ??
        null,
      manual_actual_bets_summary: {
        total: actualBetsRows.length,
        settled: manualSettled.length,
        pending: actualBetsRows.filter((row) => row.status === "Pending").length,
        stake_eur: manualStake,
        pnl_eur: manualPnl,
        latest: latestManualBet
          ? {
              bet_date: latestManualBet.bet_date,
              game: `${latestManualBet.away_team} @ ${latestManualBet.home_team}`,
              selection: latestManualBet.selection,
              odds: latestManualBet.odds,
              stake_eur: latestManualBet.stake_eur,
              pnl_eur: latestManualBet.pnl_eur,
              status: latestManualBet.status,
              note: latestManualBet.user_note || latestManualBet.model_note || "",
            }
          : null,
      },
      decision_labels: {
        main_decision:
          engineState === "NO_BET" || positiveStakeBets.length === 0 || !canonicalActive ? "NO_BET" : "REVIEW_REQUIRED",
        canonical: canonicalActive,
        canonical_label: canonicalActive ? "Canonical signal present" : "Canonical: none",
        setup_profitability: broadHistoricalSupport ? "historical support only" : "no current setup support",
        local_profitable_candidate: localProfitableCandidate,
        local_profitability_rule_label: localProfitabilityLabel,
        local_profitability_rule_decision: localProfitabilityDecision,
        robust_stability: robustStabilityPassed ? "PASSED" : localProfitableCandidate ? "FAILED" : "not_triggered",
        historical_profitability_check:
          localProfitableCandidate && historicalRoiAttackStatus === "supported_discretionary_only"
            ? "PASSED"
            : localProfitableCandidate
              ? "FAILED"
              : "not_triggered",
        near_miss: Boolean(broadHistoricalSupport && anyCandidateNegativeCurrentPrice),
        vibe_live_watch: Boolean(broadHistoricalSupport && anyCandidateNegativeCurrentPrice),
        no_bet_reason:
          localProfitabilityLabel === "profitable_local_candidate_but_historical_rejected"
            ? "profitable local setup matched, but the repeatable Historical ROI Attack scanner rejected it"
            : localProfitabilityLabel === "discretionary_local_profitability_confirmed"
              ? ""
              : candidates.length > 1
                ? "no canonical pregame bets; game-specific watch-only reasons are shown separately"
                : broadHistoricalSupport && anyCandidateNegativeCurrentPrice
                  ? "negative current EV and/or negative Kelly despite broad historical support"
                : engineState === "NO_BET" || positiveStakeBets.length === 0
                  ? "no positive-stake canonical signal"
                  : "",
        steadivus_note:
          engineState === "NO_BET" || positiveStakeBets.length === 0
            ? "Good skip / no forced action."
            : "Follow canonical staking discipline only.",
      },
    };
  }, [
    actualBetsRows,
    actualBetsRowsSorted,
    availableTodayGames,
    currentDate,
    evExceptionProfitability,
    liveCandidateRows,
    todayGames,
  ]);

  const agentDashboardContext = useMemo(
    () => ({
      current_date: currentDate,
      as_of_date: summaryAsOfDate,
      dashboard_data_age_days: dashboardDataAgeDays,
      window_end: windowEndLabel,
      window_size: windowSize,
      active_filters: activeFiltersDisplay,
      data_consistency_status: dataConsistencyStatus,
      data_consistency_issues: dataConsistencyIssues,
      today_qualifying_games_count: todayQualifyingGamesCount,
      has_today_qualifying_games: hasTodayQualifyingGames,
      available_games_date: todayGames?.as_of_date ?? null,
      available_games_count: availableTodayGames.length,
      available_games: availableTodayGames,
      qualifying_bets: liveCandidateRows,
      positive_stake_bets_count: positiveStakeCount,
      engine_state: todayGames?.engine_state ?? todayGames?.canonical_model_signals?.engine_state ?? null,
      canonical_model_signals: todayGames?.canonical_model_signals ?? null,
      today_decision_context: todayDecisionContext,
      setup_profitability: todayGames?.setup_profitability ?? null,
      ev_exception_profitability: todayGames?.ev_exception_profitability ?? null,
      historical_roi_attack_scans: todayGames?.historical_roi_attack_scans ?? [],
      local_profitability_rule: todayGames?.local_profitability_rule ?? null,
      local_strategy_evaluation_window: todayGames?.local_strategy_evaluation_window ?? null,
      sources: dashboardState?.sources ?? null,
      summary_stats: summaryStats,
      strategy_summary: strategySummary,
      kpis,
      local_matched_games_count: localMatchedDisplayCount,
      manual_actual_bets_count: actualBetsRows.length,
      latest_manual_bet: actualBetsRowsSorted[0] ?? null,
    }),
    [
      activeFiltersDisplay,
      actualBetsRows.length,
      actualBetsRowsSorted,
      currentDate,
      dashboardState?.sources,
      dashboardDataAgeDays,
      dataConsistencyIssues,
      dataConsistencyStatus,
      hasTodayQualifyingGames,
      kpis,
      localMatchedDisplayCount,
      availableTodayGames,
      liveCandidateRows,
      positiveStakeCount,
      strategySummary,
      summaryAsOfDate,
      summaryStats,
      todayDecisionContext,
      todayGames,
      todayQualifyingGamesCount,
      windowEndLabel,
      windowSize,
    ],
  );

  useEffect(() => {
    if (!todayDecisionContext) return;
    console.info("Hoops today_decision_context", todayDecisionContext);
  }, [todayDecisionContext]);

  const handleAgentSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = agentInput.trim();
    if (!message || agentStatus === "sending") return;

    const nextMessages: AgentMessage[] = [...agentMessages, { role: "user", content: message }];
    setAgentMessages(nextMessages);
    setAgentInput("");
    setAgentStatus("sending");
    setAgentError(null);

    try {
      const response = await fetch(agentEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: message,
          capability: "read_only",
          context: agentDashboardContext,
          messages: nextMessages.slice(-8),
        }),
      });
      const rawResponse = await response.text();
      let data: AgentResponse = {};
      if (rawResponse.trim()) {
        try {
          data = JSON.parse(rawResponse) as AgentResponse;
        } catch {
          data = {
            error: `Agent endpoint returned non-JSON response (${response.status}).`,
            warnings: [rawResponse.slice(0, 240)],
          };
        }
      }
      if (!response.ok) {
        const warningDetail = data.warnings?.length ? ` ${data.warnings.join(" ")}` : "";
        throw new Error(data.error || data.answer || `Agent endpoint returned ${response.status}.${warningDetail}`);
      }
      const warningText = data.warnings?.length ? `\n\nWarnings:\n- ${data.warnings.join("\n- ")}` : "";
      setAgentMessages((current) => [
        ...current,
        { role: "assistant", content: `${data.answer || data.reply || "No agent response was returned."}${warningText}` },
      ]);
      setAgentStatus("idle");
    } catch (err) {
      const messageText = err instanceof Error ? err.message : "Agent request failed.";
      setAgentError(messageText);
      setAgentStatus("error");
      setAgentMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Agent backend unavailable: ${messageText}`,
        },
      ]);
    }
  };

  useEffect(() => {
    if (activeTab !== "agent-chat") return;
    const scrollEl = agentScrollRef.current;
    if (!scrollEl) return;
    scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  }, [activeTab, agentMessages, agentStatus]);

  useEffect(() => {
    if (activeTab !== "agent-chat" || agentStatus === "sending") return;
    agentInputRef.current?.focus();
  }, [activeTab, agentMessages.length, agentStatus]);

  const playTodayCase = todayDecisionContext?.local_profitability_case as Record<string, unknown> | null | undefined;
  const playTodayLabels = todayDecisionContext?.decision_labels as Record<string, unknown> | null | undefined;
  const localStrategyEvaluationWindow = todayGames?.local_strategy_evaluation_window ?? null;
  const localStrategyWindowGames =
    typeof localStrategyEvaluationWindow?.display_window_games === "number"
      ? localStrategyEvaluationWindow.display_window_games
      : null;
  const localStrategyWindowStart = localStrategyEvaluationWindow?.start || null;
  const localStrategyWindowEnd = localStrategyEvaluationWindow?.end || null;
  const localStrategyWindowSource = localStrategyEvaluationWindow?.source || localStrategyEvaluationWindow?.source_file || null;
  const evExceptionCriteria = evExceptionProfitability?.criteria ?? null;
  const playTodayFilterParams = {
    home_win_rate_min:
      typeof evExceptionCriteria?.home_win_rate_min === "number" ? evExceptionCriteria.home_win_rate_min : homeWinRateMin,
    odds_min: typeof evExceptionCriteria?.odds_min === "number" ? evExceptionCriteria.odds_min : oddsMin,
    odds_max: typeof evExceptionCriteria?.odds_max === "number" ? evExceptionCriteria.odds_max : oddsMax,
    prob_threshold:
      typeof evExceptionCriteria?.prob_threshold === "number" ? evExceptionCriteria.prob_threshold : probThreshold,
    min_ev: minEv,
  };
  const evExceptionCandidateChecks =
    (evExceptionProfitability?.per_candidate_checks as Array<Record<string, unknown>> | undefined) ??
    (evExceptionProfitability?.price_adjusted ? [evExceptionProfitability.price_adjusted as Record<string, unknown>] : []);
  const canonicalBetCount = evExceptionCandidateChecks.filter((check) => check.supports_play === true).length;
  const watchOnlyCount = evExceptionCandidateChecks.filter((check) => readRecordString(check, "classification") === "watch-only").length;
  const blockedGameSummary = evExceptionCandidateChecks
    .map((check) => {
      const game = readRecordString(check, "game");
      const blockedBy = readRecordString(check, "blocked_by");
      return game && blockedBy ? `${game.split(" vs ")[0]} blocked by ${blockedBy}` : "";
    })
    .filter(Boolean)
    .join("; ");
  const formatProbabilityValue = (value: number | null | undefined) =>
    typeof value === "number" && Number.isFinite(value) ? fmtPercent(value * 100, 1) : "—";
  const validHistoricalSetupChecks = evExceptionCandidateChecks.filter((check) => {
    const setupN = readRecordNumber(check, "setup_n");
    const setupRoi = readRecordNumber(check, "setup_roi_pct");
    const setupWinRate = readRecordNumber(check, "setup_win_rate");
    return setupN !== null && setupN > 0 && setupRoi !== null && setupWinRate !== null;
  });
  const historicalSetupDetail = validHistoricalSetupChecks
    .map((check) => {
      const game = readRecordString(check, "game") || "Game";
      return `${game}: n=${readRecordNumber(check, "setup_n") ?? 0}, WR ${formatProbabilityValue(readRecordNumber(check, "setup_win_rate"))}, ROI ${fmtPercent(readRecordNumber(check, "setup_roi_pct"), 1)}`;
    })
    .join("; ");
  const gameSpecificLabel = (check: Record<string, unknown>) => {
    const blockedBy = readRecordString(check, "blocked_by");
    const stage = readRecordString(check, "stage2_candidate_type");
    const currentEv = readRecordNumber(check, "current_ev_eur_per_100");
    if (blockedBy === "min_ev" || (currentEv !== null && currentEv < 0)) {
      return "price too short / negative EV despite historical setup";
    }
    if (stage === "LIVE_WATCH_ONLY" || blockedBy.includes("Prob")) {
      return "borderline local setup / LIVE_WATCH_ONLY / probability threshold conflict";
    }
    return readRecordString(check, "classification") || "watch-only";
  };
  const formatBlockReason = (value: string) => {
    if (value === "min_ev") return "current EV below threshold";
    if (value === "Prob<0.55") return "broader probability threshold";
    if (value === "missing_enriched_candidate_context") return "missing enriched candidate context";
    return value.replaceAll("EV<=0.00", "EV <= 0").replaceAll("Prob<", "probability < ");
  };
  const formatGameDateLabel = (value: string) => (value ? `Game date: ${value}` : "Game date: —");
  const numberToneClass = (value: number | null | undefined) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return "text-foreground";
    if (value > 0) return "text-green-300";
    if (value < 0) return "text-red-300";
    return "text-foreground";
  };
  const decisionBadgeClass = (decision: string) => {
    const normalized = decision.trim().toUpperCase();
    if (normalized.includes("LIVE_WATCH") || normalized.includes("WATCH")) {
      return "border-amber-400/50 bg-amber-500/15 text-amber-100";
    }
    if (normalized.includes("NO_BET")) {
      return "border-red-400/60 bg-red-500/15 text-red-200";
    }
    if (normalized.includes("BET") && !normalized.includes("NO_BET")) {
      return "border-green-400/50 bg-green-500/15 text-green-200";
    }
    return "border-border bg-muted/60 text-foreground";
  };
  const currentPlayDate = readRecordString(playTodayCase, "date");
  const currentPlayGame = readRecordString(playTodayCase, "game");
  const recentSetupHistoryCases: AgentLearningCase[] = ((evExceptionProfitability?.matches as Array<Record<string, unknown>> | undefined) ?? [])
    .filter((row) => {
      const date = readRecordString(row, "date");
      if (!date || (currentPlayDate && date >= currentPlayDate)) return false;
      return true;
    })
    .sort((a, b) => `${readRecordString(b, "date")} ${readRecordString(b, "home_team")} ${readRecordString(b, "away_team")}`.localeCompare(`${readRecordString(a, "date")} ${readRecordString(a, "home_team")} ${readRecordString(a, "away_team")}`))
    .slice(0, 8)
    .map((row) => {
      const win = readRecordNumber(row, "win");
      const pnl = readRecordNumber(row, "pnl_100") ?? readRecordNumber(row, "home_ml_pnl_100");
      const odds = readRecordNumber(row, "odds_1");
      const prob = readRecordNumber(row, "prob_used") ?? readRecordNumber(row, "home_team_prob");
      const game = `${readRecordString(row, "home_team") || "—"} vs ${readRecordString(row, "away_team") || "—"}`;
      return {
        date: readRecordString(row, "date"),
        game,
        home_team: readRecordString(row, "home_team"),
        away_team: readRecordString(row, "away_team"),
        agent_decision: win === 1 ? "WON" : win === 0 ? "LOST" : "SETTLED",
        agent_label: "historical_setup_match",
        lesson: `Comparable setup result: ${win === 1 ? "home ML won" : win === 0 ? "home ML lost" : "settled result"} at odds ${odds !== null ? fmtNumber(odds, 2) : "—"}; prob ${prob !== null ? fmtPercent(prob * 100, 1) : "—"}; flat P/L ${pnl !== null ? formatSigned(pnl, 0) : "—"}.`,
      };
    });
  const previousLearningCaseMap = new Map<string, AgentLearningCase>();
  [...agentLearningCases, ...recentSetupHistoryCases].forEach((learningCase) => {
    const sameDate = readRecordString(learningCase, "date") === currentPlayDate;
    const sameGame = readRecordString(learningCase, "game") === currentPlayGame;
    if (sameDate && sameGame) return;
    const key = `${readRecordString(learningCase, "date")}-${readRecordString(learningCase, "game")}-${readRecordString(learningCase, "agent_label")}`;
    previousLearningCaseMap.set(key, learningCase);
  });
  const previousLearningCases = Array.from(previousLearningCaseMap.values())
    .sort((a, b) => `${readRecordString(b, "date")} ${readRecordString(b, "game")}`.localeCompare(`${readRecordString(a, "date")} ${readRecordString(a, "game")}`));
  const playTodayLines = [
    {
      label: "Canonical model",
      value: readRecordString(playTodayCase, "canonical_decision") || readRecordString(playTodayLabels, "canonical_label").replace(/^Canonical:\s*/i, "") || "NO_BET",
      detail: `Stage 1 canonical signal: ${readRecordString(playTodayCase, "canonical_signal") || "false"}.`,
    },
    {
      label: "Engine state",
      value: String(todayDecisionContext?.engine_state ?? "NO_BET"),
      detail: "No canonical positive-stake bet was produced for the slate.",
    },
    {
      label: "Available games",
      value: String(availableTodayGames.length),
      detail: `${canonicalBetCount} canonical bets; ${watchOnlyCount} watch-only candidates.`,
    },
    {
      label: "Local setup status",
      value: canonicalBetCount > 0 ? "canonical_present" : "no canonical local bet",
      detail: "Game-specific setup checks are shown below.",
    },
    ...(validHistoricalSetupChecks.length > 0
      ? [
          {
            label: "Historical profitability",
            value: `${validHistoricalSetupChecks.length} game setup checks`,
            detail: historicalSetupDetail,
          },
        ]
      : []),
    {
      label: "Watchlist summary",
      value: watchOnlyCount > 0 ? `${watchOnlyCount} watch-only` : "none",
      detail: blockedGameSummary || "No game-specific blocks available.",
    },
    {
      label: "Slate label",
      value: canonicalBetCount > 0 ? "canonical rows require review" : "No canonical pregame bets",
      detail: `${watchOnlyCount} watch-only games; pregame action: ${canonicalBetCount > 0 ? "review canonical rows" : "SKIP"}.`,
    },
    {
      label: "Slate decision",
      value: canonicalBetCount > 0 ? "REVIEW_CANONICAL" : "SKIP",
      detail: `Canonical bets: ${canonicalBetCount}; watch-only games: ${watchOnlyCount}; pregame action: ${canonicalBetCount > 0 ? "review canonical rows" : "SKIP"}.`,
    },
  ];

  const actualSettledRows = actualBetsRows.filter((row) => ["Won", "Lost", "Void", "Cashout"].includes(row.status));
  const pendingBets = actualBetsRows.filter((row) => row.status === "Pending").length;
  const totalStake = actualBetsRows.reduce((acc, row) => acc + (parseNumeric(row.stake_eur) ?? 0), 0);
  const totalPnl = actualBetsRows.reduce((acc, row) => acc + (parseNumeric(row.pnl_eur) ?? 0), 0);
  const settledStakeForRoi = actualSettledRows.reduce((acc, row) => acc + (parseNumeric(row.stake_eur) ?? 0), 0);
  const roiPct = settledStakeForRoi > 0 ? (totalPnl / settledStakeForRoi) * 100 : 0;
  const wonCount = actualBetsRows.filter((row) => row.status === "Won").length;
  const lostCount = actualBetsRows.filter((row) => row.status === "Lost").length;
  const winRatePct = wonCount + lostCount > 0 ? (wonCount / (wonCount + lostCount)) * 100 : 0;


  if (isLoading && !payload && !dashboardState) {
    return (
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">INDEX LOADED</p>
          <h2 className="text-xl font-bold mb-3">Loading dashboard data...</h2>
          <p className="text-sm text-muted-foreground">
            Fetch status: {fetchStarted ? "started" : "pending"}. Waiting for dashboard payload files.
          </p>
        </div>
      </section>
    );
  }

  if (loadError && !payload && !dashboardState) {
    return (
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-3">Data unavailable</h2>
          <p className="text-sm text-red-500">
            {loadError}. The dashboard data could not be loaded. Ensure the data pipeline has been executed and that
            files under <code>public/data/</code> are present.
          </p>
        </div>
      </section>
    );
  }

  return (
    <>
      <section className="container mx-auto px-4 pt-6">
        <div className="glass-card p-6 space-y-2">
          <h1 className="text-3xl font-bold">Hoops Insight</h1>
          <p className="text-sm text-muted-foreground">Live games feed: {availableGamesDate}</p>
          <p className="text-sm text-muted-foreground">
            Historical settled snapshot: {summaryAsOfDate}
            {dashboardDataAgeDays !== null && dashboardDataAgeDays > 0 ? ` (${dashboardDataAgeDays} days old)` : ""}
          </p>
          <p className="text-sm text-muted-foreground">
            Settled dashboard stats window: {windowStartLabel} → {windowEndLabel} · {windowSize} games
          </p>
          <p className="text-sm text-muted-foreground">Live strategy: {liveStrategyLabel}</p>
          <p className="text-sm text-muted-foreground">Historical filter source: {historicalFilterSourceDisplay}</p>
        </div>
      </section>
      <section className="container mx-auto px-4 py-4">
        <div className="glass-card p-3">
          <div className="flex gap-2">
            <button className={`rounded px-3 py-1 text-sm ${activeTab === "play-today" ? "bg-primary text-primary-foreground" : "bg-muted"}`} onClick={() => setActiveTab("play-today")}>Play Today</button>
            <button className={`rounded px-3 py-1 text-sm ${activeTab === "overview" ? "bg-primary text-primary-foreground" : "bg-muted"}`} onClick={() => setActiveTab("overview")}>Overview</button>
            <button className={`rounded px-3 py-1 text-sm ${activeTab === "actual-bets" ? "bg-primary text-primary-foreground" : "bg-muted"}`} onClick={() => setActiveTab("actual-bets")}>Actual Bets</button>
            <button className={`rounded px-3 py-1 text-sm ${activeTab === "agent-chat" ? "bg-primary text-primary-foreground" : "bg-muted"}`} onClick={() => setActiveTab("agent-chat")}>Agent Chat</button>
          </div>
        </div>
      </section>
      {activeTab === "play-today" ? (
        <section className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h2 className="text-3xl font-bold">Play Today</h2>
            <p className="mt-1 text-base text-muted-foreground">
              Daily decision console for available games. Canonical model bets stay separate from local discretionary checks.
            </p>
          </div>

          <div className="glass-card p-6 md:p-7">
            <div className="mb-5 grid grid-cols-1 gap-3 md:grid-cols-2">
              <div className="readable-panel p-5">
                <div className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Play Today evaluation window</div>
                <div className="mt-1 text-lg font-semibold text-foreground">
                  {localStrategyWindowGames !== null ? `${localStrategyWindowGames} games` : "—"}
                </div>
                <div className="mt-2 text-sm leading-6 text-muted-foreground">
                  {localStrategyWindowStart && localStrategyWindowEnd
                    ? `${localStrategyWindowStart} → ${localStrategyWindowEnd}`
                    : localStrategyEvaluationWindow?.warning ?? "Script 11 local tail not available in current artifacts"}
                </div>
              </div>
              <div className="readable-panel p-5">
                <div className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Historical basis</div>
                <div className="mt-1 text-lg font-semibold text-foreground">
                  {localStrategyEvaluationWindow?.matches_script11_local_tail ? "Matches Script 11 LOCAL tail" : "—"}
                </div>
                <div className="mt-2 text-sm leading-6 text-muted-foreground">
                  {localStrategyWindowSource || "Script 11 local tail not available in current artifacts"}
                </div>
              </div>
            </div>
            <div className="readable-panel mb-5 p-5">
              <div className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Active filter params</div>
              <div className="mt-4 grid grid-cols-2 gap-4 text-base md:grid-cols-5">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">HWR min</div>
                  <div className="mt-1 text-lg font-bold text-foreground">
                    {typeof playTodayFilterParams.home_win_rate_min === "number"
                      ? fmtPercent(playTodayFilterParams.home_win_rate_min * 100, 0)
                      : "—"}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Odds</div>
                  <div className="mt-1 text-lg font-bold text-foreground">
                    {typeof playTodayFilterParams.odds_min === "number" && typeof playTodayFilterParams.odds_max === "number"
                      ? `${fmtNumber(playTodayFilterParams.odds_min, 2)}–${fmtNumber(playTodayFilterParams.odds_max, 2)}`
                      : "—"}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Probability min</div>
                  <div className="mt-1 text-lg font-bold text-foreground">
                    {typeof playTodayFilterParams.prob_threshold === "number"
                      ? fmtPercent(playTodayFilterParams.prob_threshold * 100, 0)
                      : "—"}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">EV filter</div>
                  <div className="mt-1 text-lg font-bold text-foreground">
                    {typeof playTodayFilterParams.min_ev === "number" ? `≥ ${fmtNumber(playTodayFilterParams.min_ev, 0)}` : "≥ 0"}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Historical scan</div>
                  <div className="mt-1 text-lg font-bold text-foreground">EV ignored</div>
                </div>
              </div>
            </div>
            {upcomingGameChecks.length > 0 ? (
              <div className="readable-panel mb-5 p-5">
                <div className="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
                  <div>
                    <div className="text-xl font-bold text-foreground">Upcoming game checks</div>
                    <div className="mt-1 text-sm leading-6 text-muted-foreground">
                      {upcomingGameChecksBasis?.label || "Historical comparable basis"} ·{" "}
                      {upcomingGameChecksBasis?.source || "all played games this season"} ·{" "}
                      {upcomingGameChecksBasis?.comparable_band || "±10% around current odds, HWR, and probability"}
                    </div>
                    <div className="mt-1 text-xs font-medium text-muted-foreground">
                      Source: {upcomingGameChecksBasis?.source_file || "—"}
                      {typeof upcomingHistoryMeta?.usable_rows === "number"
                        ? ` · usable rows ${upcomingHistoryMeta.usable_rows}`
                        : ""}
                    </div>
                  </div>
                  <div className="text-sm font-medium leading-6 text-muted-foreground">Diagnostic comparable-game profitability; not canonical selection.</div>
                </div>
                {upcomingHistoryMeta && upcomingHistoryMeta.usable_rows === 0 ? (
                  <div className="mt-3 rounded-md border border-amber-400/40 bg-amber-500/10 p-3 text-sm text-amber-100">
                    Comparable history did not run because the selected source file has no usable rows with
                    odds_1, prob_used, home_win_rate, and home_team_won. Regenerate or copy the enriched
                    combined_nba_predictions_acc source to populate this diagnostic.
                  </div>
                ) : null}
                {upcomingHistoryMeta?.warning ? (
                  <div className="mt-3 rounded-md border border-amber-400/40 bg-amber-500/10 p-3 text-sm text-amber-100">
                    {String(upcomingHistoryMeta.warning)}
                  </div>
                ) : null}

                <div className="mt-5 overflow-x-auto">
                  <table className="w-full border-separate border-spacing-0 text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs font-bold uppercase tracking-wide text-muted-foreground">
                        <th className="border-b border-border/80 py-3 pr-5">Game</th>
                        <th className="border-b border-border/80 py-3 pr-5 text-right">Home odds</th>
                        <th className="border-b border-border/80 py-3 pr-5 text-right">HWR</th>
                        <th className="border-b border-border/80 py-3 pr-5 text-right">Prob</th>
                        <th className="border-b border-border/80 py-3 pr-5 text-right">EV /100</th>
                        <th className="border-b border-border/80 py-3 pr-5">Comparable history</th>
                        <th className="border-b border-border/80 py-3 pr-5">Decision</th>
                      </tr>
                    </thead>
                    <tbody>
                      {upcomingGameChecks.map((check) => {
                        const evValue = readRecordNumber(check, "ev_eur_per_100");
                        const roiValue = readRecordNumber(check, "roi_pct");
                        const decision = readRecordString(check, "decision") || "—";
                        return (
                          <tr key={`${readRecordString(check, "date")}-${readRecordString(check, "game")}-upcoming-check`} className="border-b border-border/50 odd:bg-background/10 even:bg-muted/20">
                            <td className="border-b border-border/50 py-4 pr-5 align-top font-semibold text-foreground">
                              <div className="text-base">{readRecordString(check, "game") || "—"}</div>
                              <div className="mt-1 text-xs font-medium text-muted-foreground">
                                {formatGameDateLabel(readRecordString(check, "date"))}
                              </div>
                              <div className="mt-2 text-xs font-medium leading-5 text-muted-foreground">
                                Band: odds{" "}
                                {Array.isArray(check.odds_band)
                                  ? check.odds_band.map((v) => fmtNumber(typeof v === "number" ? v : null, 2)).join("–")
                                  : "—"}{" "}
                                · HWR{" "}
                                {readRecordString(check, "home_win_rate_filter_label") ||
                                  (Array.isArray(check.home_win_rate_band)
                                    ? check.home_win_rate_band.map((v) => formatProbabilityValue(typeof v === "number" ? v : null)).join("–")
                                    : "—")}{" "}
                                · prob{" "}
                                {Array.isArray(check.probability_band)
                                  ? check.probability_band.map((v) => formatProbabilityValue(typeof v === "number" ? v : null)).join("–")
                                  : "—"}
                              </div>
                            </td>
                            <td className="border-b border-border/50 py-4 pr-5 text-right align-top font-semibold tabular-nums text-foreground">{fmtNumber(readRecordNumber(check, "home_odds"), 2)}</td>
                            <td className="border-b border-border/50 py-4 pr-5 text-right align-top font-semibold tabular-nums text-foreground">{formatProbabilityValue(readRecordNumber(check, "home_win_rate"))}</td>
                            <td className="border-b border-border/50 py-4 pr-5 text-right align-top">
                              <div className="font-semibold tabular-nums text-foreground">{formatProbabilityValue(readRecordNumber(check, "probability"))}</div>
                              <div className="mt-1 text-xs font-medium text-muted-foreground">{readRecordString(check, "probability_source") || "—"}</div>
                            </td>
                            <td className={cn("border-b border-border/50 py-4 pr-5 text-right align-top font-bold tabular-nums", numberToneClass(evValue))}>
                              {formatSigned(evValue, 2)}
                            </td>
                            <td className="border-b border-border/50 py-4 pr-5 align-top">
                              <div className="font-semibold tabular-nums text-foreground">
                                n={readRecordNumber(check, "n") ?? 0}, WR {formatProbabilityValue(readRecordNumber(check, "win_rate"))}, ROI{" "}
                                <span className={numberToneClass(roiValue)}>{fmtPercent(roiValue, 1)}</span>
                              </div>
                              <div className="mt-1 text-xs font-medium leading-5 text-muted-foreground">
                                {readRecordNumber(check, "wins") ?? 0}W-{readRecordNumber(check, "losses") ?? 0}L · P/L{" "}
                                <span className={numberToneClass(readRecordNumber(check, "profit_100_flat"))}>
                                  {formatSigned(readRecordNumber(check, "profit_100_flat"), 0)}
                                </span>{" "}
                                · avg odds {fmtNumber(readRecordNumber(check, "avg_odds"), 2)}
                              </div>
                            </td>
                            <td className="border-b border-border/50 py-4 pr-5 align-top">
                              <span className={cn("inline-flex rounded-full border px-2.5 py-1 text-xs font-bold uppercase tracking-wide", decisionBadgeClass(decision))}>
                                {decision}
                              </span>
                              <div className="mt-2 text-xs font-medium leading-5 text-muted-foreground">{readRecordString(check, "reason") || "—"}</div>
                              {readRecordString(check, "blocked_by") ? (
                                <div className="mt-1 text-xs font-semibold text-muted-foreground">Blocked: {formatBlockReason(readRecordString(check, "blocked_by"))}</div>
                              ) : null}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
            <details className="mb-5 rounded-md border border-border/80 bg-muted/25 p-5">
              <summary className="cursor-pointer text-base font-bold text-foreground">
                Decision console and historical setup diagnostics
              </summary>
              <div className="mt-4">
                <div className="mb-5 rounded-md border border-border/80 bg-muted/45 p-5 font-mono text-base">
                  <div className="mb-4 text-sm uppercase tracking-wide text-muted-foreground">Decision Console</div>
                  <div className="space-y-4">
                    {playTodayLines.map((line) => (
                      <div key={line.label} className="grid gap-2 border-b border-border/60 pb-4 last:border-b-0 last:pb-0 md:grid-cols-[260px_1fr]">
                        <div className="font-semibold text-muted-foreground">{line.label}</div>
                        <div>
                          <div className="text-lg font-semibold text-foreground">{line.value}</div>
                          <div className="mt-1 text-sm leading-6 text-muted-foreground">{line.detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                  <StatCard
                    title="Available games"
                    value={`${availableTodayGames.length}`}
                    subtitle="Current slate rows"
                    icon={<Target className="w-6 h-6" />}
                  />
                  <StatCard
                    title="Canonical bets"
                    value={`${canonicalBetCount}`}
                    subtitle="Official positive-stake rows"
                    icon={<Activity className="w-6 h-6" />}
                  />
                  <StatCard
                    title="Watch-only"
                    value={`${watchOnlyCount}`}
                    subtitle="Game-specific checks below"
                    icon={<BarChart3 className="w-6 h-6" />}
                  />
                  <StatCard
                    title="Evaluation window"
                    value={localStrategyWindowGames !== null ? `${localStrategyWindowGames}` : "—"}
                    subtitle={localStrategyWindowStart && localStrategyWindowEnd ? `${localStrategyWindowStart} → ${localStrategyWindowEnd}` : "Script 11 local tail"}
                    icon={<TrendingUp className="w-6 h-6" />}
                  />
                </div>

                {evExceptionProfitability?.summary ? (
                  <div className="mt-5 rounded-md border border-border/80 bg-muted/40 p-5">
                    <div className="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
                      <div>
                        <div className="text-lg font-bold text-foreground">Historical profitability for today's game setups</div>
                        <div className="mt-1 text-sm leading-6 text-muted-foreground">
                          Each row uses that game's own setup: HWR floor, odds band, and probability floor. EV is ignored only to diagnose rows blocked by EV.
                        </div>
                      </div>
                      <div className="text-sm font-medium leading-6 text-muted-foreground">
                        Window {localStrategyWindowGames !== null ? `${localStrategyWindowGames} games · ` : ""}
                        {localStrategyWindowStart || evExceptionProfitability.summary.window_start || "—"} →{" "}
                        {localStrategyWindowEnd || evExceptionProfitability.summary.window_end || "—"}
                        {localStrategyEvaluationWindow?.matches_script11_local_tail ? " · Script 11 LOCAL tail" : ""}
                      </div>
                    </div>

                    {evExceptionCandidateChecks.length > 0 ? (
                      <div className="mt-5 overflow-x-auto">
                        <table className="w-full border-separate border-spacing-0 text-sm">
                          <thead>
                            <tr className="border-b border-border text-left text-xs font-bold uppercase tracking-wide text-muted-foreground">
                              <th className="border-b border-border/80 py-3 pr-5">Game</th>
                              <th className="border-b border-border/80 py-3 pr-5">Setup filter</th>
                              <th className="border-b border-border/80 py-3 pr-5 text-right">Setup n</th>
                              <th className="border-b border-border/80 py-3 pr-5">Setup WR / ROI</th>
                              <th className="border-b border-border/80 py-3 pr-5 text-right">Price-only n</th>
                              <th className="border-b border-border/80 py-3 pr-5">Price-only WR / ROI</th>
                              <th className="border-b border-border/80 py-3 pr-5">Current prob / EV</th>
                              <th className="border-b border-border/80 py-3 pr-5">Decision / stage</th>
                            </tr>
                          </thead>
                          <tbody>
                            {evExceptionCandidateChecks.map((check) => (
                              <tr key={`${readRecordString(check, "date")}-${readRecordString(check, "game")}`} className="odd:bg-background/10 even:bg-muted/20">
                                <td className="border-b border-border/50 py-4 pr-5 align-top text-base font-semibold text-foreground">{readRecordString(check, "game") || "—"}</td>
                                <td className="border-b border-border/50 py-4 pr-5 align-top font-medium text-foreground">
                                  HWR ≥ {formatProbabilityValue(readRecordNumber(check, "setup_hwr_min"))}
                                  <div className="mt-1 text-xs font-medium leading-5 text-muted-foreground">
                                    Odds {Array.isArray(check.setup_odds_band)
                                      ? check.setup_odds_band.map((v) => fmtNumber(v, 2)).join("–")
                                      : "—"} · prob ≥ {formatProbabilityValue(readRecordNumber(check, "setup_prob_min"))}
                                  </div>
                                </td>
                                <td className="border-b border-border/50 py-4 pr-5 text-right align-top font-semibold tabular-nums text-foreground">{readRecordNumber(check, "setup_n") ?? 0}</td>
                                <td className="border-b border-border/50 py-4 pr-5 align-top">
                                  <span className="font-semibold tabular-nums text-foreground">{formatProbabilityValue(readRecordNumber(check, "setup_win_rate"))}</span>{" "}
                                  / <span className={cn("font-semibold tabular-nums", numberToneClass(readRecordNumber(check, "setup_roi_pct")))}>{fmtPercent(readRecordNumber(check, "setup_roi_pct"), 1)}</span>
                                  <div className="mt-1 text-xs font-medium leading-5 text-muted-foreground">
                                    P/L {formatSigned(readRecordNumber(check, "setup_profit_100_flat"), 0)}
                                  </div>
                                </td>
                                <td className="border-b border-border/50 py-4 pr-5 text-right align-top font-semibold tabular-nums text-foreground">{readRecordNumber(check, "n") ?? 0}</td>
                                <td className="border-b border-border/50 py-4 pr-5 align-top">
                                  <span className="font-semibold tabular-nums text-foreground">{formatProbabilityValue(readRecordNumber(check, "win_rate"))}</span>{" "}
                                  / <span className={cn("font-semibold tabular-nums", numberToneClass(readRecordNumber(check, "roi_pct")))}>{fmtPercent(readRecordNumber(check, "roi_pct"), 1)}</span>
                                  <div className="mt-1 text-xs font-medium leading-5 text-muted-foreground">
                                    odds {fmtNumber(readRecordNumber(check, "current_odds"), 2)} · BE{" "}
                                    {formatProbabilityValue(readRecordNumber(check, "break_even_probability"))}
                                  </div>
                                </td>
                                <td className="border-b border-border/50 py-4 pr-5 align-top">
                                  <span className="font-semibold tabular-nums text-foreground">{formatProbabilityValue(readRecordNumber(check, "current_prob_used"))}</span>{" "}
                                  / <span className={cn("font-semibold tabular-nums", numberToneClass(readRecordNumber(check, "current_ev_eur_per_100")))}>{formatSigned(readRecordNumber(check, "current_ev_eur_per_100"), 2)}</span>
                                  {readRecordString(check, "borderline_probability_rounding") === "true" || check.borderline_probability_rounding === true ? (
                                    <div className="mt-1 text-xs font-medium leading-5 text-amber-200">Rounded display clears the 40.0% filter; raw probability remains borderline</div>
                                  ) : null}
                                  {readRecordString(check, "blocked_by") ? (
                                    <div className="mt-1 text-xs font-semibold leading-5 text-muted-foreground">Blocked: {formatBlockReason(readRecordString(check, "blocked_by"))}</div>
                                  ) : null}
                                  {readRecordString(check, "stage2_candidate_type") === "LIVE_WATCH_ONLY" ? (
                                    <div className="mt-1 text-xs font-medium leading-5 text-muted-foreground">
                                      LIVE_WATCH_ONLY: local setup layer differs from the broader watchlist block threshold
                                    </div>
                                  ) : null}
                                </td>
                                <td className="border-b border-border/50 py-4 pr-5 align-top">
                                  <span className={cn("inline-flex rounded-full border px-2.5 py-1 text-xs font-bold uppercase tracking-wide", decisionBadgeClass(check.supports_play === true ? "BET" : readRecordString(check, "stage2_candidate_type") || "NO_BET"))}>
                                    {check.supports_play === true ? "REVIEW" : "NO_BET"}
                                  </span>
                                  <div className="mt-2 text-xs font-semibold leading-5 text-muted-foreground">
                                    {readRecordString(check, "stage2_candidate_type") || readRecordString(check, "classification") || "—"}
                                  </div>
                                  <div className="mt-1 text-xs font-medium leading-5 text-muted-foreground">
                                    Stake: {check.supports_play === true ? "review" : "none pregame"}
                                  </div>
                                  <div className="mt-1 text-xs font-medium leading-5 text-muted-foreground">
                                    Label: {gameSpecificLabel(check)}
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </details>

            <div className="mt-5 rounded-md border border-amber-400/40 bg-amber-500/10 p-4 text-base leading-7 text-amber-100">
              Historical setup profitability is diagnostic only. It does not create a canonical bet unless the
              Robust++++ stability layer and repeatable scanner verification both pass.
            </div>

            <details className="mt-5 rounded-md border border-border p-4">
              <summary className="cursor-pointer text-base font-medium text-foreground">
                Previous games / learning history ({previousLearningCases.length})
              </summary>
              {previousLearningCases.length > 0 ? (
                <div className="mt-4 space-y-3">
                  {previousLearningCases.map((learningCase) => (
                    <div
                      key={`${readRecordString(learningCase, "date")}-${readRecordString(learningCase, "game")}-${readRecordString(learningCase, "agent_label")}`}
                      className="rounded-md border border-border/70 bg-muted/30 p-4 text-base"
                    >
                      <div className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
                        <div className="font-medium text-foreground">
                          {readRecordString(learningCase, "date") || "—"} · {readRecordString(learningCase, "game") || "—"}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {readRecordString(learningCase, "agent_decision") || "—"} · {readRecordString(learningCase, "agent_label") || "—"}
                        </div>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-muted-foreground">
                        {readRecordString(learningCase, "lesson") || readRecordString(learningCase, "reason") || "—"}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-base text-muted-foreground">
                  No previous learning cases yet. New cases will stay here after future dashboard generations.
                </p>
              )}
            </details>
          </div>
        </section>
      ) : activeTab === "agent-chat" ? (
        <section className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h2 className="text-2xl font-bold">Agent Chat</h2>
            <p className="text-sm text-muted-foreground">
              Backend endpoint: <code>{agentEndpoint}</code>. The agent only receives a compact dashboard context and the
              last few chat messages.
            </p>
          </div>

          <div className="glass-card p-6">
            <div className="mb-4 rounded-lg border border-amber-400/40 bg-amber-500/10 p-3 text-xs text-amber-100">
              Use this as a dashboard explanation assistant, not as autonomous betting approval. Refresh the Basketball_prediction
              data first before relying on live recommendations.
            </div>
            <details className="mb-4 rounded-lg border border-border p-3 text-xs">
              <summary className="cursor-pointer font-medium text-foreground">Agent daily decision context</summary>
              <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-muted p-3 text-muted-foreground">
                {JSON.stringify(todayDecisionContext, null, 2)}
              </pre>
            </details>
            <div ref={agentScrollRef} className="mb-4 max-h-96 space-y-3 overflow-y-auto rounded-lg border border-border p-4">
              {agentMessages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`rounded-lg p-3 text-sm ${
                    message.role === "user" ? "ml-auto bg-primary text-primary-foreground" : "mr-auto bg-muted"
                  } max-w-3xl`}
                >
                  <div className="mb-1 text-xs font-semibold uppercase tracking-wide opacity-70">
                    {message.role === "user" ? "You" : "Agent"}
                  </div>
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              ))}
            </div>
            {agentError && <p className="mb-3 text-sm text-red-400">{agentError}</p>}
            <form className="flex flex-col gap-3 md:flex-row" onSubmit={handleAgentSubmit}>
              <input
                ref={agentInputRef}
                className="min-h-11 flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={agentInput}
                onChange={(event) => setAgentInput(event.target.value)}
                placeholder="Ask about stale data, filters, ROI, matched games..."
                disabled={agentStatus === "sending"}
              />
              <button
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-60"
                type="submit"
                disabled={agentStatus === "sending" || !agentInput.trim()}
              >
                {agentStatus === "sending" ? "Sending..." : "Send"}
              </button>
            </form>
          </div>
        </section>
      ) : activeTab === "actual-bets" ? (
        <section className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h2 className="text-2xl font-bold">Actual Bets</h2>
            <p className="text-sm text-muted-foreground">Manual betting log from public/data/actual_bets_manual.csv only.</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <StatCard title="Total bets" value={`${actualBetsRows.length}`} icon={<Target className="w-6 h-6" />} />
            <StatCard title="Settled bets" value={`${actualSettledRows.length}`} subtitle={`Pending: ${pendingBets}`} icon={<Activity className="w-6 h-6" />} />
            <StatCard title="Total stake / PnL" value={fmtCurrencyEUR(totalStake, 2)} subtitle={`P/L: ${formatSigned(totalPnl)}`} icon={<TrendingUp className="w-6 h-6" />} />
            <StatCard title="ROI / Win rate" value={`${fmtPercent(roiPct, 2)} / ${fmtPercent(winRatePct, 2)}`} icon={<BarChart3 className="w-6 h-6" />} />
          </div>
          <div className="glass-card p-6 overflow-x-auto">
            {actualBetsRowsSorted.length === 0 ? (
              <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
                No manual bets logged yet. Add rows to public/data/actual_bets_manual.csv.
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead><tr className="text-left border-b border-border"><th className="py-2 pr-4">Bet date</th><th className="py-2 pr-4">Game</th><th className="py-2 pr-4">Selection</th><th className="py-2 pr-4">Market</th><th className="py-2 pr-4">Odds</th><th className="py-2 pr-4">Stake</th><th className="py-2 pr-4">PnL</th><th className="py-2 pr-4">Status</th><th className="py-2 pr-4">Note</th></tr></thead>
                <tbody>
                  {actualBetsRowsSorted.map((row) => {
                    const statusClass = row.status === "Won" ? "text-emerald-400" : row.status === "Lost" ? "text-red-400" : "text-muted-foreground";
                    return <tr key={row.bet_id} className="border-b border-border/50"><td className="py-2 pr-4">{row.bet_date}</td><td className="py-2 pr-4">{row.away_team} @ {row.home_team}</td><td className="py-2 pr-4">{row.selection}</td><td className="py-2 pr-4">{row.market}</td><td className="py-2 pr-4">{row.odds || "—"}</td><td className="py-2 pr-4">{row.stake_eur ? fmtCurrencyEUR(parseNumeric(row.stake_eur) ?? 0, 2) : "—"}</td><td className="py-2 pr-4">{row.pnl_eur ? formatSigned(parseNumeric(row.pnl_eur) ?? 0) : "—"}</td><td className={`py-2 pr-4 font-medium ${statusClass}`}>{row.status || "—"}</td><td className="py-2 pr-4">{row.user_note || row.model_note || "—"}</td></tr>;
                  })}
                </tbody>
              </table>
            )}
          </div>
        </section>
      ) : (
        <>

      {/* Today status */}
      <section className="container mx-auto px-4 py-6">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-3">Today Status</h2>
          <div className="space-y-4 text-sm">
            <div>
              <p className="text-foreground">
                {availableTodayGames.length} games available for {availableGamesDate}
              </p>
              <p className="text-muted-foreground">
                {liveCandidateCount} live candidate rows · {positiveStakeCount} positive-stake bets
              </p>
            </div>
            {availableTodayGames.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="py-2 pr-4">Game</th>
                      <th className="py-2 pr-4">Home Win Rate</th>
                      <th className="py-2 pr-4">Raw / Used Prob</th>
                      <th className="py-2 pr-4">Live EV</th>
                      <th className="py-2 pr-4">Odds</th>
                    </tr>
                  </thead>
                  <tbody>
                    {availableTodayGames.map((game) => (
                      <tr key={`${game.date}-${game.home_team}-${game.away_team}`} className="border-b border-border/50">
                        <td className="py-2 pr-4 font-medium">
                          {game.home_team} vs {game.away_team}
                          <div className="text-xs font-normal text-muted-foreground">
                            {formatGameDateLabel(game.date ?? "")}
                          </div>
                        </td>
                        <td className="py-2 pr-4">
                          {typeof game.home_win_rate === "number" ? fmtPercent(game.home_win_rate * 100, 1) : "—"}
                          {typeof game.home_wins === "number" && typeof game.home_games === "number"
                            ? ` (${game.home_wins}/${game.home_games})`
                            : ""}
                          {game.hwr_window_label ? (
                            <div className="text-xs text-muted-foreground">{game.hwr_window_label}</div>
                          ) : null}
                        </td>
                        <td className="py-2 pr-4">
                          {typeof game.home_team_prob === "number" ? fmtPercent(game.home_team_prob * 100, 1) : "—"}
                          {" / "}
                          {typeof game.prob_used === "number" ? fmtPercent(game.prob_used * 100, 1) : "—"}
                        </td>
                        <td className="py-2 pr-4">
                          {typeof game.ev_live_eur_per_100 === "number" ? fmtNumber(game.ev_live_eur_per_100, 2) : "—"}
                        </td>
                        <td className="py-2 pr-4">
                          {typeof game.home_odds === "number" ? fmtNumber(game.home_odds, 2) : "—"} /{" "}
                          {typeof game.away_odds === "number" ? fmtNumber(game.away_odds, 2) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-muted-foreground">No current games are available in today_games.json.</p>
            )}
            {liveCandidateRows.length > 0 ? (
              <div className="rounded-lg border border-border p-3">
                <p className="mb-2 font-medium text-foreground">Live candidates</p>
                <div className="space-y-2">
                  {liveCandidateRows.map((row) => (
                    <div key={`${row.date}-${row.home_team}-${row.away_team}`} className="text-muted-foreground">
                      <span className="text-foreground">
                        {row.home_team} vs {row.away_team}
                      </span>
                      {" · "}EV/100: {row["EV_€_per_100"] || "—"}
                      {" · "}prob used: {row.prob_used || "—"}
                      {" · "}Kelly: {row.kelly_full || "—"}
                      {" · "}Stake: {row.stake_eur || "—"}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground">No live candidate rows under the current live constraints.</p>
            )}
            {evExceptionProfitability?.summary ? (
              <div className="rounded-lg border border-border p-3">
                <p className="mb-1 font-medium text-foreground">EV exception historical profitability</p>
                <p className="mb-2 text-xs text-muted-foreground">
                  Broad historical support only · active setup without EV filter · window {evExceptionProfitability.summary.window_start || "—"} →{" "}
                  {evExceptionProfitability.summary.window_end || "—"}
                </p>
                {evExceptionProfitability.warning ? (
                  <div className="mb-3 rounded border border-amber-400/50 bg-amber-500/10 p-2 text-xs text-amber-200">
                    {evExceptionProfitability.warning}
                  </div>
                ) : null}
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Matches</p>
                    <p className="font-medium text-foreground">{evExceptionProfitability.summary.n ?? 0}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Wins / Win rate</p>
                    <p className="font-medium text-foreground">
                      {evExceptionProfitability.summary.wins ?? 0} /{" "}
                      {typeof evExceptionProfitability.summary.win_rate === "number"
                        ? fmtPercent(evExceptionProfitability.summary.win_rate * 100, 1)
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Flat €100 P/L</p>
                    <p className="font-medium text-foreground">
                      {formatSigned(evExceptionProfitability.summary.profit_100_flat ?? 0)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">ROI / Avg odds</p>
                    <p className="font-medium text-foreground">
                      {typeof evExceptionProfitability.summary.roi_pct === "number"
                        ? fmtPercent(evExceptionProfitability.summary.roi_pct, 1)
                        : "—"}{" "}
                      /{" "}
                      {typeof evExceptionProfitability.summary.avg_odds === "number"
                        ? fmtNumber(evExceptionProfitability.summary.avg_odds, 2)
                        : "—"}
                    </p>
                  </div>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  {evExceptionProfitability.note || "EV is ignored only for this diagnostic scan."}
                </p>
                {evExceptionProfitability.price_adjusted ? (
                  <div className="mt-3 rounded border border-border/70 p-3">
                    <p className="mb-2 text-sm font-medium text-foreground">Current-price check</p>
                    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Matches</p>
                        <p className="font-medium text-foreground">{evExceptionProfitability.price_adjusted.n ?? 0}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Win rate / BE</p>
                        <p className="font-medium text-foreground">
                          {typeof evExceptionProfitability.price_adjusted.win_rate === "number"
                            ? fmtPercent(evExceptionProfitability.price_adjusted.win_rate * 100, 1)
                            : "—"}{" "}
                          /{" "}
                          {typeof evExceptionProfitability.price_adjusted.break_even_probability === "number"
                            ? fmtPercent(evExceptionProfitability.price_adjusted.break_even_probability * 100, 1)
                            : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Flat €100 P/L</p>
                        <p className="font-medium text-foreground">
                          {formatSigned(evExceptionProfitability.price_adjusted.profit_100_flat ?? 0)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Classification</p>
                        <p className="font-medium text-foreground">
                          {evExceptionProfitability.price_adjusted.supports_play ? "historical support" : "watch-only"}
                        </p>
                      </div>
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground">
                      Odds band {evExceptionProfitability.price_adjusted.odds_band?.map((v) => typeof v === "number" ? fmtNumber(v, 2) : "—").join("–") || "—"} ·
                      prob band {evExceptionProfitability.price_adjusted.prob_used_band?.map((v) => typeof v === "number" ? fmtPercent(v * 100, 1) : "—").join("–") || "—"} ·
                      HWR source: {evExceptionProfitability.price_adjusted.hwr_source_file || "—"}
                    </p>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      </section>

      {/* Context / Assumptions */}
      <section className="container mx-auto px-4 py-6">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-3">Context / Assumptions</h2>
          {loadError && <p className="text-sm text-red-400 mb-3">Data unavailable: {loadError}{staleMessage ? ` (${staleMessage})` : ""}</p>}
          <div className="text-sm text-muted-foreground space-y-3">
            <div>
              <span className="font-medium text-foreground">Active Filters (effective)</span>
              <div className="text-foreground">{activeFiltersDisplay}</div>
            </div>
            <div className="text-foreground">Live strategy: {liveStrategyLabel}</div>
            <div className="text-foreground">Historical filter source: {historicalFilterSourceDisplay}</div>
            <div className="text-foreground">{paramsUsedDisplay}</div>
            {fallbackUsed && (
              <p className="text-xs text-amber-300">
                {noBetModeActive
                  ? "No-bet mode is active due to the stability gate; thresholds are shown for transparency."
                  : "Fallback historical filters are active because strategy parameters could not be loaded from the preferred source."}
              </p>
            )}
            {!activeParamsComplete && (
              <p className="text-xs text-amber-300">
                Warning: active_params is missing or incomplete in dashboard_state.json; strategy-filter and Today Status displays are unavailable.
              </p>
            )}
            {activeParamsComplete && !activeParamsEconomicallyMeaningful && (
              <p className="text-xs text-amber-300">
                Warning: active_params values are outside expected ranges; strategy-filter and Today Status displays are disabled.
              </p>
            )}
            <div className="text-foreground">Params source: {paramsSourceDisplay}</div>
            {strategyParamsParseStatus === "parse_error" && (
              <p className="text-xs text-red-300">
                strategy_params.json was found but could not be parsed. Defaults were applied. Error: {strategyParamsParseError ?? "unknown"}
              </p>
            )}
            {dataConsistencyStatus !== "ok" && (
              <div className="rounded border border-red-400/50 bg-red-500/10 p-2 text-xs text-red-200">
                <p className="font-semibold">
                  {legacyBetLogOnlyIssue
                    ? "Legacy bet_log source is stale versus snapshot."
                    : "Dashboard data sources are out of sync."}
                </p>
                <ul className="list-disc pl-4">
                  {dataConsistencyIssues.length > 0 ? (
                    dataConsistencyIssues.map((issue) => <li key={issue}>{issue}</li>)
                  ) : (
                    <li>combined_latest.csv and local_matched_games_latest.csv refer to different snapshots.</li>
                  )}
                </ul>
                {hasBetLogStaleIssue && (
                  <p className="mt-2">
                    Manual Actual Bets are tracked separately in <code>actual_bets_manual.csv</code>.
                  </p>
                )}
              </div>
            )}
            <p>Parameters are shown for transparency only. No live strategy selection or optimization is applied.</p>
            <p>Historical results and statistical summaries only; no future predictions are shown.</p>
          </div>
        </div>
      </section>

      {/* Performance overview */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Performance Overview</h2>
          <p className="text-sm text-muted-foreground">Source: {combinedSource} (played games only, windowed).</p>
        </div>
        <TooltipProvider delayDuration={100}>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              title={renderMetricTitle(
                "Overall Accuracy",
                <>
                  <p>
                    Percentage of games in the current window where the model correctly predicted the winning team.
                    Computed on played games only.
                  </p>
                  <p>Accuracy = correct predictions / total games in window.</p>
                </>,
              )}
              value={overallAccuracyPct}
              subtitle={<div>Window games: {windowGamesLabel}</div>}
              icon={<Target className="w-6 h-6" />}
            />

            <StatCard
              title={renderMetricTitle(
                "Calibration (Brier)",
                <>
                  <p>Brier Score measures how well predicted probabilities match actual outcomes.</p>
                  <p>It is the mean squared error between predicted probabilities and actual results (0 or 1).</p>
                  <p>Lower values indicate better calibrated probabilities.</p>
                  <p>Computed on the last 200 played games only.</p>
                </>,
              )}
              value={fmtNumber(calibrationMetrics.brierAfter, 3)}
              subtitle={`Before: ${fmtNumber(calibrationMetrics.brierBefore, 3)}`}
              icon={<BarChart3 className="w-6 h-6" />}
            />

            <StatCard
              title={renderMetricTitle(
                "LogLoss",
                <>
                  <p>
                    Log Loss evaluates probabilistic predictions by penalizing confident wrong predictions more
                    strongly.
                  </p>
                  <p>Lower values indicate better probability estimates.</p>
                  <p>Unlike accuracy, LogLoss accounts for confidence, not just correctness.</p>
                </>,
              )}
              value={fmtNumber(calibrationMetrics.logLossAfter, 3)}
              subtitle={`Before: ${fmtNumber(calibrationMetrics.logLossBefore, 3)}`}
              icon={<TrendingUp className="w-6 h-6" />}
            />

            <StatCard
              title={renderMetricTitle(
                "ECE",
                <>
                  <p>
                    Expected Calibration Error (ECE) measures the average difference between predicted probabilities
                    and observed win frequencies across probability bins.
                  </p>
                  <p>Lower values indicate better calibration.</p>
                  <p>An ECE of 0 means perfect calibration.</p>
                </>,
              )}
              value={fmtNumber(calibrationMetrics.ece, 3)}
              subtitle={`Before: ${fmtNumber(calibrationMetrics.ece, 3)}`}
              icon={<Activity className="w-6 h-6" />}
            />
          </div>
        </TooltipProvider>
      </section>

      {/* Calibration quality */}
      <section className="container mx-auto px-4 pb-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Calibration Quality</h2>
          <p className="text-sm text-muted-foreground">
            Slope: {fmtNumber(calibrationMetrics.calibrationSlope, 3)} • Intercept:{" "}
            {fmtNumber(calibrationMetrics.calibrationIntercept, 3)} • Avg predicted:{" "}
            {fmtPercent(calibrationMetrics.avgPredictedProb * 100, 1)} • Base rate:{" "}
            {fmtPercent(calibrationMetrics.baseRate * 100, 1)}
          </p>
        </div>
      </section>

      {/* Strategy (simulated window subset) */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-2xl font-bold">Strategy (Simulated Window Subset)</h2>
            <p className="text-sm text-muted-foreground">
              Simulated historical subset from local_matched_games, restricted to the current window. This is not a
              record of actual placed bets.
            </p>
          </div>
          <span className="rounded-full border border-border bg-muted/30 px-3 py-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            local_matched_games
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatCard
            title="Strategy matches in window"
            value={`${localMatchedCountDisplay}`}
            subtitle="Games that matched the filter stack"
            icon={<Target className="w-6 h-6" />}
          />
          <StatCard
            title="Wins / Win rate"
            value={`${localMatchedWinsDisplay} / ${fmtPercent(localMatchedWinRateDisplay, 1)}`}
            subtitle={`n=${localMatchedCountBreakdown} • window ${localMatchedWindowStartLabel} → ${localMatchedWindowEndLabel}`}
            icon={<TrendingUp className="w-6 h-6" />}
          />
          <StatCard
            title="Bankroll (Last 200 Window)"
            value={fmtCurrencyEUR(simulatedBankroll, 2)}
            subtitle={`Net P/L: ${formatSigned(localMatchedProfitSumDisplay)} • start ${fmtCurrencyEUR(
              START_BANKROLL_SIM,
              0,
            )}`}
            icon={<Activity className="w-6 h-6" />}
          />
          <StatCard
            title="ROI / Sharpe / Max DD"
            value={strategyRoiDisplay}
            subtitle={`Sharpe: ${strategySharpeDisplay} • DD: ${strategyMaxDrawdownDisplay}`}
            icon={<BarChart3 className="w-6 h-6" />}
          />
        </div>

        <div className="glass-card p-6">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
            <div>
              <h3 className="text-lg font-semibold">LOCAL MATCHED GAMES (Latest)</h3>
              <p className="text-xs text-muted-foreground">Source: {localMatchedSourceLabel}</p>
            </div>
            <div className="text-xs text-muted-foreground">
              {localMatchedGamesRowsSorted.length > 0
                ? `n=${localMatchedDisplayCount} • Wins=${localMatchedDisplayWins} • P/L ${formatSigned(
                    localMatchedDisplayProfitSum,
                  )}`
                : "—"}
            </div>
          </div>

          <div className="overflow-x-auto">
            {localMatchedGamesRowsSorted.length === 0 ? (
              <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
                <p>No simulated matches in window</p>
                <p className="mt-1">No simulated matches are available for the current window.</p>
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left border-b border-border">
                    <th className="py-2 pr-4">Date</th>
                    <th className="py-2 pr-4">Home</th>
                    <th className="py-2 pr-4">Away</th>
                    <th className="py-2 pr-4">Home Win Rate</th>
                    <th className="py-2 pr-4">Prob ISO</th>
                    <th className="py-2 pr-4">Prob Used</th>
                    <th className="py-2 pr-4">Odds</th>
                    <th className="py-2 pr-4">EV €/100</th>
                    <th className="py-2 pr-4">Odds filter</th>
                    <th className="py-2 pr-4">Win</th>
                    <th className="py-2 pr-4">P/L</th>
                  </tr>
                </thead>
                <tbody>
                  {localMatchedGamesRowsSorted.map((row) => {
                    const oddsInRange =
                      oddsMin !== null && oddsMax !== null ? row.odds_1 >= oddsMin && row.odds_1 <= oddsMax : null;
                    return (
                      <tr
                        key={`${row.date}-${row.home_team}-${row.away_team}`}
                        className="border-b border-border/50"
                      >
                        <td className="py-2 pr-4">{row.date}</td>
                        <td className="py-2 pr-4 font-medium">{row.home_team}</td>
                        <td className="py-2 pr-4">{row.away_team}</td>
                        <td className="py-2 pr-4">{fmtNumber(row.home_win_rate, 2)}</td>
                        <td className="py-2 pr-4">{fmtNumber(row.prob_iso, 2)}</td>
                        <td className="py-2 pr-4">{fmtNumber(row.prob_used, 2)}</td>
                        <td className="py-2 pr-4">{fmtNumber(row.odds_1, 2)}</td>
                        <td className="py-2 pr-4">{fmtNumber(row.ev_eur_per_100, 2)}</td>
                        <td className="py-2 pr-4">
                          {oddsRangeLabel ? (
                            <span className={oddsInRange ? "text-emerald-400" : "text-red-400"}>
                              {oddsInRange ? "✓" : "✕"} {oddsRangeLabel}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="py-2 pr-4">{row.win === 1 ? "✅" : "❌"}</td>
                        <td className="py-2 pr-4">{formatSigned(row.pnl)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </section>

      {/* Placed bets overview */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Manual Actual Bets</h2>
          <p className="text-sm text-muted-foreground">Source: actual_bets_manual.csv (manual real-bet log).</p>
        </div>

        <div className="glass-card p-6 mb-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <StatCard title="Total manual bets" value={`${actualBetsRows.length}`} icon={<Target className="w-6 h-6" />} />
            <StatCard title="Settled / Pending" value={`${actualSettledRows.length} / ${pendingBets}`} icon={<Activity className="w-6 h-6" />} />
            <StatCard title="Manual stake / PnL" value={fmtCurrencyEUR(totalStake, 2)} subtitle={`P/L: ${formatSigned(totalPnl)}`} icon={<TrendingUp className="w-6 h-6" />} />
            <StatCard title="Manual ROI / Win rate" value={`${fmtPercent(roiPct, 2)} / ${fmtPercent(winRatePct, 2)}`} icon={<BarChart3 className="w-6 h-6" />} />
          </div>
          {actualBetsRowsSorted.length === 0 ? (
            <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
              No manual bets logged yet.
            </div>
          ) : (
            <div className="rounded-lg border border-border p-4 text-sm">
              <div className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">Latest manual bet</div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <div><span className="text-muted-foreground">Bet date:</span> {actualBetsRowsSorted[0].bet_date || "—"}</div>
                <div><span className="text-muted-foreground">Game:</span> {actualBetsRowsSorted[0].away_team} @ {actualBetsRowsSorted[0].home_team}</div>
                <div><span className="text-muted-foreground">Selection:</span> {actualBetsRowsSorted[0].selection || "—"}</div>
                <div><span className="text-muted-foreground">Market:</span> {actualBetsRowsSorted[0].market || "—"}</div>
                <div><span className="text-muted-foreground">Odds:</span> {actualBetsRowsSorted[0].odds || "—"}</div>
                <div><span className="text-muted-foreground">Stake:</span> {actualBetsRowsSorted[0].stake_eur ? fmtCurrencyEUR(parseNumeric(actualBetsRowsSorted[0].stake_eur) ?? 0, 2) : "—"}</div>
                <div><span className="text-muted-foreground">PnL:</span> {actualBetsRowsSorted[0].pnl_eur ? formatSigned(parseNumeric(actualBetsRowsSorted[0].pnl_eur) ?? 0) : "—"}</div>
                <div><span className="text-muted-foreground">Status:</span> {actualBetsRowsSorted[0].status || "—"}</div>
                <div className="md:col-span-2"><span className="text-muted-foreground">Note:</span> {actualBetsRowsSorted[0].user_note || actualBetsRowsSorted[0].model_note || "—"}</div>
              </div>
            </div>
          )}
        </div>

        <div className="mb-6">
          <h2 className="text-2xl font-bold">Legacy Live Bet Log</h2>
          <p className="text-sm text-muted-foreground">Source: {betLogFlatSource} (settled only).</p>
          <p className="text-xs text-muted-foreground mt-1">
            This legacy bet_log_flat_live.csv source is separate from the manual Actual Bets log.
          </p>
          <p className="text-xs text-muted-foreground mt-1">Real bets (settled via combined_*)</p>
          <p className="text-xs text-muted-foreground">Actually placed &amp; settled bets only</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Settled bets (2026)"
            value={`${settledBetsSummary.count}`}
            subtitle="Settled bets from the live log"
            icon={<Target className="w-6 h-6" />}
          />
          <StatCard
            title="Wins / Win rate"
            value={`${settledBetsSummary.wins} / ${fmtPercent(settledWinRatePct, 2)}`}
            subtitle={`${settledBetsSummary.wins} wins · ${settledBetsSummary.count - settledBetsSummary.wins} losses`}
            icon={<TrendingUp className="w-6 h-6" />}
          />
          <StatCard
            title="Bankroll (2026 YTD · Placed Bets)"
            value={fmtCurrencyEUR(realBankroll, 2)}
            subtitle={`Start ${fmtCurrencyEUR(START_BANKROLL_REAL, 0)} • Net P/L: ${fmtCurrencyEUR(
              settledBetsSummary.profit_eur,
              2,
            )}`}
            icon={<Activity className="w-6 h-6" />}
          />
          <StatCard
            title="ROI / Avg Odds"
            value={fmtPercent(settledBetsSummary.roi_pct, 2)}
            subtitle={`Avg odds: ${fmtNumber(settledBetsSummary.avg_odds, 2)}`}
            icon={<BarChart3 className="w-6 h-6" />}
          />
        </div>
      </section>

      {/* Placed bets (real) */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Settled Bets (2026)</h2>
          <p className="text-sm text-muted-foreground">
            Source: {betLogFlatSource}, settled against {combinedSource} outcomes.
          </p>
        </div>

        <div className="glass-card p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <StatCard
              title="Bankroll (2026 YTD · Placed Bets)"
              value={fmtCurrencyEUR(realBankroll, 2)}
              subtitle={`Start ${fmtCurrencyEUR(START_BANKROLL_REAL, 0)} • Net P/L: ${fmtCurrencyEUR(
                settledBetsSummary.profit_eur,
                2,
              )}`}
              icon={<Activity className="w-6 h-6" />}
            />
            <StatCard
              title="Settled Bets (2026)"
              value={`${settledBetsSummary.count}`}
              subtitle={`${settledBetsSummary.count} settled • ${settledBetsSummary.wins}W-${
                settledBetsSummary.count - settledBetsSummary.wins
              }L • ROI: ${fmtPercent(settledBetsSummary.roi_pct, 0)} • avg odds: ${fmtNumber(
                settledBetsSummary.avg_odds,
                2,
              )}`}
              icon={<Target className="w-6 h-6" />}
            />
          </div>

          <div className="text-xs text-muted-foreground mb-6">
            These are real placed bets, settled after the fact using final results from {combinedSource}.
          </div>

          <div className="overflow-x-auto">
            {settledBetsRows.length === 0 ? (
              <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
                No settled bets recorded for 2026 yet.
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left border-b border-border">
                    <th className="py-2 pr-4">Date</th>
                    <th className="py-2 pr-4">Home</th>
                    <th className="py-2 pr-4">Away</th>
                    <th className="py-2 pr-4">Stake</th>
                    <th className="py-2 pr-4">Odds</th>
                    <th className="py-2 pr-4">Result</th>
                    <th className="py-2 pr-4">P/L</th>
                  </tr>
                </thead>
                <tbody>
                  {settledBetsRows.map((bet) => (
                    <tr
                      key={`${bet.date}-${bet.home_team}-${bet.away_team}-${bet.pick_team}`}
                      className="border-b border-border/50"
                    >
                      <td className="py-2 pr-4">{bet.date}</td>
                      <td className="py-2 pr-4 font-medium">{bet.home_team}</td>
                      <td className="py-2 pr-4">{bet.away_team}</td>
                      <td className="py-2 pr-4">{fmtCurrencyEUR(bet.stake, 2)}</td>
                      <td className="py-2 pr-4">{fmtNumber(bet.odds, 2)}</td>
                      <td className="py-2 pr-4">{bet.win === 1 ? "✅" : "❌"}</td>
                      <td className="py-2 pr-4">{formatSigned(bet.pnl)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </section>

      {/* Home win rates */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Home Win Rates</h2>
          <p className="text-sm text-muted-foreground">
            Diagnostic table ranked by home win rate descending.
          </p>
        </div>

        <div className="glass-card p-6 overflow-x-auto">
          {homeWinRatesDiagnosticRows.length === 0 ? (
            <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
              No home win rate data available yet.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="py-2 pr-4">Team</th>
                  <th className="py-2 pr-4">Home Win Rate</th>
                  <th className="py-2 pr-4">Home Wins</th>
                  <th className="py-2 pr-4">Home Games</th>
                  <th className="py-2 pr-4">Window Games</th>
                </tr>
              </thead>
              <tbody>
                {homeWinRatesDiagnosticRows.map((row) => (
                  <tr key={row.team} className="border-b border-border/50">
                    <td className="py-2 pr-4 font-medium">{row.team}</td>
                    <td className="py-2 pr-4">{fmtPercent(row.homeWinRate * 100, 1)}</td>
                    <td className="py-2 pr-4">{row.homeWins}</td>
                    <td className="py-2 pr-4">{row.homeGames}</td>
                    <td className="py-2 pr-4">{row.windowGames}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-4 text-xs text-muted-foreground">
            Showing all teams with home win rate &gt; 50% in the last 200 games.
          </p>
        </div>
      </section>

      <section className="container mx-auto px-4 pb-10">
        <p className="text-xs text-muted-foreground">
          Dashboard parity note: sections distinguish latest settled historical data, current run, simulated window
          subset, and real placed bets.
        </p>
      </section>
      </>
      )}
    </>
  );
};

export default Index;
