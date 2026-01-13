import { useEffect, useMemo, useState } from "react";
import { StatCard } from "@/components/cards/StatCard";
import type {
  LastRunPayload,
  SummaryPayload,
  TablesPayload,
} from "@/data/dashboardTypes";
import { Target, TrendingUp, Activity, BarChart3, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fmtCurrencyEUR, fmtNumber, fmtPercent } from "@/lib/format";

const Index = () => {
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [tables, setTables] = useState<TablesPayload | null>(null);
  const [lastRun, setLastRun] = useState<LastRunPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const baseUrl = import.meta.env.BASE_URL ?? "/";

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [summaryRes, tablesRes, lastRunRes] = await Promise.all([
          fetch(`${baseUrl}data/summary.json`),
          fetch(`${baseUrl}data/tables.json`),
          fetch(`${baseUrl}data/last_run.json`),
        ]);

        if (!summaryRes.ok || !tablesRes.ok) {
          throw new Error("Failed to load dashboard data.");
        }

        const summaryJson = (await summaryRes.json()) as SummaryPayload;
        const tablesJson = (await tablesRes.json()) as TablesPayload;
        const lastRunJson = lastRunRes.ok
          ? ((await lastRunRes.json()) as LastRunPayload)
          : null;

        if (alive) {
          setSummary(summaryJson);
          setTables(tablesJson);
          setLastRun(lastRunJson);
        }
      } catch (err) {
        if (alive) {
          setLoadError(err instanceof Error ? err.message : "Failed to load data.");
        }
      }
    };

    load();
    return () => {
      alive = false;
    };
  }, []);

  const summaryStats = summary?.summary_stats ?? {
    total_games: 0,
    overall_accuracy: 0,
    as_of_date: "—",
  };

  const calibrationMetrics = tables?.calibration_metrics ?? {
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
  const homeWinRatesLast20 = tables?.home_win_rates_last20 ?? [];
  const localMatchedGamesRows = tables?.local_matched_games_rows ?? [];
  const localMatchedGamesCount =
    tables?.local_matched_games_count ?? localMatchedGamesRows.length;
  const localMatchedGamesMismatch = tables?.local_matched_games_mismatch ?? false;
  const localMatchedGamesNote = tables?.local_matched_games_note ?? "";
  const localMatchedGamesProfitSum =
    tables?.local_matched_games_profit_sum_table ?? 0;
  const bankrollLast200 = tables?.bankroll_last_200 ?? {
    start: 1000,
    stake: 100,
    net_pl: 0,
    bankroll: 1000,
  };
  const bankrollYtd2026 = tables?.bankroll_ytd_2026 ?? {
    start: 1000,
    stake: 100,
    net_pl: 0,
    bankroll: 1000,
  };
  const localMatchedGamesAvgOdds = tables?.local_matched_games_avg_odds ?? 0;
  const settledBetsRows = tables?.settled_bets_rows ?? [];
  const settledBetsSummary = tables?.settled_bets_summary ?? {
    count: 0,
    wins: 0,
    profit_eur: 0,
    roi_pct: 0,
    avg_odds: 0,
  };

  const strategySummary = summary?.strategy_summary ?? {
    totalBets: 0,
    totalProfitEur: 0,
    roiPct: 0,
    avgEvPer100: 0,
    winRate: 0,
    sharpeStyle: null,
    profitMetricsAvailable: false,
    asOfDate: "—",
  };
  const strategyParams = summary?.strategy_params ?? {
    source: "missing",
    params: {},
    params_used: {},
    active_filters: "No active filters.",
  };
  const strategyFilterStats = summary?.strategy_filter_stats ?? {
    window_size: 200,
    filters: [],
    matched_games_count: 0,
  };
  const metricsSnapshotSource = summary?.source?.metrics_snapshot_source ?? "missing";
  const betLogFlatSource = summary?.source?.bet_log_flat_file ?? "missing";
  const localMatchedGamesSource = tables?.local_matched_games_source ?? "";
  const localParamsMissing =
    strategyParams.source === "missing" || metricsSnapshotSource === "missing";
  const realBetsAvailable =
    summary?.real_bets_available !== false && betLogFlatSource !== "missing";
  const summaryAsOfDate = summary?.as_of_date ?? "—";
  const lastUpdateTimestamp =
    lastRun?.generated_at ??
    summary?.generated_at ??
    lastRun?.run_timestamp ??
    summary?.last_run ??
    lastRun?.last_run ??
    "—";

  const overallAccuracyPct = fmtPercent(summaryStats.overall_accuracy * 100, 2);
  const calibrationWindowSize = calibrationMetrics.windowSize || strategyFilterStats.window_size;
  const windowGamesLabel = calibrationWindowSize || summaryStats.total_games;

  const topHomeTeams = useMemo(() => {
    return [...homeWinRatesLast20].sort((a, b) => b.homeWinRate - a.homeWinRate);
  }, [homeWinRatesLast20]);

  const strategySubsetAvailable = strategySummary.totalBets > 0;
  const strategySubsetWins = strategySubsetAvailable
    ? localMatchedGamesRows.length
      ? localMatchedGamesRows.filter((game) => game.win === 1).length
      : Math.round(strategySummary.winRate * strategySummary.totalBets)
    : 0;
  const matchedGamesCount = strategyFilterStats.matched_games_count ?? 0;
  const strategyLatestRowDate =
    localMatchedGamesRows.length === 0
      ? null
      : localMatchedGamesRows.reduce((latest, row) => (row.date > latest ? row.date : latest), "—");

  const strategyParamsList = useMemo(() => {
    return Object.entries(strategyParams.params_used ?? {});
  }, [strategyParams.params_used]);

  const formatActiveFilters = (value?: string | null) => {
    if (!value) {
      return null;
    }
    const trimmed = value.trim();
    if (!trimmed || trimmed.toLowerCase() === "none") {
      return null;
    }
    return trimmed;
  };

  const formatFilterLabels = (filters?: Array<{ label: string }>) => {
    if (!filters || filters.length === 0) {
      return null;
    }
    return filters.map((filter) => filter.label).join(" • ");
  };

  const activeFiltersLabel =
    formatActiveFilters(lastRun?.active_filters_human) ??
    formatActiveFilters(lastRun?.active_filters) ??
    formatActiveFilters(strategyParams.active_filters ?? null) ??
    formatFilterLabels(lastRun?.strategy_filter_stats?.filters) ??
    formatFilterLabels(strategyFilterStats.filters) ??
    "No active filters.";

  const formatSigned = (value: number | null | undefined) => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return "—";
    }
    const sign = value >= 0 ? "+" : "-";
    return `${sign}€${fmtNumber(Math.abs(value), 2)}`;
  };

  return (
    <>
      {/* Header / Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
        <div className="container mx-auto px-4 py-16 relative">
          <div className="max-w-3xl mx-auto text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
              <Activity className="w-4 h-4" />
              Hoops Insight • Results & Statistics
            </div>

            <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
              Historical NBA Results & Model Statistics
            </h1>

            <p className="text-lg text-muted-foreground mb-2">
              This page displays historical results and statistical summaries only.
            </p>

            <p className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">Legal:</span> This
              website does not provide predictions for future sporting or betting
              outcomes. It serves purely for historical model accuracy and
              statistical analysis of NBA games.
            </p>
            {loadError && (
              <p className="mt-3 text-sm text-red-400">
                Data unavailable: {loadError}
              </p>
            )}
          </div>
        </div>
      </section>

      {/* Window performance (model) */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Window Performance (Model)</h2>
          <p className="text-sm text-muted-foreground">
            Source: combined_nba_predictions_* (played games only, windowed).
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <StatCard
            title="Overall Accuracy"
            value={overallAccuracyPct}
            subtitle={
              <div className="space-y-1">
                <div>Window games: {windowGamesLabel}</div>
                <div>
                  Strategy subset in window:{" "}
                  {strategySubsetAvailable ? (
                    <>
                      {strategySummary.totalBets} matches • {strategySubsetWins} won
                    </>
                  ) : (
                    "N/A (missing local snapshot)"
                  )}
                </div>
              </div>
            }
            icon={<Target className="w-6 h-6" />}
          />

          <StatCard
            title="As of"
            value={summaryAsOfDate}
            subtitle={
              <div className="space-y-1">
                <div>Last update: {lastUpdateTimestamp}</div>
                <div>Window size: {strategyFilterStats.window_size}</div>
                {localMatchedGamesSource && strategyLatestRowDate ? (
                  <div>Strategy latest row: {strategyLatestRowDate}</div>
                ) : null}
              </div>
            }
            icon={<TrendingUp className="w-6 h-6" />}
          />

          <StatCard
            title={
              <div className="flex items-center gap-2">
                <span>Calibration (Brier)</span>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground"
                        aria-label="Brier Score info"
                      >
                        <Info className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <div className="font-semibold">Brier Score</div>
                      <div className="text-xs text-muted-foreground">
                        Measures probability accuracy (0=best). ~0.20 good, ~0.25 ok,
                        &gt;0.30 weak (depends on base rate).
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            }
            value={fmtNumber(calibrationMetrics.brierAfter, 3)}
            subtitle={`Before: ${fmtNumber(calibrationMetrics.brierBefore, 3)}`}
            icon={<BarChart3 className="w-6 h-6" />}
          />
        </div>
      </section>

      {/* Calibration */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Calibration Quality</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Calibration metrics computed on the last {calibrationWindowSize} played games only. Historical results only.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Brier Score</div>
              <div className="text-sm text-muted-foreground">
                {fmtNumber(calibrationMetrics.brierAfter, 3)} (raw)
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Log Loss</div>
              <div className="text-sm text-muted-foreground">
                {fmtNumber(calibrationMetrics.logLossAfter, 3)} (raw)
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">ECE</div>
              <div className="text-sm text-muted-foreground">
                {fmtNumber(calibrationMetrics.ece, 3)} (raw)
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Calibration Slope</div>
              <div className="text-sm text-muted-foreground">
                {fmtNumber(calibrationMetrics.calibrationSlope, 3)} (raw)
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Calibration Intercept</div>
              <div className="text-sm text-muted-foreground">
                {fmtNumber(calibrationMetrics.calibrationIntercept, 3)} (raw)
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Avg Predicted Prob</div>
              <div className="text-sm text-muted-foreground">
                {fmtPercent(calibrationMetrics.avgPredictedProb * 100, 2)} (raw)
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Base Rate</div>
              <div className="text-sm text-muted-foreground">
                {fmtPercent(calibrationMetrics.baseRate * 100, 2)}
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Actual Win %</div>
              <div className="text-sm text-muted-foreground">
                {fmtPercent(calibrationMetrics.actualWinPct * 100, 2)}
              </div>
            </div>
          </div>
          <div className="text-xs text-muted-foreground mt-4">
            Fitted games: {calibrationMetrics.fittedGames} • Window: {calibrationWindowSize} • As of{" "}
            {calibrationMetrics.asOfDate}
          </div>
        </div>
      </section>

      {/* Strategy (simulated) */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Strategy (Simulated on Window Subset)</h2>
          <p className="text-sm text-muted-foreground">
            Source: local_matched_games_YYYY-MM-DD.csv (window subset only).
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          <StatCard
            title="Bankroll (Last 200 Games)"
            value={fmtCurrencyEUR(bankrollLast200.bankroll, 2)}
            subtitle={`Start ${fmtCurrencyEUR(bankrollLast200.start, 0)} • Net P/L: ${fmtCurrencyEUR(bankrollLast200.net_pl, 2)}`}
            icon={<Activity className="w-6 h-6" />}
          />

          <StatCard
            title="Bankroll (2026 YTD, Simulated)"
            value={fmtCurrencyEUR(bankrollYtd2026.bankroll, 2)}
            subtitle={`Start ${fmtCurrencyEUR(bankrollYtd2026.start, 0)} • Net P/L 2026: ${fmtCurrencyEUR(bankrollYtd2026.net_pl, 2)}`}
            icon={<Activity className="w-6 h-6" />}
          />
        </div>

        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Strategy Filter Coverage</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Descriptive filter pass rates for the last {strategyFilterStats.window_size} games window. No recommendations or picks.
          </p>
          <div className="rounded-lg border border-border px-4 py-3 text-sm text-muted-foreground mb-6">
            Active filters: <span className="text-foreground">{activeFiltersLabel}</span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-3">Filter params</div>
              {strategyParamsList.length === 0 ? (
                <div className="text-sm text-muted-foreground">
                  No strategy params found.
                </div>
              ) : (
                <ul className="space-y-2 text-sm">
                  {strategyParamsList.map(([key, value]) => (
                    <li key={key} className="flex items-center justify-between">
                      <span className="text-muted-foreground">{key}</span>
                      <span className="font-medium text-foreground">{String(value)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-3">Coverage counts</div>
              <ul className="space-y-2 text-sm">
                {strategyFilterStats.filters.map((item) => (
                  <li key={item.label} className="flex items-center justify-between">
                    <span className="text-muted-foreground">{item.label}</span>
                    <span className="font-medium text-foreground">{item.count}</span>
                  </li>
                ))}
                <li className="flex items-center justify-between pt-2 border-t border-border/50">
                  <span className="text-muted-foreground">Matched games</span>
                  <span className="font-medium text-foreground">{matchedGamesCount}</span>
                </li>
              </ul>
              <div className="mt-4 text-xs text-muted-foreground">
                Matched subset accuracy:{" "}
                {strategySubsetAvailable
                  ? fmtPercent(
                      (strategySubsetWins / Math.max(strategySummary.totalBets, 1)) * 100,
                      2,
                    )
                  : "N/A"}{" "}
                • Matched games: {matchedGamesCount}
              </div>
              <div className="text-xs text-muted-foreground">
                Params source: {strategyParams.source}
              </div>
            </div>
          </div>
          <div className="text-xs text-muted-foreground mt-4">
            Avg odds (window subset): {fmtNumber(localMatchedGamesAvgOdds, 2)} • Lower odds keep ROI interpretation stable.
          </div>
        </div>
      </section>

      {/* Local params bet log summary (historical only) */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Local Params Bet Log Summary (Settled Games)</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Aggregated statistics from strategy-matched bets in the last {strategyFilterStats.window_size} games window. No future recommendations are displayed.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Bets (Local Params)</div>
              <div className="text-2xl font-bold">{localParamsMissing ? 0 : strategySummary.totalBets}</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Profit (Local Params)</div>
              <div className="text-2xl font-bold">
                {!localParamsMissing && strategySummary.profitMetricsAvailable
                  ? fmtCurrencyEUR(strategySummary.totalProfitEur, 2)
                  : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">ROI (Local Params)</div>
              <div className="text-2xl font-bold">
                {!localParamsMissing && strategySummary.profitMetricsAvailable
                  ? fmtPercent(strategySummary.roiPct, 2)
                  : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Avg Stake (Local Params)</div>
              <div className="text-2xl font-bold">{fmtCurrencyEUR(bankrollLast200.stake, 2)}</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Avg EV €/100 (Local Params)</div>
              <div className="text-2xl font-bold">
                {localParamsMissing ? "N/A" : fmtNumber(strategySummary.avgEvPer100, 2)}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Avg Odds (Local Params)</div>
              <div className="text-2xl font-bold">
                {localParamsMissing ? "N/A" : fmtNumber(localMatchedGamesAvgOdds, 2)}
              </div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground mt-4">
            As of {strategySummary.asOfDate} • Windowed historical / settled only.
          </div>
        </div>
      </section>

      {/* Risk metrics */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">
            Risk Metrics (Local Params, Last {strategyFilterStats.window_size} Games Window)
          </h2>
          <p className="text-sm text-muted-foreground mb-6">
            Risk metrics are computed on settled bets that match the local params within the same windowed date range.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Sharpe Ratio</div>
              <div className="text-2xl font-bold">
                {!localParamsMissing && typeof strategySummary.sharpeStyle === "number"
                  ? fmtNumber(strategySummary.sharpeStyle, 3)
                  : "—"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Max Drawdown</div>
              <div className="text-2xl font-bold">
                {fmtCurrencyEUR(summary?.kpis?.max_drawdown_eur, 2)}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Max Drawdown %</div>
              <div className="text-2xl font-bold">
                {fmtPercent(summary?.kpis?.max_drawdown_pct, 2)}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Local matched games overview */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">
            LOCAL MATCHED GAMES (LAST {strategyFilterStats.window_size} WINDOW)
          </h2>
          <p className="text-sm text-muted-foreground mb-2">
            n_trades (last {strategyFilterStats.window_size} window): {localMatchedGamesCount}
          </p>
          <p className="text-sm text-muted-foreground mb-6">
            Rows: {localMatchedGamesCount} • Net P/L: {fmtCurrencyEUR(localMatchedGamesProfitSum, 2)}
          </p>

          {localMatchedGamesMismatch || localMatchedGamesRows.length === 0 ? (
            <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
              {localMatchedGamesNote || "No matched games recorded for this window."}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left border-b border-border">
                    <th className="py-2 pr-4">Date</th>
                    <th className="py-2 pr-4">Home</th>
                    <th className="py-2 pr-4">Away</th>
                    <th className="py-2 pr-4">Home Win Rate</th>
                    <th className="py-2 pr-4">Prob Iso</th>
                    <th className="py-2 pr-4">Prob Used</th>
                    <th className="py-2 pr-4">Odds</th>
                    <th className="py-2 pr-4">EV €/100</th>
                    <th className="py-2 pr-4">Win</th>
                    <th className="py-2 pr-4">P/L</th>
                  </tr>
                </thead>
                <tbody>
                  {localMatchedGamesRows.map((game) => (
                    <tr
                      key={`${game.date}-${game.home_team}-${game.away_team}`}
                      className="border-b border-border/50"
                    >
                      <td className="py-2 pr-4">{game.date}</td>
                      <td className="py-2 pr-4 font-medium">{game.home_team}</td>
                      <td className="py-2 pr-4">{game.away_team}</td>
                      <td className="py-2 pr-4">{fmtNumber(game.home_win_rate, 2)}</td>
                      <td className="py-2 pr-4">{fmtNumber(game.prob_iso, 3)}</td>
                      <td className="py-2 pr-4">{fmtNumber(game.prob_used, 3)}</td>
                      <td className="py-2 pr-4">{fmtNumber(game.odds_1, 2)}</td>
                      <td className="py-2 pr-4">{fmtNumber(game.ev_eur_per_100, 2)}</td>
                      <td className="py-2 pr-4">{game.win === 1 ? "✅" : "❌"}</td>
                      <td className="py-2 pr-4">{formatSigned(game.pnl)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* Placed bets (real) */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Placed Bets (Real) — 2026 YTD</h2>
          <p className="text-sm text-muted-foreground">
            Source: bet_log_flat_live.csv, settled against combined_nba_predictions_* outcomes.
          </p>
        </div>
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Settled Bets (2026)</h2>
          <p className="text-sm text-muted-foreground mb-6">
            These are real placed bets, settled after the fact using final results from combined_*.
          </p>

          {!realBetsAvailable ? (
            <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground mb-6">
              Real bet log not included in this CI build (N/A).
            </div>
          ) : null}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Count</div>
              <div className="text-2xl font-bold">
                {realBetsAvailable ? settledBetsSummary.count : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Wins</div>
              <div className="text-2xl font-bold">
                {realBetsAvailable ? settledBetsSummary.wins : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">P/L</div>
              <div className="text-2xl font-bold">
                {realBetsAvailable
                  ? fmtCurrencyEUR(settledBetsSummary.profit_eur, 2)
                  : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">ROI</div>
              <div className="text-2xl font-bold">
                {realBetsAvailable ? fmtPercent(settledBetsSummary.roi_pct, 2) : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Avg Odds</div>
              <div className="text-2xl font-bold">
                {realBetsAvailable ? fmtNumber(settledBetsSummary.avg_odds, 2) : "N/A"}
              </div>
            </div>
          </div>

          {realBetsAvailable ? (
            <div className="text-xs text-muted-foreground mt-4">
              Bets are settled only when a matching played game is available. Source: {betLogFlatSource}
            </div>
          ) : null}

          <div className="mt-6 overflow-x-auto">
            {!realBetsAvailable ? (
              <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
                Real bet log not included in this CI build (N/A).
              </div>
            ) : settledBetsRows.length === 0 ? (
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
                    <th className="py-2 pr-4">Pick</th>
                    <th className="py-2 pr-4">Odds</th>
                    <th className="py-2 pr-4">Stake</th>
                    <th className="py-2 pr-4">Win</th>
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
                      <td className="py-2 pr-4">{bet.pick_team}</td>
                      <td className="py-2 pr-4">{fmtNumber(bet.odds, 2)}</td>
                      <td className="py-2 pr-4">{fmtCurrencyEUR(bet.stake, 2)}</td>
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
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">
            Home Win Rate (Last {strategyFilterStats.window_size} Games Window)
          </h2>
          <p className="text-sm text-muted-foreground mb-6">
            Windowed home win rate per team; computed only on home games inside the last {strategyFilterStats.window_size} games.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="py-2 pr-4">Team</th>
                  <th className="py-2 pr-4">Home Win Rate</th>
                  <th className="py-2 pr-4">Home Wins</th>
                  <th className="py-2 pr-4">Home Games</th>
                  <th className="py-2 pr-4">Last 20 Games</th>
                </tr>
              </thead>
              <tbody>
                {topHomeTeams.map((t) => (
                  <tr key={t.team} className="border-b border-border/50">
                    <td className="py-2 pr-4 font-medium">{t.team}</td>
                    <td className="py-2 pr-4">
                      {fmtPercent(t.homeWinRate * 100, 0)}
                    </td>
                    <td className="py-2 pr-4">{t.homeWins}</td>
                    <td className="py-2 pr-4">{t.totalHomeGames}</td>
                    <td className="py-2 pr-4">{t.totalLast20Games}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="text-xs text-muted-foreground mt-4">
            Showing all teams with home win rate &gt; 50% in the last {strategyFilterStats.window_size} games.
          </div>
        </div>
      </section>
    </>
  );
};

export default Index;
