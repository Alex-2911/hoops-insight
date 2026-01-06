import { useEffect, useMemo, useState } from "react";
import { StatCard } from "@/components/cards/StatCard";
import type {
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

const Index = () => {
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [tables, setTables] = useState<TablesPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const baseUrl = import.meta.env.BASE_URL ?? "/";

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [summaryRes, tablesRes] = await Promise.all([
          fetch(`${baseUrl}data/summary.json`),
          fetch(`${baseUrl}data/tables.json`),
        ]);

        if (!summaryRes.ok || !tablesRes.ok) {
          throw new Error("Failed to load dashboard data.");
        }

        const summaryJson = (await summaryRes.json()) as SummaryPayload;
        const tablesJson = (await tablesRes.json()) as TablesPayload;

        if (alive) {
          setSummary(summaryJson);
          setTables(tablesJson);
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
  };
  const homeWinRatesLast20 = tables?.home_win_rates_last20 ?? [];
  const betLogSummary = tables?.bet_log_summary ?? {
    asOfDate: "—",
    totalBets: 0,
    totalStakedEur: 0,
    totalProfitEur: 0,
    roiPct: 0,
    avgStakeEur: 0,
    avgProfitPerBetEur: 0,
    winRate: 0,
  };
  const bankrollHistory = tables?.bankroll_history ?? [];
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

  const strategySummary = summary?.strategy_summary ?? {
    totalBets: 0,
    totalProfitEur: 0,
    roiPct: 0,
    avgEvPer100: 0,
    winRate: 0,
    sharpeStyle: null,
    profitMetricsAvailable: false,
  };
  const strategyParams = summary?.strategy_params ?? {
    source: "missing",
    params: {},
  };
  const strategyFilterStats = summary?.strategy_filter_stats ?? {
    window_size: 200,
    filters: [],
    matched_games_count: 0,
  };
  const metricsSnapshotSource = summary?.source?.metrics_snapshot_source ?? "missing";
  const localParamsMissing =
    strategyParams.source === "missing" || metricsSnapshotSource === "missing";

  const lastBankroll = bankrollHistory[bankrollHistory.length - 1];

  const overallAccuracyPct = (summaryStats.overall_accuracy * 100).toFixed(2);

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

  const strategyParamsList = useMemo(() => {
    return Object.entries(strategyParams.params ?? {});
  }, [strategyParams.params]);

  const formatSigned = (value: number) => {
    const sign = value >= 0 ? "+" : "-";
    return `${sign}€${Math.abs(value).toFixed(2)}`;
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

      {/* Top stats */}
      <section className="container mx-auto px-4 py-10">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              title="Overall Accuracy"
              value={`${overallAccuracyPct}%`}
              subtitle={
                <div className="space-y-1">
                  <div>Played games: {summaryStats.total_games}</div>
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
              title="Last Update"
              value={summaryStats.as_of_date}
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
                value={`${calibrationMetrics.brierAfter.toFixed(3)}`}
                subtitle={`Before: ${calibrationMetrics.brierBefore.toFixed(3)}`}
                icon={<BarChart3 className="w-6 h-6" />}
              />

              <StatCard
                title="Bankroll (Historical)"
                value={lastBankroll ? `€${lastBankroll.balance.toFixed(2)}` : "—"}
                subtitle={lastBankroll ? `Last day P/L: €${lastBankroll.profit.toFixed(2)}` : "—"}
                icon={<Activity className="w-6 h-6" />}
              />
        </div>
      </section>

      {/* Local params KPIs */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <div className="flex flex-col gap-1 mb-6">
            <h2 className="text-xl font-bold">Local Params KPIs</h2>
            <p className="text-xs text-muted-foreground">
              Params source: {strategyParams.source}
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Bets</div>
              <div className="text-2xl font-bold">
                {localParamsMissing ? 0 : strategySummary.totalBets}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Profit</div>
              <div className="text-2xl font-bold">
                {!localParamsMissing && strategySummary.profitMetricsAvailable
                  ? `€${strategySummary.totalProfitEur.toFixed(2)}`
                  : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">ROI</div>
              <div className="text-2xl font-bold">
                {!localParamsMissing && strategySummary.profitMetricsAvailable
                  ? `${strategySummary.roiPct.toFixed(2)}%`
                  : "N/A"}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Avg EV €/100</div>
              <div className="text-2xl font-bold">
                {localParamsMissing ? "N/A" : strategySummary.avgEvPer100.toFixed(2)}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Win Rate</div>
              <div className="text-2xl font-bold">
                {localParamsMissing
                  ? "N/A"
                  : `${(strategySummary.winRate * 100).toFixed(2)}%`}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>Sharpe Ratio</span>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground"
                        aria-label="Sharpe Ratio info"
                      >
                        <Info className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Risk-adjusted return: mean(P/L) / std(P/L). Higher is
                      better; 0≈flat, 1≈good (rule of thumb).
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <div className="text-2xl font-bold">
                {!localParamsMissing && strategySummary.sharpeStyle !== null
                  ? strategySummary.sharpeStyle.toFixed(2)
                  : "N/A"}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Strategy filter coverage */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Strategy Filter Coverage</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Window games: {strategyFilterStats.window_size}
          </p>
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
            </div>
          </div>
        </div>
      </section>

      {/* Calibration */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Calibration Quality</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Metrics compare raw probabilities vs isotonic-calibrated probabilities.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Brier Score</div>
              <div className="text-sm text-muted-foreground">
                Before: <span className="font-medium text-foreground">{calibrationMetrics.brierBefore.toFixed(6)}</span>
              </div>
              <div className="text-sm text-muted-foreground">
                After: <span className="font-medium text-foreground">{calibrationMetrics.brierAfter.toFixed(6)}</span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Fitted games: {calibrationMetrics.fittedGames} • As of {calibrationMetrics.asOfDate}
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Log Loss</div>
              <div className="text-sm text-muted-foreground">
                Before: <span className="font-medium text-foreground">{calibrationMetrics.logLossBefore.toFixed(6)}</span>
              </div>
              <div className="text-sm text-muted-foreground">
                After: <span className="font-medium text-foreground">{calibrationMetrics.logLossAfter.toFixed(6)}</span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Lower is better • Historical only
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Bet log summary (historical only) */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Historical Bet Log Summary (Settled Games)</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Aggregated statistics from played games only. No future recommendations are displayed.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Total Bets</div>
              <div className="text-2xl font-bold">{betLogSummary.totalBets}</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Total Profit</div>
              <div className="text-2xl font-bold">€{betLogSummary.totalProfitEur.toFixed(2)}</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">ROI</div>
              <div className="text-2xl font-bold">{betLogSummary.roiPct.toFixed(2)}%</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Avg Stake</div>
              <div className="text-2xl font-bold">€{betLogSummary.avgStakeEur.toFixed(2)}</div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground mt-4">
            As of {betLogSummary.asOfDate} • Historical / settled only.
          </div>
        </div>
      </section>

      {/* Local matched games overview */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">
            LOCAL MATCHED GAMES (LAST {strategyFilterStats.window_size} WINDOW)
          </h2>
          <p className="text-sm text-muted-foreground mb-6">
            Rows: {localMatchedGamesCount} • Net P/L: €{localMatchedGamesProfitSum.toFixed(2)}
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
                    <th className="py-2 pr-4">Odds 1</th>
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
                      <td className="py-2 pr-4">{game.home_win_rate.toFixed(2)}</td>
                      <td className="py-2 pr-4">{game.prob_iso.toFixed(3)}</td>
                      <td className="py-2 pr-4">{game.prob_used.toFixed(3)}</td>
                      <td className="py-2 pr-4">{game.odds_1.toFixed(2)}</td>
                      <td className="py-2 pr-4">{game.ev_eur_per_100.toFixed(2)}</td>
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

      {/* Bankroll snapshots */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-6">Bankroll Snapshots</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">
                Bankroll (Last 200 Games)
              </div>
              <div className="text-2xl font-bold">
                €{bankrollLast200.bankroll.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Start €{bankrollLast200.start.toFixed(0)} • Net P/L: €
                {bankrollLast200.net_pl.toFixed(2)} • €{bankrollLast200.stake.toFixed(0)}{" "}
                flat stake (example)
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Bankroll (2026 YTD)</div>
              <div className="text-2xl font-bold">
                €{bankrollYtd2026.bankroll.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Start €{bankrollYtd2026.start.toFixed(0)} • Net P/L 2026: €
                {bankrollYtd2026.net_pl.toFixed(2)} • €{bankrollYtd2026.stake.toFixed(0)}{" "}
                flat stake (example)
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Home win rates */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">
            Home Win Rate (Last 20 Games Window)
          </h2>
          <p className="text-sm text-muted-foreground mb-6">
            Last-20 window per team; win rate computed only on home games inside that
            window.
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
                      {(t.homeWinRate * 100).toFixed(0)}%
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
            Showing {topHomeTeams.length} teams with home win rate &gt; 50%.
          </div>
        </div>
      </section>
    </>
  );
};

export default Index;
