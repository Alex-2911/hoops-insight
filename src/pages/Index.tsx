import { useEffect, useMemo, useState } from "react";
import { StatCard } from "@/components/cards/StatCard";
import type { DashboardPayload, DashboardState } from "@/data/dashboardTypes";
import { Target, TrendingUp, Activity, BarChart3 } from "lucide-react";
import { fmtCurrencyEUR, fmtNumber, fmtPercent } from "@/lib/format";
import { shouldShowRiskMetrics } from "@/lib/riskMetrics";

const Index = () => {
  const [payload, setPayload] = useState<DashboardPayload | null>(null);
  const [dashboardState, setDashboardState] = useState<DashboardState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const baseUrl = import.meta.env.BASE_URL ?? "/";

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [payloadRes, stateRes] = await Promise.all([
          fetch(`${baseUrl}data/dashboard_payload.json`),
          fetch(`${baseUrl}data/dashboard_state.json`),
        ]);

        if (!payloadRes.ok || !stateRes.ok) {
          throw new Error("Failed to load dashboard data.");
        }

        const payloadJson = (await payloadRes.json()) as DashboardPayload;
        const stateJson = (await stateRes.json()) as DashboardState;

        if (alive) {
          setPayload(payloadJson);
          setDashboardState(stateJson);
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

  const summary = payload?.summary ?? null;
  const tables = payload?.tables ?? null;
  const windowInfo = payload?.window ?? {
    size: 0,
    start: "—",
    end: "—",
    games_count: 0,
  };

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
  const localMatchedGamesCount = localMatchedGamesRows.length;
  const localMatchedGamesMismatch = tables?.local_matched_games_mismatch ?? false;
  const localMatchedGamesNote = tables?.local_matched_games_note ?? "";
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
  const settledBetsRows = tables?.settled_bets_rows ?? [];
  const settledBetsSummary = useMemo(() => {
    if (!settledBetsRows.length) {
      return {
        count: 0,
        wins: 0,
        profit_eur: 0,
        roi_pct: 0,
        avg_odds: 0,
      };
    }
    const wins = settledBetsRows.filter((row) => row.win === 1).length;
    const profit = settledBetsRows.reduce((acc, row) => acc + (row.pnl ?? 0), 0);
    const totalStake = settledBetsRows.reduce((acc, row) => acc + (row.stake ?? 0), 0);
    const avgOdds =
      settledBetsRows.reduce((acc, row) => acc + (row.odds ?? 0), 0) /
      settledBetsRows.length;
    return {
      count: settledBetsRows.length,
      wins,
      profit_eur: profit,
      roi_pct: totalStake > 0 ? (profit / totalStake) * 100 : 0,
      avg_odds: avgOdds,
    };
  }, [settledBetsRows]);

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
  const metricsSnapshotSummary = payload?.metrics_snapshot_summary ?? {
    realized_count: null,
    realized_profit_eur: null,
    realized_roi: null,
    realized_win_rate: null,
    realized_sharpe: null,
    ev_mean: null,
    eval_base_date_max: null,
  };
  const metricsSnapshotSource =
    dashboardState?.sources?.metrics_snapshot ??
    payload?.sources?.metrics_snapshot ??
    summary?.source?.metrics_snapshot_source ??
    "metrics_snapshot.json";
  const betLogFlatSource = dashboardState?.sources?.bet_log ?? "bet_log_flat_live.csv";
  const localMatchedGamesSource =
    dashboardState?.sources?.local_matched ?? "local_matched_games_latest.csv";
  const combinedSource = dashboardState?.sources?.combined ?? "combined_latest.csv";
  const summaryAsOfDate =
    dashboardState?.as_of_date ?? payload?.as_of_date ?? summary?.as_of_date ?? "—";
  const overallAccuracyPct = fmtPercent(summaryStats.overall_accuracy * 100, 2);
  const windowSize =
    dashboardState?.window_size ||
    windowInfo.size ||
    calibrationMetrics.windowSize ||
    summaryStats.total_games ||
    200;
  const windowStartLabel =
    dashboardState?.window_start ?? windowInfo.start ?? summary?.window_start ?? "—";
  const windowEndLabel =
    dashboardState?.window_end ?? windowInfo.end ?? summary?.window_end ?? summaryAsOfDate ?? "—";
  const windowGamesLabel = windowInfo.games_count ?? windowSize;
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

  const topHomeTeams = useMemo(() => {
    return [...homeWinRatesLast20]
      .filter((team) => team.homeWinRate > 0.5)
      .sort((a, b) => b.homeWinRate - a.homeWinRate);
  }, [homeWinRatesLast20]);

  const settledSimulatedBetsCount = localMatchedGamesCount;
  const strategySubsetAvailable = settledSimulatedBetsCount > 0;
  const strategySubsetWins = strategySubsetAvailable
    ? localMatchedGamesRows.filter((game) => game.win === 1).length
    : 0;
  const strategyLatestRowDate = useMemo(() => {
    if (localMatchedGamesRows.length === 0) {
      return null;
    }
    const dates = localMatchedGamesRows.map((row) => row.date).filter(Boolean);
    if (dates.length === 0) {
      return null;
    }
    return [...dates].sort().at(-1) ?? null;
  }, [localMatchedGamesRows]);

  const localParamsSummary = useMemo(() => {
    const count = settledSimulatedBetsCount;
    if (count === 0) {
      return {
        totalBets: 0,
        totalProfitEur: 0,
        roiPct: 0,
        avgEvPer100: 0,
        avgOdds: 0,
      };
    }
    const totals = localMatchedGamesRows.reduce(
      (acc, row) => {
        acc.profit += row.pnl ?? 0;
        acc.ev += row.ev_eur_per_100 ?? 0;
        acc.odds += row.odds_1 ?? 0;
        return acc;
      },
      { profit: 0, ev: 0, odds: 0 },
    );
    const totalStake = bankrollLast200.stake * count;
    const roiPct = totalStake > 0 ? (totals.profit / totalStake) * 100 : 0;
    return {
      totalBets: count,
      totalProfitEur: totals.profit,
      roiPct,
      avgEvPer100: totals.ev / count,
      avgOdds: totals.odds / count,
    };
  }, [bankrollLast200.stake, localMatchedGamesRows, settledSimulatedBetsCount]);

  const strategyAsOfDate =
    strategyLatestRowDate ??
    dashboardState?.strategy_as_of_date ??
    strategySummary.asOfDate ??
    "—";
  const showRiskMetrics = shouldShowRiskMetrics(settledSimulatedBetsCount);
  const localMatchedGamesProfitSumDisplay = localParamsSummary.totalProfitEur;

  const formatSigned = (value: number | null | undefined) => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return "—";
    }
    const sign = value >= 0 ? "+" : "-";
    return `${sign}€${fmtNumber(Math.abs(value), 2)}`;
  };

  return (
    <>
      {/* Top bar */}
      <section className="container mx-auto px-4 py-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img
                src={`${baseUrl}favicon.ico`}
                alt="Hoops Insight"
                className="h-8 w-8 rounded-md border border-border"
              />
              <span className="text-lg font-semibold tracking-tight">Hoops Insight</span>
            </div>
            <button className="rounded-full border border-border px-4 py-1 text-sm font-medium">
              Dashboard
            </button>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2 text-xs text-muted-foreground">
            <span className="rounded-full border border-border px-3 py-1">
              Historical only
            </span>
            <span className="rounded-full border border-border px-3 py-1">
              Last update: {summaryAsOfDate}
            </span>
            <span className="rounded-full border border-border px-3 py-1">
              Window: {windowSize} games
            </span>
          </div>
          {loadError && (
            <p className="text-sm text-red-400">Data unavailable: {loadError}</p>
          )}
        </div>
      </section>

      {/* Context & Assumptions */}
      <section className="container mx-auto px-4 py-6">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-3">Context &amp; Assumptions</h2>
          <div className="text-sm text-muted-foreground space-y-3">
            <div>
              <span className="font-medium text-foreground">Active Filters (effective)</span>
              <div className="text-foreground">
                {`${activeFiltersEffective} | window ${windowSize} (${windowStartLabel} → ${windowEndLabel})`}
              </div>
            </div>
            <div className="text-foreground">Params used: {paramsUsedLabel}</div>
            <div className="text-foreground">Params source: {paramsSourceLabel}</div>
            <p>
              Historical results and statistical summaries only; no future predictions are shown.
            </p>
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer">metrics_snapshot</summary>
              <div className="mt-2 flex flex-wrap gap-3">
                <span>Source: {metricsSnapshotSource}</span>
                <span>Realized count: {metricsSnapshotSummary.realized_count ?? "—"}</span>
                <span>
                  Realized profit:{" "}
                  {metricsSnapshotSummary.realized_profit_eur !== null
                    ? fmtCurrencyEUR(metricsSnapshotSummary.realized_profit_eur, 2)
                    : "—"}
                </span>
                <span>
                  Realized ROI:{" "}
                  {metricsSnapshotSummary.realized_roi !== null
                    ? fmtPercent(metricsSnapshotSummary.realized_roi * 100, 2)
                    : "—"}
                </span>
                <span>
                  Realized win rate:{" "}
                  {metricsSnapshotSummary.realized_win_rate !== null
                    ? fmtPercent(metricsSnapshotSummary.realized_win_rate * 100, 2)
                    : "—"}
                </span>
                <span>
                  Realized Sharpe:{" "}
                  {metricsSnapshotSummary.realized_sharpe !== null
                    ? fmtNumber(metricsSnapshotSummary.realized_sharpe, 3)
                    : "—"}
                </span>
                <span>Snapshot as of: {metricsSnapshotSummary.eval_base_date_max ?? "—"}</span>
              </div>
            </details>
          </div>
        </div>
      </section>

      {/* Window performance (model) */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Window Performance (Model)</h2>
          <p className="text-sm text-muted-foreground">
            Source: {combinedSource} (played games only, windowed).
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Overall Accuracy"
            value={overallAccuracyPct}
            subtitle={
              <div>Window games: {windowGamesLabel}</div>
            }
            icon={<Target className="w-6 h-6" />}
          />

          <StatCard
            title="Calibration (Brier)"
            value={fmtNumber(calibrationMetrics.brierAfter, 3)}
            subtitle={`Before: ${fmtNumber(calibrationMetrics.brierBefore, 3)}`}
            icon={<BarChart3 className="w-6 h-6" />}
          />

          <StatCard
            title="LogLoss"
            value={fmtNumber(calibrationMetrics.logLossAfter, 3)}
            subtitle={`Before: ${fmtNumber(calibrationMetrics.logLossBefore, 3)}`}
            icon={<TrendingUp className="w-6 h-6" />}
          />

          <StatCard
            title="ECE"
            value={fmtNumber(calibrationMetrics.ece, 3)}
            subtitle={`Before: ${fmtNumber(calibrationMetrics.ece, 3)}`}
            icon={<Activity className="w-6 h-6" />}
          />
        </div>
      </section>

      {/* Strategy (simulated) */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Strategy (Simulated on Window Subset)</h2>
          <p className="text-sm text-muted-foreground">
            Source: {localMatchedGamesSource} (window subset only).
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Strategy matches in window"
            value={`${settledSimulatedBetsCount}`}
            subtitle={`Window games: ${windowGamesLabel}`}
            icon={<Target className="w-6 h-6" />}
          />
          <StatCard
            title="Wins / Win rate"
            value={`${strategySubsetWins} / ${fmtPercent(
              strategySubsetAvailable ? (strategySubsetWins / settledSimulatedBetsCount) * 100 : 0,
              1,
            )}`}
            subtitle={`Strategy as of ${strategyAsOfDate}`}
            icon={<TrendingUp className="w-6 h-6" />}
          />
          <StatCard
            title="Bankroll (Last 200 Window)"
            value={fmtCurrencyEUR(bankrollLast200.bankroll, 2)}
            subtitle={`Net P/L: ${fmtCurrencyEUR(bankrollLast200.net_pl, 2)}`}
            icon={<Activity className="w-6 h-6" />}
          />
          <StatCard
            title="ROI / Sharpe / Max DD"
            value={fmtPercent(localParamsSummary.roiPct, 2)}
            subtitle={`Sharpe: ${
              showRiskMetrics && typeof strategySummary.sharpeStyle === "number"
                ? fmtNumber(strategySummary.sharpeStyle, 2)
                : "—"
            } • DD: ${
              showRiskMetrics ? fmtCurrencyEUR(summary?.kpis?.max_drawdown_eur, 0) : "—"
            }`}
            icon={<BarChart3 className="w-6 h-6" />}
          />
        </div>
      </section>

      {/* Local matched games overview */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">
            LOCAL MATCHED GAMES (Window)
          </h2>
          <p className="text-sm text-muted-foreground mb-2">
            n={settledSimulatedBetsCount} • Wins={strategySubsetWins} • P/L{" "}
            {formatSigned(localMatchedGamesProfitSumDisplay)}
          </p>
          <p className="text-xs text-muted-foreground mb-6">
            Source: {localMatchedGamesSource || "missing"} • Window:{" "}
            {windowStartLabel} → {windowEndLabel}
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
            Source: {betLogFlatSource}, settled against {combinedSource} outcomes.
          </p>
        </div>
        <div className="glass-card p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <StatCard
              title="Bankroll (2026 YTD · Placed Bets)"
              value={fmtCurrencyEUR(bankrollYtd2026.bankroll, 2)}
              subtitle={`Start ${fmtCurrencyEUR(bankrollYtd2026.start, 0)} • Net P/L: ${fmtCurrencyEUR(bankrollYtd2026.net_pl, 2)}`}
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
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">
            Home Win Rates (Window)
          </h2>
          <p className="text-sm text-muted-foreground mb-6">
            Windowed home win rate per team; computed only on home games inside the last {windowSize} games.
          </p>

          <div className="overflow-x-auto">
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
            Showing all teams with home win rate &gt; 50% in the last {windowSize} games.
          </div>
        </div>
      </section>
    </>
  );
};

export default Index;
