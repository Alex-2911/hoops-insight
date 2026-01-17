import { useEffect, useMemo, useState } from "react";
import { StatCard } from "@/components/cards/StatCard";
import type { DashboardPayload, DashboardState } from "@/data/dashboardTypes";
import { Target, TrendingUp, Activity, BarChart3, Info } from "lucide-react";
import { fmtCurrencyEUR, fmtNumber, fmtPercent } from "@/lib/format";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

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
  const settledBetsRows = tables?.settled_bets_rows ?? [];
  const START_BANKROLL_REAL = 1000;
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
    const avgOdds =
      settledBetsRows.reduce((acc, row) => acc + (row.odds ?? 0), 0) /
      settledBetsRows.length;
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
    settledBetsSummary.count > 0
      ? (settledBetsSummary.wins / settledBetsSummary.count) * 100
      : 0;
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

  const formatSigned = (value: number | null | undefined) => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return "—";
    }
    const sign = value >= 0 ? "+" : "-";
    return `${sign}€${fmtNumber(Math.abs(value), 2)}`;
  };

  return (
    <>
      {/* Context & Assumptions */}
      <section className="container mx-auto px-4 py-6">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-3">Context &amp; Assumptions</h2>
          {loadError && (
            <p className="text-sm text-red-400 mb-3">Data unavailable: {loadError}</p>
          )}
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
                </>
              )}
              value={overallAccuracyPct}
              subtitle={
                <div>Window games: {windowGamesLabel}</div>
              }
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
                </>
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
                </>
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
                </>
              )}
              value={fmtNumber(calibrationMetrics.ece, 3)}
              subtitle={`Before: ${fmtNumber(calibrationMetrics.ece, 3)}`}
              icon={<Activity className="w-6 h-6" />}
            />
          </div>
        </TooltipProvider>
      </section>

      {/* Placed bets overview */}
      <section className="container mx-auto px-4 py-10">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Placed Bets (Real) — Overview</h2>
          <p className="text-sm text-muted-foreground">
            Source: {betLogFlatSource} (settled only).
          </p>
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
            subtitle={`${settledBetsSummary.wins} wins · ${
              settledBetsSummary.count - settledBetsSummary.wins
            } losses`}
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
          <h2 className="text-2xl font-bold">Placed Bets (Real) — 2026 YTD</h2>
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
