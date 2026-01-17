import { useEffect, useMemo, useState } from "react";
import { StatCard } from "@/components/cards/StatCard";
import type {
  SummaryPayload,
  TablesPayload,
} from "@/data/dashboardTypes";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Target, TrendingUp, Activity, BarChart3, Info, BookOpen } from "lucide-react";

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
  }, [baseUrl]);

  const summaryStats = summary?.summary_stats ?? {
    total_games: 0,
    overall_accuracy: 0,
    as_of_date: "—",
    window_size: 0,
  };
  const windowSize = summaryStats.window_size || summaryStats.total_games;
  const strategySubsetInWindow = summary?.strategy_subset_in_window ?? {
    matches: 0,
    wins: 0,
  };

  const historicalStats = tables?.historical_stats ?? [];
  const calibrationMetrics = tables?.calibration_metrics ?? {
    asOfDate: "—",
    brierBefore: 0,
    brierAfter: 0,
    logLossBefore: 0,
    logLossAfter: 0,
    fittedGames: 0,
  };
  const calibrationQuality = tables?.calibration_quality ?? {
    window_size: windowSize,
    as_of_date: "—",
    fitted_games: 0,
    brier_before: null,
    brier_after: null,
    logloss_before: null,
    logloss_after: null,
    ece_before: null,
    ece_after: null,
    calibration_slope_before: null,
    calibration_slope_after: null,
    calibration_intercept_before: null,
    calibration_intercept_after: null,
    avg_pred_before: null,
    avg_pred_after: null,
    base_rate: null,
    n_bins: 10,
    binning_method: "equal_width",
    reliability_bins_before: [],
    reliability_bins_after: [],
  };
  const homeWinRatesWindow = tables?.home_win_rates_window ?? [];
  const homeWinRateThreshold = tables?.home_win_rate_threshold ?? 0.5;
  const homeWinRateShownCount = tables?.home_win_rate_shown_count ?? homeWinRatesWindow.length;
  const strategyFilterStats = tables?.strategy_filter_stats ?? {
    total_games: 0,
    params_source: "default",
    home_win_rate_min: 0,
    odds_min: 0,
    odds_max: 0,
    prob_min: 0,
    min_ev: 0,
    passed_home_win_rate: 0,
    passed_odds_range: 0,
    passed_prob_threshold: 0,
    passed_ev_threshold: 0,
    matched_games_count: 0,
    matched_games_accuracy: 0,
  };
  const localMatchedGamesRows = tables?.local_matched_games_rows ?? [];
  const localMatchedGamesCount = tables?.local_matched_games_count ?? localMatchedGamesRows.length;
  const localMatchedGamesMismatch = tables?.local_matched_games_mismatch ?? false;
  const localMatchedGamesNote = tables?.local_matched_games_note ?? null;
  const settled2026Rows = tables?.bets_2026_settled_rows ?? [];
  const settled2026Count = tables?.bets_2026_settled_count ?? settled2026Rows.length;
  const settled2026Summary = tables?.bets_2026_settled_summary ?? null;
  const strategySummary = tables?.strategy_summary ?? {
    asOfDate: "—",
    totalBets: 0,
    totalStakedEur: null,
    totalProfitEur: null,
    roiPct: null,
    avgStakeEur: null,
    avgProfitPerBetEur: null,
    winRate: 0,
    avgEvPer100: null,
    profitMetricsAvailable: false,
  };
  const bankrollHistory = tables?.bankroll_history ?? [];
  const riskMetrics = summary?.risk_metrics ?? { sharpe: null };
  const maxDrawdownEur = summary?.kpis.max_drawdown_eur ?? null;
  const maxDrawdownPct = summary?.kpis.max_drawdown_pct ?? null;
  const bankrollLast200Eur =
    summary?.kpis.bankroll_last_200_eur ??
    summary?.kpis.bankroll_window_end ??
    summary?.bankroll?.window_200.bankroll_eur ??
    null;
  const netPlLast200Eur =
    summary?.kpis.net_pl_last_200_eur ?? summary?.kpis.bankroll_window_pnl_sum ?? null;
  const bankrollYtdEur =
    summary?.kpis.bankroll_2026_ytd_eur ??
    summary?.kpis.bankroll_ytd_2026_eur ??
    summary?.bankroll?.ytd_2026.bankroll_eur ??
    null;
  const netPlYtdEur =
    summary?.kpis.net_pl_2026_ytd_eur ?? summary?.kpis.bankroll_ytd_2026_pnl_sum ?? null;
  const ytdNote = summary?.ytd_note ?? null;
  const bets2026 = summary?.bets_2026_settled_overview ?? null;
  const showProfitMetrics = strategySummary.profitMetricsAvailable;

  const overallAccuracyPct = (summaryStats.overall_accuracy * 100).toFixed(2);
  const strategyMatchedGames = summary?.kpis.strategy_matched_games ?? strategySubsetInWindow.matches;
  const strategyMatchedWins = summary?.kpis.strategy_matched_wins ?? strategySubsetInWindow.wins;
  const activeFilters = summary?.active_filters ?? null;
  const activeFiltersHuman = summary?.active_filters_human ?? null;
  const paramsUsedType = summary?.params_used_type ?? null;
  const paramsSource = activeFilters?.params_source ?? null;

  const topHomeTeams = useMemo(() => {
    return [...homeWinRatesWindow].sort((a, b) => {
      if (b.homeWinRate !== a.homeWinRate) {
        return b.homeWinRate - a.homeWinRate;
      }
      if (b.totalHomeGames !== a.totalHomeGames) {
        return b.totalHomeGames - a.totalHomeGames;
      }
      return b.homeWins - a.homeWins;
    });
  }, [homeWinRatesWindow]);

  const settled2026Sorted = useMemo(() => {
    return [...settled2026Rows].sort((a, b) => (a.date < b.date ? 1 : -1));
  }, [settled2026Rows]);

  const formatMetric = (value: number | null, digits = 3) =>
    value === null || Number.isNaN(value) ? "N/A" : value.toFixed(digits);
  const formatPct = (value: number | null, digits = 2) =>
    value === null || Number.isNaN(value) ? "N/A" : `${(value * 100).toFixed(digits)}%`;
  const formatFixed = (value: number, digits: number) => value.toFixed(digits);
  const formatTrimmed = (value: number, digits = 2) => {
    const str = value.toFixed(digits).replace(/\.?0+$/, "");
    return str === "" ? "0" : str;
  };
  const formatUnicodeMinus = (value: number, digits = 2, trim = true) => {
    const str = trim ? formatTrimmed(value, digits) : formatFixed(value, digits);
    return str.startsWith("-") ? `−${str.slice(1)}` : str;
  };

  const KPI_HELP: Record<
    string,
    {
      title: string;
      short: string;
      sections: Array<{ label: string; text: string }>;
    }
  > = {
    active_filters: {
      title: "Active Filters (effective)",
      short: "Filters affect the strategy subset only (window). Not YTD.",
      sections: [
        {
          label: "What it is",
          text: "Effective params used to define the strategy subset in the last N games window.",
        },
        {
          label: "Source",
          text: "metrics_snapshot.{json,csv} → params_used (fair-selected if available).",
        },
        {
          label: "Does NOT affect",
          text: "2026 YTD / Settled Bets (placed bets).",
        },
        {
          label: "Fields",
          text: "odds_min/max, prob_min, home_win_rate_min, min_ev.",
        },
        {
          label: "UI",
          text: "active_filters_human is derived from active_filters.",
        },
      ],
    },
    params_source: {
      title: "Params Source",
      short: "Where the effective params were loaded from.",
      sections: [
        {
          label: "What it is",
          text: "File path used to load params_used for Active Filters.",
        },
        {
          label: "Why it matters",
          text: "Verifies whether LOCAL/GLOBAL params were used.",
        },
      ],
    },
    params_used_type: {
      title: "Params Used",
      short: "Informational only when snapshot provides it.",
      sections: [
        {
          label: "Meaning",
          text: "LOCAL or GLOBAL when fair-selection metadata is available.",
        },
      ],
    },
    overall_accuracy: {
      title: "Overall Accuracy",
      short: "Accuracy over the last N played games window.",
      sections: [
        { label: "Source", text: "combined_nba_predictions_* (windowed)." },
        { label: "Why it matters", text: "Model quality independent of betting." },
      ],
    },
    calibration_brier: {
      title: "Calibration (Brier)",
      short: "Lower is better; measures probabilistic accuracy.",
      sections: [
        { label: "Source", text: "combined_nba_predictions_* (windowed)." },
        {
          label: "Why it matters",
          text: "Shows whether predicted probabilities are well calibrated.",
        },
      ],
    },
    bankroll_window: {
      title: "Bankroll (Last 200 Window)",
      short: "Simulated bankroll on local_matched_games in the window.",
      sections: [
        { label: "Source", text: "local_matched_games restricted to window membership." },
        {
          label: "Why it matters",
          text: "Strategy subset performance, not actual placed bets.",
        },
      ],
    },
    bankroll_ytd: {
      title: "Bankroll (2026 YTD · Placed Bets)",
      short: "Real placed bets from bet_log_flat_live.csv; settled via combined_* results.",
      sections: [
        {
          label: "Source",
          text: "bet_log_flat_live.csv (placed) + combined_* (played results).",
        },
        {
          label: "Settlement",
          text: "Win → stake*(odds-1); loss → -stake.",
        },
        {
          label: "Only counts",
          text: "Settled rows (must have outcome).",
        },
        {
          label: "Fallback",
          text: "windowed_local_matched_fallback (shown via note fields).",
        },
      ],
    },
    settled_bets: {
      title: "Settled Bets (2026)",
      short: "Same pipeline as YTD; avg odds contextualize ROI.",
      sections: [
        {
          label: "Metrics",
          text: "count, wins, win_rate, profit, roi, avg_odds, avg_stake.",
        },
        {
          label: "Why avg odds",
          text: "Higher odds imply higher variance, ROI less stable.",
        },
      ],
    },
    local_matched: {
      title: "Local Matched Games (Window)",
      short: "Simulated subset restricted to the last N window.",
      sections: [
        {
          label: "Source",
          text: "local_matched_games_YYYY-MM-DD.csv restricted to window membership.",
        },
        {
          label: "Why it matters",
          text: "Explains strategy table vs YTD differences.",
        },
      ],
    },
    settled_table: {
      title: "Settled Bets Table (2026)",
      short: "Placed & settled only; independent of strategy filters.",
      sections: [
        {
          label: "Source",
          text: "bet_log_flat_live.csv settled via combined_*.",
        },
        {
          label: "Why it matters",
          text: "Shows the actual placed bets behind YTD.",
        },
      ],
    },
    calibration_quality: {
      title: "Calibration Quality",
      short: "How well predicted probabilities reflect actual outcomes.",
      sections: [
        {
          label: "What it is",
          text: "Calibration metrics computed on the last N played games only.",
        },
        {
          label: "Why it matters",
          text: "Shows whether probabilities align with real outcomes.",
        },
      ],
    },
    before_after: {
      title: "Before / After",
      short: "Calibration metrics before and after probability calibration.",
      sections: [
        {
          label: "Meaning",
          text: "Identical values indicate no calibration was applied in this window.",
        },
      ],
    },
    metrics_snapshot: {
      title: "metrics_snapshot",
      short: "Snapshot of fair-selected params_used and window metadata.",
      sections: [
        {
          label: "Used for",
          text: "Active Filters (effective) and params_used_type.",
        },
      ],
    },
  };

  const InfoButton = ({
    id,
    align = "center",
    className = "",
  }: {
    id: keyof typeof KPI_HELP;
    align?: "center" | "start" | "end";
    className?: string;
  }) => {
    const item = KPI_HELP[id];
    if (!item) {
      return null;
    }
    return (
      <Popover>
        <Tooltip>
          <TooltipTrigger asChild>
            <PopoverTrigger asChild>
              <button
                type="button"
                aria-label={`Info: ${item.title}`}
                className={`inline-flex items-center text-muted-foreground hover:text-foreground ${className}`}
              >
                <Info className="h-4 w-4" />
              </button>
            </PopoverTrigger>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-[320px] text-xs leading-relaxed">
            {item.short}
          </TooltipContent>
        </Tooltip>
        <PopoverContent align={align} className="max-w-[360px] space-y-3">
          <div className="text-sm font-semibold text-foreground">{item.title}</div>
          {item.sections.map((section) => (
            <div key={section.label} className="space-y-1">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                {section.label}
              </div>
              <div className="text-sm text-foreground">{section.text}</div>
            </div>
          ))}
        </PopoverContent>
      </Popover>
    );
  };

  const getCalibratedPair = (
    before: number | null,
    after: number | null,
    formatter: (value: number | null) => string,
  ) => {
    const beforeValid = before !== null && !Number.isNaN(before);
    const afterValid = after !== null && !Number.isNaN(after);
    if (beforeValid && afterValid) {
      return { text: `${formatter(before)} / ${formatter(after)}`, footer: "Before / After", suffix: null };
    }
    if (afterValid) {
      return { text: `${formatter(after)}`, footer: null, suffix: "(calibrated)" };
    }
    if (beforeValid) {
      return { text: `${formatter(before)}`, footer: null, suffix: "(raw)" };
    }
    return { text: "N/A", footer: null, suffix: null };
  };

  // Active filters are derived from summary.active_filters_human (params_used), with fallback formatting.
  const activeFiltersText = (() => {
    if (activeFiltersHuman && activeFiltersHuman.trim()) {
      return activeFiltersHuman.trim();
    }
    if (!activeFilters) {
      return null;
    }
    return [
      `HW ≥ ${formatFixed(activeFilters.home_win_rate_min, 2)}`,
      `odds ${formatFixed(activeFilters.odds_min, 2)}–${formatFixed(activeFilters.odds_max, 2)}`,
      `p ≥ ${formatFixed(activeFilters.prob_min, 2)}`,
      `EV > ${formatUnicodeMinus(activeFilters.min_ev, 1, true)}`,
      `window ${activeFilters.window_size} (${activeFilters.window_start} → ${activeFilters.window_end})`,
    ].join(" | ");
  })();

  const matchedGamesRows = useMemo(() => {
    return [...localMatchedGamesRows].sort((a, b) => (a.date < b.date ? 1 : -1));
  }, [localMatchedGamesRows]);

  return (
    <TooltipProvider delayDuration={250}>
      <>
      <div className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur">
        <div className="container mx-auto flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="text-lg font-semibold">Hoops Insight</span>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2 text-xs text-muted-foreground">
            <span className="rounded-full bg-muted px-3 py-1 text-xs font-medium text-foreground">
              Historical only
            </span>
            <span className="rounded-full border border-border px-3 py-1">
              As of: {summaryStats.as_of_date}
            </span>
            <span className="rounded-full border border-border px-3 py-1">
              Window: {windowSize} games
            </span>
            <Drawer>
              <DrawerTrigger className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-foreground hover:bg-muted">
                <BookOpen className="h-3.5 w-3.5" />
                How to read
              </DrawerTrigger>
              <DrawerContent>
                <DrawerHeader>
                  <DrawerTitle>How to read this dashboard</DrawerTitle>
                  <DrawerDescription>
                    Quick guide to window vs strategy vs placed bets.
                  </DrawerDescription>
                </DrawerHeader>
                <div className="px-4 pb-6 text-sm text-muted-foreground">
                  <ul className="list-disc space-y-2 pl-4">
                    <li>Window metrics = model quality on the last N played games.</li>
                    <li>Strategy = simulated subset using active filters.</li>
                    <li>YTD = actual placed bets from the bet log (independent of filters).</li>
                    <li>Settled = placed bets settled via played results.</li>
                  </ul>
                </div>
              </DrawerContent>
            </Drawer>
          </div>
        </div>
      </div>

      {loadError && (
        <section className="container mx-auto px-4 py-6">
          <div className="rounded-lg border border-border p-4 text-sm text-red-400">
            Data unavailable: {loadError}
          </div>
        </section>
      )}

      {/* Section A — Context & Assumptions */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-bold">Context &amp; Assumptions</h2>
            <span className="inline-flex items-center gap-2 rounded-full bg-muted px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              metrics_snapshot
              <InfoButton id="metrics_snapshot" className="text-muted-foreground" />
            </span>
          </div>
          <div className="mt-6 rounded-lg border border-border p-4">
            <div className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              Active Filters (effective)
              <InfoButton id="active_filters" />
            </div>
            <div className="mt-2 text-base font-semibold text-foreground">
              {activeFiltersText ?? "—"}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                {paramsUsedType ? `Params used: ${paramsUsedType} (fair-selected)` : "Params used: Historical"}
                <InfoButton id="params_used_type" className="text-muted-foreground" />
              </span>
              <span className="inline-flex items-center gap-1">
                {paramsSource ? `Params source: ${paramsSource}` : "Params source: —"}
                <InfoButton id="params_source" className="text-muted-foreground" />
              </span>
            </div>
            <div className="mt-1 text-[11px] text-muted-foreground">
              Parameters are shown for transparency only. No live strategy selection or optimization is applied.
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              Active Filters apply only to the simulated strategy subset (Local Matched Games). Placed Bets (2026 YTD)
              reflect real bets and are not filtered or re-optimized.
            </div>
          </div>
        </div>
      </section>

      {/* Section B — Window Performance (Model) */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-bold">Window Performance (Model)</h2>
            <span className="rounded-full bg-muted px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              combined_*
            </span>
          </div>
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <StatCard
              title={
                <span className="inline-flex items-center gap-2">
                  Overall Accuracy
                  <InfoButton id="overall_accuracy" />
                </span>
              }
              value={`${overallAccuracyPct}%`}
              subtitle={`Window games: ${summaryStats.total_games}`}
              icon={<Target className="w-6 h-6" />}
            />
            <StatCard
              title={
                <span className="inline-flex items-center gap-2">
                  Calibration (Brier)
                  <InfoButton id="calibration_brier" />
                </span>
              }
              value={formatMetric(calibrationMetrics.brierAfter, 3)}
              subtitle={`Before: ${formatMetric(calibrationMetrics.brierBefore, 3)}`}
              icon={<BarChart3 className="w-6 h-6" />}
            />
            <StatCard
              title="LogLoss (After)"
              value={formatMetric(calibrationQuality.logloss_after, 3)}
              subtitle={`Before: ${formatMetric(calibrationQuality.logloss_before, 3)}`}
              icon={<TrendingUp className="w-6 h-6" />}
            />
            <StatCard
              title="ECE (After)"
              value={formatMetric(calibrationQuality.ece_after, 3)}
              subtitle={`Before: ${formatMetric(calibrationQuality.ece_before, 3)}`}
              icon={<Activity className="w-6 h-6" />}
            />
          </div>

          <Accordion type="single" collapsible className="mt-6">
            <AccordionItem value="calibration-quality">
              <AccordionTrigger>
                <span className="inline-flex items-center gap-2">
                  Calibration Quality
                  <InfoButton id="calibration_quality" />
                </span>
              </AccordionTrigger>
              <AccordionContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {(() => {
                    const brierPair = getCalibratedPair(
                      calibrationQuality.brier_before,
                      calibrationQuality.brier_after,
                      (value) => formatMetric(value),
                    );
                    const logLossPair = getCalibratedPair(
                      calibrationQuality.logloss_before,
                      calibrationQuality.logloss_after,
                      (value) => formatMetric(value),
                    );
                    const ecePair = getCalibratedPair(
                      calibrationQuality.ece_before,
                      calibrationQuality.ece_after,
                      (value) => formatMetric(value),
                    );
                    const slopePair = getCalibratedPair(
                      calibrationQuality.calibration_slope_before,
                      calibrationQuality.calibration_slope_after,
                      (value) => formatMetric(value),
                    );
                    const interceptPair = getCalibratedPair(
                      calibrationQuality.calibration_intercept_before,
                      calibrationQuality.calibration_intercept_after,
                      (value) => formatMetric(value),
                    );
                    const avgPredPair = getCalibratedPair(
                      calibrationQuality.avg_pred_before,
                      calibrationQuality.avg_pred_after,
                      (value) => formatPct(value),
                    );
                    return (
                      <>
                        <div className="rounded-lg border border-border p-4">
                          <div className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                            <span>Brier Score</span>
                            <Info
                              className="w-3.5 h-3.5"
                              title="Brier misst die Genauigkeit von Wahrscheinlichkeiten (0 = perfekt). Grob: <0.20 gut, 0.20-0.25 okay, >0.25 eher schwach (kontextabhaengig)."
                            />
                          </div>
                        <div className="text-lg font-bold">
                          {brierPair.text} {brierPair.suffix}
                        </div>
                        {brierPair.footer && (
                          <div className="mt-1 inline-flex items-center gap-2 text-xs text-muted-foreground">
                            <span>{brierPair.footer}</span>
                            <InfoButton id="before_after" />
                          </div>
                        )}
                      </div>
                        <div className="rounded-lg border border-border p-4">
                          <div className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                            <span>Log Loss</span>
                            <Info
                              className="w-3.5 h-3.5"
                              title="Log Loss bestraft zu sichere Fehlprognosen staerker. Niedriger = besser."
                            />
                          </div>
                          <div className="text-lg font-bold">
                            {logLossPair.text} {logLossPair.suffix}
                          </div>
                          {logLossPair.footer && (
                            <div className="text-xs text-muted-foreground mt-1">{logLossPair.footer}</div>
                          )}
                        </div>
                        <div className="rounded-lg border border-border p-4">
                          <div className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                            <span>ECE</span>
                            <Info
                              className="w-3.5 h-3.5"
                              title="Expected Calibration Error: Differenz zwischen vorhergesagter Wahrscheinlichkeit und echter Trefferquote. Niedriger = besser."
                            />
                          </div>
                          <div className="text-lg font-bold">
                            {ecePair.text} {ecePair.suffix}
                          </div>
                          {ecePair.footer && (
                            <div className="text-xs text-muted-foreground mt-1">{ecePair.footer}</div>
                          )}
                        </div>
                        <div className="rounded-lg border border-border p-4">
                          <div className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                            <span>Calibration Slope</span>
                            <Info
                              className="w-3.5 h-3.5"
                              title="Ideal: Slope ~1, Intercept ~0. Abweichungen deuten auf Over/Underconfidence hin."
                            />
                          </div>
                          <div className="text-lg font-bold">
                            {slopePair.text} {slopePair.suffix}
                          </div>
                          {slopePair.footer && (
                            <div className="text-xs text-muted-foreground mt-1">{slopePair.footer}</div>
                          )}
                        </div>
                        <div className="rounded-lg border border-border p-4">
                          <div className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                            <span>Calibration Intercept</span>
                            <Info
                              className="w-3.5 h-3.5"
                              title="Ideal: Slope ~1, Intercept ~0. Abweichungen deuten auf Over/Underconfidence hin."
                            />
                          </div>
                          <div className="text-lg font-bold">
                            {interceptPair.text} {interceptPair.suffix}
                          </div>
                          {interceptPair.footer && (
                            <div className="text-xs text-muted-foreground mt-1">{interceptPair.footer}</div>
                          )}
                        </div>
                        <div className="rounded-lg border border-border p-4">
                          <div className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                            <span>Avg Predicted Prob</span>
                            <Info
                              className="w-3.5 h-3.5"
                              title="Avg Predicted Prob = durchschnittlich vorhergesagte Home-Win-Wahrscheinlichkeit."
                            />
                          </div>
                          <div className="text-lg font-bold">
                            {avgPredPair.text} {avgPredPair.suffix}
                          </div>
                          {avgPredPair.footer && (
                            <div className="text-xs text-muted-foreground mt-1">{avgPredPair.footer}</div>
                          )}
                        </div>
                        <div className="rounded-lg border border-border p-4">
                          <div className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                            <span>Base Rate</span>
                            <Info
                              className="w-3.5 h-3.5"
                              title="Base Rate = tatsaechliche Home-Win-Rate im Window."
                            />
                          </div>
                          <div className="text-lg font-bold">{formatPct(calibrationQuality.base_rate)}</div>
                          <div className="text-xs text-muted-foreground mt-1">Actual Win %</div>
                        </div>
                      </>
                    );
                  })()}
                </div>
                <div className="text-xs text-muted-foreground mt-4">
                  Fitted games: {calibrationQuality.fitted_games} • Window: {calibrationQuality.window_size} • As of{" "}
                  {calibrationQuality.as_of_date}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </section>

      {/* Section C — Strategy (Simulated on Window Subset) */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-bold">Strategy (Simulated on Window Subset)</h2>
            <span className="rounded-full bg-muted px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              local_matched_games
            </span>
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
            Simulated performance on local_matched_games restricted to the window (not actual placed bets).
          </p>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Strategy matches in window</div>
              <div className="text-2xl font-bold">{strategyMatchedGames}</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Wins / Win rate</div>
              <div className="text-2xl font-bold">
                {strategyMatchedWins} / {(strategyFilterStats.matched_games_accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                Bankroll (Last 200 Window)
                <InfoButton id="bankroll_window" />
              </div>
              <div className="text-2xl font-bold">
                {bankrollLast200Eur === null ? "—" : `€${bankrollLast200Eur.toFixed(2)}`}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Net P/L: {netPlLast200Eur === null ? "—" : `€${netPlLast200Eur.toFixed(2)}`}
              </div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">ROI / Sharpe / Max DD</div>
              <div className="text-2xl font-bold">
                {strategySummary.roiPct === null || !showProfitMetrics ? "N/A" : `${strategySummary.roiPct.toFixed(2)}%`}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Sharpe: {riskMetrics.sharpe == null || !showProfitMetrics ? "N/A" : riskMetrics.sharpe.toFixed(2)} •
                DD: {maxDrawdownEur == null || !showProfitMetrics ? "N/A" : `€${maxDrawdownEur.toFixed(0)}`}
              </div>
            </div>
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-between gap-3">
            <h3 className="inline-flex items-center gap-2 text-lg font-semibold">
              LOCAL MATCHED GAMES (Window)
              <InfoButton id="local_matched" />
            </h3>
            <span className="rounded-full bg-muted px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Simulated subset (window-only)
            </span>
            <div className="text-xs text-muted-foreground">
              n={localMatchedGamesCount} • Wins={strategyMatchedWins} • P/L{" "}
              {strategySummary.totalProfitEur == null || !showProfitMetrics
                ? "N/A"
                : `${strategySummary.totalProfitEur >= 0 ? "+" : ""}€${strategySummary.totalProfitEur.toFixed(0)}`}
            </div>
          </div>

          {localMatchedGamesCount === 0 || localMatchedGamesMismatch ? (
            <div className="mt-4 rounded-lg border border-border p-4 text-sm text-muted-foreground">
              {localMatchedGamesNote ?? "Table uses settled rows only; some rows missing in source file."}
            </div>
          ) : (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-background">
                  <tr className="text-left border-b border-border">
                    <th className="py-2 pr-4">Date</th>
                    <th className="py-2 pr-4">Home</th>
                    <th className="py-2 pr-4">Away</th>
                    <th className="py-2 pr-4">Home Win Rate</th>
                    <th className="py-2 pr-4">Prob ISO</th>
                    <th className="py-2 pr-4">Prob Used</th>
                    <th className="py-2 pr-4">Odds</th>
                    <th className="py-2 pr-4">EV €/100</th>
                    <th className="py-2 pr-4">Win</th>
                    <th className="py-2 pr-4">P/L</th>
                  </tr>
                </thead>
                <tbody>
                  {matchedGamesRows.map((row) => {
                    const winLabel = row.win === 1 ? "✅" : "❌";
                    const pnl = `${row.pnl >= 0 ? "+" : ""}${row.pnl.toFixed(2)}`;
                    return (
                      <tr
                        key={`${row.date}-${row.home_team.trim().toUpperCase()}-${row.away_team.trim().toUpperCase()}`}
                        className="border-b border-border/50"
                      >
                        <td className="py-2 pr-4">{row.date}</td>
                        <td className="py-2 pr-4 font-medium">{row.home_team}</td>
                        <td className="py-2 pr-4 font-medium">{row.away_team}</td>
                        <td className="py-2 pr-4">{row.home_win_rate.toFixed(2)}</td>
                        <td className="py-2 pr-4">{row.prob_iso.toFixed(2)}</td>
                        <td className="py-2 pr-4">{row.prob_used.toFixed(2)}</td>
                        <td className="py-2 pr-4">{row.odds_1.toFixed(2)}</td>
                        <td className="py-2 pr-4">{row.ev_per_100.toFixed(2)}</td>
                        <td className="py-2 pr-4">{winLabel}</td>
                        <td className="py-2 pr-4">{pnl}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* Section D — Placed Bets (Real) — 2026 YTD */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-bold">Placed Bets (Real) — 2026 YTD</h2>
            <span className="rounded-full bg-muted px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              bet_log_flat_live
            </span>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <StatCard
              title={
                <span className="inline-flex items-center gap-2">
                  Bankroll (2026 YTD · Placed Bets)
                  <InfoButton id="bankroll_ytd" />
                </span>
              }
              value={bankrollYtdEur === null ? "—" : `€${bankrollYtdEur.toFixed(2)}`}
              subtitle={
                <>
                  <span className="block">
                    Start €1000 • Net P/L 2026:{" "}
                    {netPlYtdEur === null ? "—" : `€${netPlYtdEur.toFixed(2)}`} • €100 flat stake (example)
                  </span>
                  {ytdNote && (
                    <span className="block text-xs text-muted-foreground mt-1">
                      Note: {ytdNote}
                    </span>
                  )}
                </>
              }
              icon={<Activity className="w-6 h-6" />}
            />
            <StatCard
              title={
                <span className="inline-flex items-center gap-2">
                  Settled Bets (2026)
                  <InfoButton id="settled_bets" />
                </span>
              }
              value={
                bets2026?.settled_bets == null || (bets2026.settled_bets === 0 && bets2026.note)
                  ? "—"
                  : `${bets2026.settled_bets}`
              }
              subtitle={
                <>
                  {/* Placed & settled only; independent of strategy filters. */}
                  {/* Avg odds is shown to contextualize ROI; this is not a pick recommendation. */}
                  <span className="block">
                    {`${bets2026?.wins ?? "—"}W-${bets2026?.settled_bets == null ? "—" : bets2026.settled_bets - (bets2026.wins ?? 0)}L • `}
                    {`ROI: ${
                      bets2026?.roi_pct == null
                        ? "N/A"
                        : `${formatUnicodeMinus(bets2026.roi_pct, 2)}%`
                    } • `}
                    {`avg odds: ${
                      bets2026?.avg_odds == null ? "N/A" : formatTrimmed(bets2026.avg_odds, 2)
                    }`}
                  </span>
                  <span className="block text-xs text-muted-foreground mt-1">
                    Source: bet_log_flat_live.csv → settled via combined_*
                  </span>
                  {bets2026?.note && (
                    <span className="block text-xs text-muted-foreground mt-1">
                      Note: {bets2026.note}
                    </span>
                  )}
                </>
              }
              icon={<Target className="w-6 h-6" />}
            />
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-between gap-3">
            <h3 className="inline-flex items-center gap-2 text-lg font-semibold">
              SETTLED BETS (2026)
              <InfoButton id="settled_table" />
            </h3>
            <span className="rounded-full bg-muted px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Real bets (settled via combined_*)
            </span>
            <span className="text-xs text-muted-foreground">Actually placed &amp; settled bets only • Source: bet_log_flat_live.csv</span>
          </div>

          {settled2026Count === 0 ? (
            <div className="mt-4 rounded-lg border border-border p-4 text-sm text-muted-foreground">
              No settled bets for 2026 found.
            </div>
          ) : (
            <div className="mt-4 overflow-x-auto">
              {settled2026Summary && (
                <div className="text-sm text-muted-foreground mb-3">
                  {`${settled2026Summary.count} settled • ${settled2026Summary.wins}W-${
                    settled2026Summary.count - settled2026Summary.wins
                  }L • P/L ${
                    settled2026Summary.pnl_sum == null
                      ? "N/A"
                      : `${settled2026Summary.pnl_sum >= 0 ? "+" : ""}€${formatUnicodeMinus(
                          settled2026Summary.pnl_sum,
                          2,
                          false,
                        )}`
                  } • ROI ${
                    settled2026Summary.roi_pct == null
                      ? "N/A"
                      : `${formatUnicodeMinus(settled2026Summary.roi_pct, 2, false)}%`
                  } • avg odds ${
                    settled2026Summary.avg_odds == null
                      ? "N/A"
                      : formatTrimmed(settled2026Summary.avg_odds, 2)
                  }`}
                </div>
              )}
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-background">
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
                  {settled2026Sorted.map((row) => {
                    const winLabel = row.win === 1 ? "✅" : row.win === 0 ? "❌" : "—";
                    const stakeLabel = `€${row.stake.toFixed(2)}`;
                    const oddsLabel = row.odds.toFixed(2);
                    const pnlLabel =
                      row.pnl == null
                        ? "—"
                        : `${row.pnl >= 0 ? "+" : ""}€${formatUnicodeMinus(row.pnl, 2, false)}`;
                    return (
                      <tr
                        key={`${row.date}-${row.home_team.trim().toUpperCase()}-${row.away_team.trim().toUpperCase()}`}
                        className="border-b border-border/50"
                      >
                        <td className="py-2 pr-4">{row.date}</td>
                        <td className="py-2 pr-4 font-medium">{row.home_team}</td>
                        <td className="py-2 pr-4 font-medium">{row.away_team}</td>
                        <td className="py-2 pr-4">{stakeLabel}</td>
                        <td className="py-2 pr-4">{oddsLabel}</td>
                        <td className="py-2 pr-4">{winLabel}</td>
                        <td className="py-2 pr-4">{pnlLabel}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* Home win rates */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-bold">Home Win Rates (Window)</h2>
            <span className="rounded-full bg-muted px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              combined_*
            </span>
          </div>
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
                    <td className="py-2 pr-4">{(t.homeWinRate * 100).toFixed(0)}%</td>
                    <td className="py-2 pr-4">{t.homeWins}</td>
                    <td className="py-2 pr-4">{t.totalHomeGames}</td>
                    <td className="py-2 pr-4">{t.totalGames}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="text-xs text-muted-foreground mt-4">
            {homeWinRateShownCount === 0
              ? `No teams above ${(homeWinRateThreshold * 100).toFixed(0)}% home win rate in the last ${windowSize} games.`
              : `Showing all teams with home win rate > ${(homeWinRateThreshold * 100).toFixed(0)}% in the last ${windowSize} games.`}
          </div>
        </div>
      </section>
      </>
    </TooltipProvider>
  );
};

export default Index;
