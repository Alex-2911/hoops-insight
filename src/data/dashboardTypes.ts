export interface SummaryStats {
  total_games: number;
  overall_accuracy: number;
  as_of_date: string;
  window_size: number;
}

export interface RiskMetrics {
  sharpe: number | null;
}

// Derived from bet_log_flat_live settled via combined_*; independent of strategy params.
export interface BetsSettledOverview {
  year: number;
  source_type: string;
  source_file: string;
  note?: string | null;
  settled_bets: number;
  wins: number;
  win_rate: number | null;
  profit_eur: number | null;
  roi_pct: number | null;
  avg_odds: number | null;
  avg_stake_eur: number | null;
}

// Active filters reflect params_used and should match active_filters_human.
export interface ActiveFilters {
  window_size: number;
  window_start: string;
  window_end: string;
  params_source: string;
  home_win_rate_min: number;
  odds_min: number;
  odds_max: number;
  prob_min: number;
  min_ev: number;
}

export interface SummaryPayload {
  last_run: string;
  as_of_date: string;
  summary_stats: SummaryStats;
  active_filters?: ActiveFilters;
  active_filters_human?: string;
  // Informational only (LOCAL/GLOBAL); may be null if snapshot lacks it.
  params_used_type?: string;
  ytd_source?: {
    type: string;
    file?: string;
  };
  ytd_note?: string | null;
  bets_2026_settled_overview?: BetsSettledOverview;
  strategy_subset_in_window?: {
    matches: number;
    wins: number;
  };
  bankroll?: {
    window_200: {
      start: number;
      flat_stake: number;
      bankroll_eur: number | null;
    };
    ytd_2026: {
      start: number;
      flat_stake: number;
      bankroll_eur: number | null;
    };
  };
  kpis: {
    total_bets: number;
    win_rate: number;
    avg_profit_per_bet_eur: number | null;
    max_drawdown_eur: number | null;
    max_drawdown_pct: number | null;
    strategy_matched_games?: number;
    strategy_matched_wins?: number;
    strategy_win_rate?: number;
    bankroll_window_end?: number | null;
    bankroll_window_pnl_sum?: number | null;
    bankroll_window_trades?: number;
    bankroll_window_sharpe?: number | null;
    bankroll_window_max_dd_eur?: number | null;
    bankroll_window_max_dd_pct?: number | null;
    bankroll_ytd_2026_eur?: number | null;
    bankroll_ytd_2026_pnl_sum?: number | null;
    bankroll_ytd_2026_trades?: number;
    bankroll_last_200_eur?: number | null;
    net_pl_last_200_eur?: number | null;
    bankroll_2026_ytd_eur?: number | null;
    net_pl_2026_ytd_eur?: number | null;
    bankroll_start_eur?: number;
    flat_stake_eur?: number;
    bankroll_year?: number;
  };
  risk_metrics: RiskMetrics;
  source: {
    combined_file: string;
    bet_log_file?: string;
    metrics_snapshot_file?: string;
    local_matched_games_source?: string;
    params_source?: string;
    bet_history_source?: string;
  };
}

export interface HistoricalStat {
  date: string;
  accuracy: number;
  totalGames: number;
  correctGames: number;
}

export interface AccuracyThresholdStat {
  label: string;
  thresholdType: "gt" | "lt";
  threshold: number;
  accuracy: number;
  sampleSize: number;
}

export interface HomeWinRate {
  team: string;
  totalGames: number;
  totalHomeGames: number;
  homeWins: number;
  homeWinRate: number;
}

export interface CalibrationMetrics {
  asOfDate: string;
  brierBefore: number;
  brierAfter: number;
  logLossBefore: number;
  logLossAfter: number;
  fittedGames: number;
}

export interface CalibrationBin {
  bin_center: number;
  n: number;
  avg_pred: number;
  avg_outcome: number;
}

export interface CalibrationQuality {
  window_size: number;
  as_of_date: string;
  fitted_games: number;
  brier_before: number | null;
  brier_after: number | null;
  logloss_before: number | null;
  logloss_after: number | null;
  ece_before: number | null;
  ece_after: number | null;
  calibration_slope_before: number | null;
  calibration_slope_after: number | null;
  calibration_intercept_before: number | null;
  calibration_intercept_after: number | null;
  avg_pred_before: number | null;
  avg_pred_after: number | null;
  base_rate: number | null;
  n_bins: number;
  binning_method: string;
  reliability_bins_before: CalibrationBin[];
  reliability_bins_after: CalibrationBin[];
}

export interface StrategySummary {
  asOfDate: string;
  totalBets: number;
  totalStakedEur: number | null;
  totalProfitEur: number | null;
  roiPct: number | null;
  avgStakeEur: number | null;
  avgProfitPerBetEur: number | null;
  winRate: number;
  avgEvPer100: number | null;
  profitMetricsAvailable: boolean;
}

export interface BankrollEntry {
  date: string;
  balance: number;
  betsPlaced: number;
  profit: number;
}

export interface LocalMatchedGameRow {
  date: string;
  home_team: string;
  away_team: string;
  home_win_rate: number | null;
  prob_iso: number | null;
  prob_used: number | null;
  odds_1: number | null;
  ev_per_100: number | null;
  win: number | null;
  pnl: number | null;
}

// Derived from bet_log_flat_live settled via combined_* played results.
export interface SettledBetRow {
  date: string;
  home_team: string;
  away_team: string;
  stake: number;
  odds: number;
  win: number | null;
  pnl: number | null;
}

export interface SettledBetsSummary {
  count: number;
  wins: number;
  win_rate: number | null;
  pnl_sum: number | null;
  roi_pct: number | null;
  avg_odds: number | null;
  avg_stake: number | null;
}

export interface StrategyFilterStats {
  total_games: number;
  params_source: string;
  home_win_rate_min: number;
  odds_min: number;
  odds_max: number;
  prob_min: number;
  min_ev: number;
  passed_home_win_rate: number;
  passed_odds_range: number;
  passed_prob_threshold: number;
  passed_ev_threshold: number;
  matched_games_count: number;
  matched_games_accuracy: number;
}

export interface TablesPayload {
  historical_stats: HistoricalStat[];
  accuracy_threshold_stats: AccuracyThresholdStat[];
  calibration_metrics: CalibrationMetrics;
  calibration_quality: CalibrationQuality;
  home_win_rates_window: HomeWinRate[];
  home_win_rate_threshold?: number;
  home_win_rate_shown_count?: number;
  strategy_filter_stats: StrategyFilterStats;
  strategy_summary: StrategySummary;
  bankroll_history: BankrollEntry[];
  local_matched_games_rows: LocalMatchedGameRow[];
  local_matched_games_count: number;
  local_matched_games_mismatch?: boolean;
  local_matched_games_note?: string | null;
  bets_2026_settled_rows?: SettledBetRow[];
  bets_2026_settled_count?: number;
  bets_2026_settled_summary?: SettledBetsSummary;
}
