export interface SummaryStats {
  total_games: number;
  overall_accuracy: number;
  as_of_date: string;
}

export interface SummaryPayload {
  last_run: string;
  as_of_date: string;
  window_end?: string;
  window_start?: string;
  window_size?: number;
  generated_at?: string;
  real_bets_available?: boolean;
  summary_stats: SummaryStats;
  kpis: {
    total_bets: number;
    win_rate: number;
    roi_pct: number;
    avg_ev_per_100: number;
    avg_profit_per_bet_eur: number;
    max_drawdown_eur: number | null;
    max_drawdown_pct: number | null;
  };
  strategy_summary: {
    totalBets: number;
    totalProfitEur: number;
    roiPct: number;
    avgEvPer100: number;
    winRate: number;
    sharpeStyle: number | null;
    profitMetricsAvailable: boolean;
    asOfDate: string;
  };
  strategy_params: {
    source: string;
    params: Record<string, string | number | boolean | null>;
    params_used: Record<string, string | number | boolean | null>;
    active_filters?: string;
    params_used_label?: string;
  };
  strategy_filter_stats: {
    window_size: number;
    filters: Array<{ label: string; count: number }>;
    matched_games_count: number;
    window_start?: string;
    window_end?: string;
  };
  strategy_counts: {
    window_games_count: number;
    filter_pass_count: number;
    simulated_bets_count: number;
    settled_bets_count: number;
  };
  source: {
    combined_file: string;
    bet_log_file: string;
    bet_log_flat_file?: string;
  };
}

export interface DashboardPayload {
  as_of_date: string;
  window: {
    size: number;
    start?: string | null;
    end?: string | null;
    games_count?: number;
  };
  active_filters_effective: string;
  params_used_label?: string;
  summary: SummaryPayload;
  tables: TablesPayload;
  last_run?: LastRunPayload | null;
  sources?: {
    combined_file: string;
    bet_log_flat: string;
    copied?: Record<string, string>;
  };
}

export interface DashboardState {
  as_of_date: string;
  window_size: number;
  window_start?: string | null;
  window_end?: string | null;
  active_filters_text: string;
  params_used_label: string;
  params_source_label: string;
  strategy_as_of_date?: string | null;
  strategy_matches_window?: number;
  last_update_utc: string;
  sources: {
    combined: string;
    bet_log: string;
  };
}

export interface StrategyParamsFile {
  params_used_label?: string;
  params_used: Record<string, string | number | boolean | null>;
}

export interface LastRunPayload {
  last_run: string;
  as_of_date: string;
  window_end?: string;
  model_window_end?: string;
  generated_at?: string;
  run_timestamp?: string;
  active_filters?: string | null;
  active_filters_human?: string | null;
  strategy_filter_stats?: {
    filters?: Array<{ label: string; count: number }>;
  };
  records?: {
    played_games?: number;
    bet_log_rows?: number;
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
  totalLast20Games: number;
  totalHomeGames: number;
  homeWins: number;
  homeWinRate: number;
}

export interface HomeWinRateWindowRow {
  team: string;
  homeWinRate: number;
  homeWins: number;
  homeGames: number;
  windowGames: number;
}

export interface CalibrationMetrics {
  asOfDate: string;
  brierBefore: number;
  brierAfter: number;
  logLossBefore: number;
  logLossAfter: number;
  fittedGames: number;
  ece: number;
  calibrationSlope: number;
  calibrationIntercept: number;
  avgPredictedProb: number;
  baseRate: number;
  actualWinPct: number;
  windowSize: number;
}

export interface BetLogSummary {
  asOfDate: string;
  totalBets: number;
  totalStakedEur: number;
  totalProfitEur: number;
  roiPct: number;
  avgStakeEur: number;
  avgProfitPerBetEur: number;
  winRate: number;
  avgEvPer100: number;
}

export interface BankrollEntry {
  date: string;
  balance: number;
  betsPlaced: number;
  profit: number;
}

export interface SettledBet {
  date: string;
  home_team: string;
  away_team: string;
  pick_team: string;
  odds: number;
  stake: number;
  win: number;
  pnl: number;
}

export interface LocalMatchedGameRow {
  date: string;
  home_team: string;
  away_team: string;
  home_win_rate: number;
  prob_iso: number;
  prob_used: number;
  odds_1: number;
  ev_eur_per_100: number;
  win: number;
  pnl: number;
}

export interface BankrollSummary {
  start: number;
  stake: number;
  net_pl: number;
  bankroll: number;
}

export interface TablesPayload {
  historical_stats: HistoricalStat[];
  accuracy_threshold_stats: AccuracyThresholdStat[];
  calibration_metrics: CalibrationMetrics;
  home_win_rates_last20: HomeWinRate[];
  home_win_rates_window?: HomeWinRateWindowRow[];
  bet_log_summary: BetLogSummary;
  bankroll_history: BankrollEntry[];
  bankroll_ytd_2026: BankrollSummary;
  settled_bets_rows: SettledBet[];
  settled_bets_summary: {
    count: number;
    wins: number;
    profit_eur: number;
    roi_pct: number;
    avg_odds: number;
  };
  local_matched_games_rows?: LocalMatchedGameRow[];
  local_matched_games_count?: number;
  local_matched_games_profit_sum_table?: number;
  local_matched_games_note?: string;
  local_matched_games_mismatch?: boolean;
}
