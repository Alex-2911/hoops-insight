export interface SummaryStats {
  total_games: number;
  overall_accuracy: number;
  as_of_date: string;
}

export interface SummaryPayload {
  last_run: string;
  as_of_date: string;
  real_bets_available?: boolean;
  summary_stats: SummaryStats;
  kpis: {
    total_bets: number;
    win_rate: number;
    roi_pct: number;
    avg_ev_per_100: number;
    avg_profit_per_bet_eur: number;
    max_drawdown_eur: number;
    max_drawdown_pct: number;
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
    active_filters: string;
  };
  strategy_filter_stats: {
    window_size: number;
    filters: Array<{ label: string; count: number }>;
    matched_games_count: number;
  };
  source: {
    combined_file: string;
    bet_log_file: string;
    bet_log_flat_file?: string;
    metrics_snapshot_source: string;
  };
}

export interface LastRunPayload {
  last_run: string;
  as_of_date: string;
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

export interface LocalMatchedGame {
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
  bet_log_summary: BetLogSummary;
  bankroll_history: BankrollEntry[];
  local_matched_games_rows: LocalMatchedGame[];
  local_matched_games_count: number;
  local_matched_games_profit_sum_table: number;
  local_matched_games_mismatch: boolean;
  local_matched_games_note: string;
  local_matched_games_source?: string;
  bankroll_last_200: BankrollSummary;
  bankroll_ytd_2026: BankrollSummary;
  local_matched_games_avg_odds: number;
  settled_bets_rows: SettledBet[];
  settled_bets_summary: {
    count: number;
    wins: number;
    profit_eur: number;
    roi_pct: number;
    avg_odds: number;
  };
}
