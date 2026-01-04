export interface SummaryStats {
  total_games: number;
  overall_accuracy: number;
  as_of_date: string;
}

export interface SummaryPayload {
  last_run: string;
  as_of_date: string;
  summary_stats: SummaryStats;
  kpis: {
    total_bets: number;
    win_rate: number;
    avg_profit_per_bet_eur: number;
    max_drawdown_eur: number;
    max_drawdown_pct: number;
  };
  source: {
    combined_file: string;
    bet_log_file: string;
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

export interface LocalMatchedGamesWindow {
  windowSize: number;
  n_trades: number;
  games: LocalMatchedGame[];
}

export interface TablesPayload {
  historical_stats: HistoricalStat[];
  accuracy_threshold_stats: AccuracyThresholdStat[];
  calibration_metrics: CalibrationMetrics;
  home_win_rates_last20: HomeWinRate[];
  bet_log_summary: BetLogSummary;
  bankroll_history: BankrollEntry[];
  local_matched_games: LocalMatchedGamesWindow;
}
