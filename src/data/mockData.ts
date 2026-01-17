// Mock data simulating historical outputs of the Basketball_prediction pipeline
// IMPORTANT: This dashboard is statistics-only and does NOT expose future predictions.

export interface HistoricalStat {
  date: string; // YYYY-MM-DD
  accuracy: number; // 0..1
  totalGames: number;
  correctGames: number;
}

export interface AccuracyThresholdStat {
  label: string;          // e.g. "> 0.60"
  thresholdType: "gt" | "lt";
  threshold: number;      // e.g. 0.60
  accuracy: number;       // 0..1
  sampleSize: number;     // number of historical games in this bucket
}

export interface HomeWinRate {
  team: string;           // e.g. "BOS"
  totalGames: number;
  totalHomeGames: number;
  homeWins: number;
  homeWinRate: number;    // 0..1
}

export interface StrategyFilterStats {
  totalGames: number;
  homeWinRateMin: number;
  oddsMin: number;
  oddsMax: number;
  probMin: number;
  minEv: number;
  passedHomeWinRate: number;
  passedOddsRange: number;
  passedProbThreshold: number;
  passedEvThreshold: number;
  matchedGamesCount: number;
  matchedGamesAccuracy: number;
}

export interface CalibrationMetrics {
  asOfDate: string;       // YYYY-MM-DD (last update)
  brierBefore: number;
  brierAfter: number;
  logLossBefore: number;
  logLossAfter: number;
  fittedGames: number;
}

export interface StrategySummary {
  asOfDate: string;       // YYYY-MM-DD (last update)
  totalBets: number;
  totalStakedEur: number | null;
  totalProfitEur: number | null;
  roiPct: number | null;         // ROI% on historical (settled) bets
  avgStakeEur: number | null;
  avgProfitPerBetEur: number | null;
  winRate: number;        // 0..1 (historical)
  avgEvPer100: number | null;
  profitMetricsAvailable: boolean;
}

export interface BankrollEntry {
  date: string;   // YYYY-MM-DD
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

// ---------------------------------------------------------------------------
// Historical accuracy data (played games only)
// ---------------------------------------------------------------------------

export const historicalStats: HistoricalStat[] = [
  { date: "2026-01-01", accuracy: 0.62, totalGames: 8, correctGames: 5 },
  { date: "2026-01-02", accuracy: 0.75, totalGames: 6, correctGames: 4 },
  { date: "2026-01-03", accuracy: 0.55, totalGames: 9, correctGames: 5 },
  { date: "2026-01-04", accuracy: 0.71, totalGames: 7, correctGames: 5 },
  { date: "2026-01-05", accuracy: 0.60, totalGames: 5, correctGames: 3 },
  { date: "2026-01-06", accuracy: 0.80, totalGames: 10, correctGames: 8 },
  { date: "2026-01-07", accuracy: 0.57, totalGames: 7, correctGames: 4 },
  { date: "2026-01-08", accuracy: 0.68, totalGames: 8, correctGames: 5 },
  { date: "2026-01-09", accuracy: 0.72, totalGames: 6, correctGames: 4 },
  { date: "2026-01-10", accuracy: 0.64, totalGames: 11, correctGames: 7 },
  { date: "2026-01-11", accuracy: 0.58, totalGames: 6, correctGames: 3 },
  { date: "2026-01-12", accuracy: 0.77, totalGames: 9, correctGames: 7 },
  { date: "2026-01-13", accuracy: 0.66, totalGames: 8, correctGames: 5 },
  { date: "2026-01-14", accuracy: 0.70, totalGames: 7, correctGames: 5 },
];

// Accuracy by probability thresholds (historical / played games)
export const accuracyThresholdStats: AccuracyThresholdStat[] = [
  { label: "> 0.60", thresholdType: "gt", threshold: 0.60, accuracy: 0.7256, sampleSize: 113 },
  { label: "<= 0.40", thresholdType: "lt", threshold: 0.40, accuracy: 0.7179, sampleSize: 78 },
];

// ---------------------------------------------------------------------------
// Home win rates (last N games window; only home games count for the rate)
// ---------------------------------------------------------------------------

export const homeWinRatesWindow: HomeWinRate[] = [
  { team: "DET", totalGames: 20, totalHomeGames: 8, homeWins: 7, homeWinRate: 0.88 },
  { team: "HOU", totalGames: 20, totalHomeGames: 8, homeWins: 7, homeWinRate: 0.88 },
  { team: "NYK", totalGames: 20, totalHomeGames: 8, homeWins: 7, homeWinRate: 0.88 },
  { team: "OKC", totalGames: 20, totalHomeGames: 12, homeWins: 10, homeWinRate: 0.83 },
  { team: "BOS", totalGames: 20, totalHomeGames: 10, homeWins: 8, homeWinRate: 0.80 },
  // ...add more as needed
];

// ---------------------------------------------------------------------------
// Calibration metrics (Isotonic Regression) - historical fit only
// ---------------------------------------------------------------------------

export const calibrationMetrics: CalibrationMetrics = {
  asOfDate: "2025-12-30",
  brierBefore: 0.225921,
  brierAfter: 0.218131,
  logLossBefore: 0.644936,
  logLossAfter: 0.625809,
  fittedGames: 467,
};

// ---------------------------------------------------------------------------
// Historical (settled) bet log summary only
// NOTE: No future recommendations are shown.
// ---------------------------------------------------------------------------

export const strategySummary: StrategySummary = {
  asOfDate: "2025-12-30",
  totalBets: 23,
  totalStakedEur: 2300,
  totalProfitEur: 1008,
  roiPct: 43.83,
  avgStakeEur: 100,
  avgProfitPerBetEur: 43.83,
  winRate: 0.0, // set real value when available
  avgEvPer100: 24.01,
  profitMetricsAvailable: true,
};

// Bankroll history (historical / settled)
export const bankrollHistory: BankrollEntry[] = [
  { date: "2026-01-01", balance: 1000, betsPlaced: 3, profit: 45.20 },
  { date: "2026-01-02", balance: 1082.50, betsPlaced: 2, profit: 82.50 },
  { date: "2026-01-03", balance: 1070.20, betsPlaced: 4, profit: -12.30 },
  { date: "2026-01-04", balance: 1138.00, betsPlaced: 3, profit: 67.80 },
  { date: "2026-01-05", balance: 1153.40, betsPlaced: 2, profit: 15.40 },
  { date: "2026-01-06", balance: 1279.00, betsPlaced: 5, profit: 125.60 },
  { date: "2026-01-07", balance: 1270.10, betsPlaced: 2, profit: -8.90 },
  { date: "2026-01-08", balance: 1322.40, betsPlaced: 3, profit: 52.30 },
  { date: "2026-01-09", balance: 1400.50, betsPlaced: 2, profit: 78.10 },
  { date: "2026-01-10", balance: 1436.20, betsPlaced: 4, profit: 35.70 },
  { date: "2026-01-11", balance: 1413.70, betsPlaced: 2, profit: -22.50 },
  { date: "2026-01-12", balance: 1512.10, betsPlaced: 4, profit: 98.40 },
  { date: "2026-01-13", balance: 1553.30, betsPlaced: 3, profit: 41.20 },
  { date: "2026-01-14", balance: 1609.10, betsPlaced: 3, profit: 55.80 },
];

export const localMatchedGamesRows: LocalMatchedGameRow[] = [];

// Summary statistics (historical only)
export const summaryStats = {
  totalGames: 107,
  overallAccuracy: 0.6639,
  asOfDate: "2025-12-30",
};
