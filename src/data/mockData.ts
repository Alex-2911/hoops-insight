// Mock data simulating the Basketball_prediction 2026 pipeline output

export interface Prediction {
  id: string;
  date: string;
  homeTeam: string;
  awayTeam: string;
  homeWinProb: number;
  awayWinProb: number;
  predictedWinner: string;
  homeOdds: number;
  awayOdds: number;
  evPer100: number;
  kellyStake: number;
  isQualified: boolean;
  confidence: 'high' | 'medium' | 'low';
}

export interface HistoricalStat {
  date: string;
  accuracy: number;
  totalPredictions: number;
  correctPredictions: number;
  profit: number;
}

export interface BankrollEntry {
  date: string;
  balance: number;
  betsPlaced: number;
  profit: number;
}

// Today's predictions
export const todaysPredictions: Prediction[] = [
  {
    id: '1',
    date: '2026-01-15',
    homeTeam: 'Los Angeles Lakers',
    awayTeam: 'Boston Celtics',
    homeWinProb: 0.58,
    awayWinProb: 0.42,
    predictedWinner: 'Los Angeles Lakers',
    homeOdds: 1.85,
    awayOdds: 2.05,
    evPer100: 7.3,
    kellyStake: 3.2,
    isQualified: true,
    confidence: 'high'
  },
  {
    id: '2',
    date: '2026-01-15',
    homeTeam: 'Golden State Warriors',
    awayTeam: 'Phoenix Suns',
    homeWinProb: 0.52,
    awayWinProb: 0.48,
    predictedWinner: 'Golden State Warriors',
    homeOdds: 1.95,
    awayOdds: 1.90,
    evPer100: 1.4,
    kellyStake: 0.8,
    isQualified: false,
    confidence: 'low'
  },
  {
    id: '3',
    date: '2026-01-15',
    homeTeam: 'Milwaukee Bucks',
    awayTeam: 'Miami Heat',
    homeWinProb: 0.65,
    awayWinProb: 0.35,
    predictedWinner: 'Milwaukee Bucks',
    homeOdds: 1.55,
    awayOdds: 2.50,
    evPer100: 0.75,
    kellyStake: 0.5,
    isQualified: false,
    confidence: 'medium'
  },
  {
    id: '4',
    date: '2026-01-15',
    homeTeam: 'Denver Nuggets',
    awayTeam: 'Dallas Mavericks',
    homeWinProb: 0.61,
    awayWinProb: 0.39,
    predictedWinner: 'Denver Nuggets',
    homeOdds: 1.70,
    awayOdds: 2.25,
    evPer100: 3.7,
    kellyStake: 2.1,
    isQualified: true,
    confidence: 'high'
  },
  {
    id: '5',
    date: '2026-01-15',
    homeTeam: 'Philadelphia 76ers',
    awayTeam: 'New York Knicks',
    homeWinProb: 0.55,
    awayWinProb: 0.45,
    predictedWinner: 'Philadelphia 76ers',
    homeOdds: 1.80,
    awayOdds: 2.10,
    evPer100: -1.0,
    kellyStake: 0,
    isQualified: false,
    confidence: 'medium'
  },
  {
    id: '6',
    date: '2026-01-15',
    homeTeam: 'Cleveland Cavaliers',
    awayTeam: 'Chicago Bulls',
    homeWinProb: 0.72,
    awayWinProb: 0.28,
    predictedWinner: 'Cleveland Cavaliers',
    homeOdds: 1.40,
    awayOdds: 3.00,
    evPer100: 0.8,
    kellyStake: 0.6,
    isQualified: false,
    confidence: 'high'
  }
];

// Historical accuracy data
export const historicalStats: HistoricalStat[] = [
  { date: '2026-01-01', accuracy: 0.62, totalPredictions: 8, correctPredictions: 5, profit: 45.20 },
  { date: '2026-01-02', accuracy: 0.75, totalPredictions: 6, correctPredictions: 4, profit: 82.50 },
  { date: '2026-01-03', accuracy: 0.55, totalPredictions: 9, correctPredictions: 5, profit: -12.30 },
  { date: '2026-01-04', accuracy: 0.71, totalPredictions: 7, correctPredictions: 5, profit: 67.80 },
  { date: '2026-01-05', accuracy: 0.60, totalPredictions: 5, correctPredictions: 3, profit: 15.40 },
  { date: '2026-01-06', accuracy: 0.80, totalPredictions: 10, correctPredictions: 8, profit: 125.60 },
  { date: '2026-01-07', accuracy: 0.57, totalPredictions: 7, correctPredictions: 4, profit: -8.90 },
  { date: '2026-01-08', accuracy: 0.68, totalPredictions: 8, correctPredictions: 5, profit: 52.30 },
  { date: '2026-01-09', accuracy: 0.72, totalPredictions: 6, correctPredictions: 4, profit: 78.10 },
  { date: '2026-01-10', accuracy: 0.64, totalPredictions: 11, correctPredictions: 7, profit: 35.70 },
  { date: '2026-01-11', accuracy: 0.58, totalPredictions: 6, correctPredictions: 3, profit: -22.50 },
  { date: '2026-01-12', accuracy: 0.77, totalPredictions: 9, correctPredictions: 7, profit: 98.40 },
  { date: '2026-01-13', accuracy: 0.66, totalPredictions: 8, correctPredictions: 5, profit: 41.20 },
  { date: '2026-01-14', accuracy: 0.70, totalPredictions: 7, correctPredictions: 5, profit: 55.80 },
];

// Bankroll history
export const bankrollHistory: BankrollEntry[] = [
  { date: '2026-01-01', balance: 1000, betsPlaced: 3, profit: 45.20 },
  { date: '2026-01-02', balance: 1082.50, betsPlaced: 2, profit: 82.50 },
  { date: '2026-01-03', balance: 1070.20, betsPlaced: 4, profit: -12.30 },
  { date: '2026-01-04', balance: 1138.00, betsPlaced: 3, profit: 67.80 },
  { date: '2026-01-05', balance: 1153.40, betsPlaced: 2, profit: 15.40 },
  { date: '2026-01-06', balance: 1279.00, betsPlaced: 5, profit: 125.60 },
  { date: '2026-01-07', balance: 1270.10, betsPlaced: 2, profit: -8.90 },
  { date: '2026-01-08', balance: 1322.40, betsPlaced: 3, profit: 52.30 },
  { date: '2026-01-09', balance: 1400.50, betsPlaced: 2, profit: 78.10 },
  { date: '2026-01-10', balance: 1436.20, betsPlaced: 4, profit: 35.70 },
  { date: '2026-01-11', balance: 1413.70, betsPlaced: 2, profit: -22.50 },
  { date: '2026-01-12', balance: 1512.10, betsPlaced: 4, profit: 98.40 },
  { date: '2026-01-13', balance: 1553.30, betsPlaced: 3, profit: 41.20 },
  { date: '2026-01-14', balance: 1609.10, betsPlaced: 3, profit: 55.80 },
];

// Summary statistics
export const summaryStats = {
  totalPredictions: 107,
  overallAccuracy: 0.66,
  totalProfit: 609.10,
  roi: 8.7,
  winStreak: 4,
  qualifiedBetsToday: 2,
  averageEV: 3.2,
};
