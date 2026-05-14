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
  model?: {
    calibration?: Partial<CalibrationMetrics> & {
      asOfDate?: string;
      actualWinPct?: number;
      fittedGames?: number;
    };
    homeWinRatesLast20?: HomeWinRate[];
  };
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
  strategy?: {
    roiPct?: number;
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
  snapshot_as_of_date?: string;
  window_size: number;
  window_start?: string | null;
  window_end?: string | null;
  active_filters_text: string;
  params_used_label: string;
  params_used?: string;
  active_params?: {
    home_win_rate_min?: number;
    odds_min?: number;
    odds_max?: number;
    prob_threshold?: number;
    min_ev?: number;
    window_size?: number;
  };
  params_source_label: string;
  strategy_params_parse_status?: "ok" | "defaults" | "parse_error" | "missing";
  strategy_params_parse_error?: string | null;
  defaults_used?: boolean;
  defaults_reason?: string | null;
  strategy_as_of_date?: string | null;
  strategy_matches_window?: number;
  data_consistency_status?: "ok" | "out_of_sync";
  data_consistency_issues?: string[];
  combined_source_file?: string;
  local_matched_source_file?: string;
  strategy_params_source_file?: string;
  strategy_params_source_type?: string;
  strategy_params_parsed_ok?: boolean;
  metrics_snapshot_source_file?: string;
  bet_log_source_file?: string;
  bet_log_latest_date_in_file?: string | null;
  params_source_type?: string;
  fallback_used?: boolean;
  fallback_reason?: string | null;
  last_update_utc: string;
  sources: {
    combined: string;
    local_matched?: string;
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
  today_shortlist?: unknown[];
  qualifying_games_today?: number;
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

export interface TodayGame {
  date?: string | null;
  home_team?: string | null;
  away_team?: string | null;
  home_team_prob?: number | null;
  away_team_prob?: number | null;
  prob_used?: number | null;
  prob_base?: number | null;
  prob_live_oos_proxy?: number | null;
  prob_iso?: number | null;
  market_implied_p_devig?: number | null;
  model_market_gap?: number | null;
  ev_live_eur_per_100?: number | null;
  candidate_stake_eur?: number | null;
  home_win_rate?: number | null;
  home_wins?: number | null;
  home_games?: number | null;
  last20_games?: number | null;
  hwr_source_file?: string | null;
  hwr_source_label?: string | null;
  hwr_window_label?: string | null;
  home_odds?: number | null;
  away_odds?: number | null;
}

export interface TodayGamesPayload {
  as_of_date?: string | null;
  source?: string | null;
  games: TodayGame[];
  qualifying_bets?: Array<Record<string, string>>;
  local_matched_games?: Array<Record<string, string>>;
  engine_state?: string | null;
  canonical_model_signals?: {
    engine_state?: string | null;
    source_file?: string | null;
    summary_file?: string | null;
    canonical_count?: number;
    canonical?: Array<Record<string, string>>;
    current_rows?: Array<Record<string, string>>;
    label?: string;
  };
  ev_exception_profitability?: {
    label?: string;
    classification?: string;
    is_betting_signal?: boolean;
    recommendation_label?: string;
    warning?: string | null;
    note?: string;
    debug_csv?: string;
    criteria?: Record<string, string | number | null>;
    current_candidates?: Array<Record<string, string | number | null>>;
    summary?: {
      n?: number;
      wins?: number;
      losses?: number;
      win_rate?: number | null;
      avg_odds?: number | null;
      profit_100_flat?: number;
      roi_pct?: number | null;
      avg_prob_used?: number | null;
      avg_home_win_rate?: number | null;
      window_start?: string | null;
      window_end?: string | null;
    };
    per_candidate_checks?: Array<Record<string, string | number | boolean | null>>;
    price_adjusted?: {
      game?: string;
      date?: string | null;
      home_team?: string | null;
      away_team?: string | null;
      label?: string;
      odds_band?: Array<number | null>;
      prob_used_band?: Array<number | null>;
      hwr_source_file?: string | null;
      hwr_source_label?: string | null;
      hwr_window_label?: string | null;
      current_odds?: number | null;
      current_prob_used?: number | null;
      current_ev_eur_per_100?: number | null;
      current_kelly?: number | null;
      current_stake_eur?: number | null;
      break_even_probability?: number | null;
      current_prob_minus_break_even?: number | null;
      n?: number;
      wins?: number;
      losses?: number;
      win_rate?: number | null;
      avg_odds?: number | null;
      profit_100_flat?: number;
      roi_pct?: number | null;
      win_rate_minus_break_even?: number | null;
      supports_play?: boolean;
      classification?: string;
    };
    matches?: Array<Record<string, string | number | null>>;
  } | null;
  upcoming_game_checks?: {
    label?: string;
    basis?: {
      label?: string;
      source?: string;
      source_file?: string | null;
      comparable_band?: string;
      hwr_band_method?: string;
      ev_included?: boolean;
      stake_model?: string;
      history_meta?: Record<string, unknown>;
    };
    rows?: Array<Record<string, string | number | boolean | Array<number | null> | null>>;
  };
  setup_profitability?: {
    summary?: Record<string, unknown> | null;
    rows?: Array<Record<string, string>>;
    matches?: Array<Record<string, string>>;
  };
  historical_roi_attack_scans?: Array<Record<string, unknown>>;
  local_profitability_rule?: {
    rule_name?: string;
    canonical_override_allowed?: boolean;
    approval_source?: string;
    cases?: Array<Record<string, unknown>>;
  };
  local_strategy_evaluation_window?: {
    label?: string;
    local_tail_used?: number | null;
    hist_df_rows?: number | null;
    local_eval_rows?: number | null;
    valid_window_size?: number | null;
    display_window_games?: number | null;
    start?: string | null;
    end?: string | null;
    source?: string | null;
    source_file?: string | null;
    matches_script11_local_tail?: boolean;
    matches_active_params?: boolean;
    warning?: string | null;
  };
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
  calibration_quality?: {
    ece: number;
    calibrationSlope: number;
    calibrationIntercept: number;
  };
  home_win_rates_last20: HomeWinRate[];
  home_win_rates_window?: HomeWinRateWindowRow[];
  home_win_rate_threshold?: number;
  home_win_rate_shown_count?: number;
  strategy_filter_stats?: {
    window_size: number;
    matched_games_count: number;
    window_start?: string | null;
    window_end?: string | null;
    filters?: Array<{ label: string; value: number }>;
  };
  strategy_summary?: Record<string, unknown>;
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
  bets_2026_settled_rows?: SettledBet[];
  bets_2026_settled_count?: number;
  bets_2026_settled_summary?: {
    count: number;
    wins: number;
    profit_eur: number;
    roi_pct: number;
    avg_odds: number;
  };
}
