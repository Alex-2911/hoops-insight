export const RISK_METRICS_MIN_BETS = 5;

export const shouldShowRiskMetrics = (betCount, minBets = RISK_METRICS_MIN_BETS) => {
  if (!Number.isFinite(betCount)) {
    return false;
  }
  return betCount >= minBets;
};
