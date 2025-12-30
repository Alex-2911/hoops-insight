import { StatCard } from "@/components/cards/StatCard";
import {
  summaryStats,
  historicalStats,
  accuracyThresholdStats,
  calibrationMetrics,
  homeWinRatesLast20,
  betLogSummary,
  bankrollHistory,
} from "@/data/mockData";
import { Target, TrendingUp, Activity, BarChart3 } from "lucide-react";

const Index = () => {
  const lastHist = historicalStats[historicalStats.length - 1];
  const lastBankroll = bankrollHistory[bankrollHistory.length - 1];

  const overallAccuracyPct = (summaryStats.overallAccuracy * 100).toFixed(2);

  const topHomeTeams = [...homeWinRatesLast20]
    .sort((a, b) => b.homeWinRate - a.homeWinRate)
    .slice(0, 8);

  return (
    <>
      {/* Header / Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
        <div className="container mx-auto px-4 py-16 relative">
          <div className="max-w-3xl mx-auto text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
              <Activity className="w-4 h-4" />
              Hoops Insight • Statistics Dashboard
            </div>

            <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
              Historical NBA Model Performance
            </h1>

            <p className="text-lg text-muted-foreground mb-2">
              This dashboard shows historical accuracy, calibration quality, and
              statistical summaries only.
            </p>

            <p className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">Legal:</span> This
              website does not provide predictions for future sporting or betting
              outcomes. It serves purely for historical model accuracy and
              statistical analysis of NBA games.
            </p>
          </div>
        </div>
      </section>

      {/* Top stats */}
      <section className="container mx-auto px-4 py-10">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Overall Accuracy"
            value={`${overallAccuracyPct}%`}
            subtitle={`Played games: ${summaryStats.totalGames}`}
            icon={<Target className="w-6 h-6" />}
          />

          <StatCard
            title="Last Update"
            value={summaryStats.asOfDate}
            subtitle={lastHist ? `Last day acc: ${(lastHist.accuracy * 100).toFixed(0)}%` : "—"}
            icon={<TrendingUp className="w-6 h-6" />}
          />

          <StatCard
            title="Calibration (Brier)"
            value={`${calibrationMetrics.brierAfter.toFixed(3)}`}
            subtitle={`Before: ${calibrationMetrics.brierBefore.toFixed(3)}`}
            icon={<BarChart3 className="w-6 h-6" />}
          />

          <StatCard
            title="Bankroll (Historical)"
            value={lastBankroll ? `€${lastBankroll.balance.toFixed(2)}` : "—"}
            subtitle={lastBankroll ? `Last day P/L: €${lastBankroll.profit.toFixed(2)}` : "—"}
            icon={<Activity className="w-6 h-6" />}
          />
        </div>
      </section>

      {/* Accuracy thresholds */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Accuracy by Threshold</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Accuracy buckets computed on historical played games only.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {accuracyThresholdStats.map((t) => (
              <div key={t.label} className="rounded-lg border border-border p-4">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{t.label}</div>
                  <div className="text-lg font-bold">
                    {(t.accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="text-sm text-muted-foreground mt-1">
                  Sample size: {t.sampleSize}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Calibration */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Calibration Quality</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Metrics compare raw probabilities vs isotonic-calibrated probabilities.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Brier Score</div>
              <div className="text-sm text-muted-foreground">
                Before: <span className="font-medium text-foreground">{calibrationMetrics.brierBefore.toFixed(6)}</span>
              </div>
              <div className="text-sm text-muted-foreground">
                After: <span className="font-medium text-foreground">{calibrationMetrics.brierAfter.toFixed(6)}</span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Fitted games: {calibrationMetrics.fittedGames} • As of {calibrationMetrics.asOfDate}
              </div>
            </div>

            <div className="rounded-lg border border-border p-4">
              <div className="font-semibold mb-2">Log Loss</div>
              <div className="text-sm text-muted-foreground">
                Before: <span className="font-medium text-foreground">{calibrationMetrics.logLossBefore.toFixed(6)}</span>
              </div>
              <div className="text-sm text-muted-foreground">
                After: <span className="font-medium text-foreground">{calibrationMetrics.logLossAfter.toFixed(6)}</span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Lower is better • Historical only
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Home win rates */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Home Win Rate (Last 20 Games Window)</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Last-20 window per team; win rate computed only on home games inside that window.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="py-2 pr-4">Team</th>
                  <th className="py-2 pr-4">Home Win Rate</th>
                  <th className="py-2 pr-4">Home Wins</th>
                  <th className="py-2 pr-4">Home Games</th>
                  <th className="py-2 pr-4">Last 20 Games</th>
                </tr>
              </thead>
              <tbody>
                {topHomeTeams.map((t) => (
                  <tr key={t.team} className="border-b border-border/50">
                    <td className="py-2 pr-4 font-medium">{t.team}</td>
                    <td className="py-2 pr-4">{(t.homeWinRate * 100).toFixed(0)}%</td>
                    <td className="py-2 pr-4">{t.homeWins}</td>
                    <td className="py-2 pr-4">{t.totalHomeGames}</td>
                    <td className="py-2 pr-4">{t.totalLast20Games}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="text-xs text-muted-foreground mt-4">
            Showing top {topHomeTeams.length} teams by home win rate (mock data).
          </div>
        </div>
      </section>

      {/* Bet log summary (historical only) */}
      <section className="container mx-auto px-4 py-10">
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold mb-2">Historical Bet Log Summary (Settled Games)</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Aggregated statistics from played games only. No future recommendations are displayed.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Total Bets</div>
              <div className="text-2xl font-bold">{betLogSummary.totalBets}</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Total Profit</div>
              <div className="text-2xl font-bold">€{betLogSummary.totalProfitEur.toFixed(2)}</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">ROI</div>
              <div className="text-2xl font-bold">{betLogSummary.roiPct.toFixed(2)}%</div>
            </div>
            <div className="rounded-lg border border-border p-4">
              <div className="text-sm text-muted-foreground">Avg Stake</div>
              <div className="text-2xl font-bold">€{betLogSummary.avgStakeEur.toFixed(2)}</div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground mt-4">
            As of {betLogSummary.asOfDate} • Historical / settled only.
          </div>
        </div>
      </section>
    </>
  );
};

export default Index;
