import { Layout } from '@/components/layout/Layout';
import { PredictionCard } from '@/components/cards/PredictionCard';
import { todaysPredictions } from '@/data/mockData';
import { Star, AlertCircle, TrendingUp } from 'lucide-react';

const Shortlist = () => {
  const qualifiedBets = todaysPredictions.filter(p => p.isQualified);
  const totalKelly = qualifiedBets.reduce((sum, p) => sum + p.kellyStake, 0);
  const avgEV = qualifiedBets.reduce((sum, p) => sum + p.evPer100, 0) / qualifiedBets.length;

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Star className="w-8 h-8 text-primary fill-primary" />
            <h1 className="text-3xl font-bold">Best Bets Shortlist</h1>
          </div>
          <p className="text-muted-foreground">
            Games that pass all qualification filters from Script 5
          </p>
        </div>

        {/* Qualification Criteria */}
        <div className="glass-card p-6 mb-8">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-primary" />
            Qualification Criteria
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-muted/30 rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">Min. EV per €100</p>
              <p className="text-xl font-bold text-primary">≥ €2.00</p>
            </div>
            <div className="p-4 bg-muted/30 rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">Min. Kelly Stake</p>
              <p className="text-xl font-bold text-primary">≥ 1.5%</p>
            </div>
            <div className="p-4 bg-muted/30 rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">Confidence Level</p>
              <p className="text-xl font-bold text-primary">High</p>
            </div>
            <div className="p-4 bg-muted/30 rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">Odds Range</p>
              <p className="text-xl font-bold text-primary">1.5 - 3.0</p>
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        {qualifiedBets.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
            <div className="glass-card p-5 text-center">
              <p className="text-sm text-muted-foreground mb-1">Qualified Bets</p>
              <p className="text-3xl font-bold text-primary">{qualifiedBets.length}</p>
            </div>
            <div className="glass-card p-5 text-center">
              <p className="text-sm text-muted-foreground mb-1">Total Kelly Allocation</p>
              <p className="text-3xl font-bold">{totalKelly.toFixed(1)}%</p>
            </div>
            <div className="glass-card p-5 text-center">
              <p className="text-sm text-muted-foreground mb-1">Average EV</p>
              <p className="text-3xl font-bold stat-positive">+€{avgEV.toFixed(2)}</p>
            </div>
          </div>
        )}

        {/* Qualified Bets */}
        {qualifiedBets.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {qualifiedBets.map((prediction) => (
              <PredictionCard key={prediction.id} prediction={prediction} />
            ))}
          </div>
        ) : (
          <div className="glass-card p-12 text-center">
            <TrendingUp className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Qualified Bets Today</h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              No games currently meet our strict value criteria. Check back later as odds update, 
              or view all predictions to see the full slate.
            </p>
          </div>
        )}

        {/* Betting Tips */}
        <div className="mt-12 glass-card p-6">
          <h3 className="font-semibold mb-4">Smart Betting Tips</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold shrink-0">1</span>
                <p className="text-muted-foreground">
                  <span className="font-medium text-foreground">Use Kelly fractions:</span> Consider using 25-50% of the suggested Kelly stake to reduce volatility.
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold shrink-0">2</span>
                <p className="text-muted-foreground">
                  <span className="font-medium text-foreground">Line shopping:</span> Always compare odds across multiple bookmakers for the best value.
                </p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold shrink-0">3</span>
                <p className="text-muted-foreground">
                  <span className="font-medium text-foreground">Track everything:</span> Record all bets to analyze your actual results vs. predictions.
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold shrink-0">4</span>
                <p className="text-muted-foreground">
                  <span className="font-medium text-foreground">Bankroll management:</span> Never bet more than you can afford to lose.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Shortlist;
