import { Prediction } from '@/data/mockData';
import { cn } from '@/lib/utils';
import { TrendingUp, Star } from 'lucide-react';

interface PredictionCardProps {
  prediction: Prediction;
}

export const PredictionCard = ({ prediction }: PredictionCardProps) => {
  const confidenceColors = {
    high: 'bg-green-500/20 text-green-400 border-green-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    low: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  return (
    <div className={cn(
      "glass-card p-5 transition-all duration-300 hover:border-primary/30",
      prediction.isQualified && "ring-1 ring-primary/30 glow-border"
    )}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {prediction.isQualified && (
            <Star className="w-4 h-4 text-primary fill-primary" />
          )}
          <span className={cn(
            "text-xs font-medium px-2 py-1 rounded-full border",
            confidenceColors[prediction.confidence]
          )}>
            {prediction.confidence.toUpperCase()}
          </span>
        </div>
        <span className="text-xs text-muted-foreground">{prediction.date}</span>
      </div>

      {/* Teams */}
      <div className="space-y-3 mb-4">
        <div className="flex items-center justify-between">
          <span className={cn(
            "font-semibold",
            prediction.predictedWinner === prediction.homeTeam && "text-primary"
          )}>
            {prediction.homeTeam}
          </span>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">
              {(prediction.homeWinProb * 100).toFixed(0)}%
            </span>
            <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
              {prediction.homeOdds.toFixed(2)}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between">
          <span className={cn(
            "font-semibold",
            prediction.predictedWinner === prediction.awayTeam && "text-primary"
          )}>
            {prediction.awayTeam}
          </span>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">
              {(prediction.awayWinProb * 100).toFixed(0)}%
            </span>
            <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
              {prediction.awayOdds.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-3 pt-4 border-t border-border">
        <div className="text-center">
          <p className="text-xs text-muted-foreground mb-1">EV/€100</p>
          <p className={cn(
            "font-semibold",
            prediction.evPer100 > 0 ? "stat-positive" : "stat-negative"
          )}>
            {prediction.evPer100 > 0 ? '+' : ''}{prediction.evPer100.toFixed(1)}€
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground mb-1">Kelly %</p>
          <p className="font-semibold text-foreground">
            {prediction.kellyStake.toFixed(1)}%
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground mb-1">Pick</p>
          <div className="flex items-center justify-center gap-1">
            <TrendingUp className="w-3 h-3 text-primary" />
            <p className="font-semibold text-primary text-sm">
              {prediction.predictedWinner.split(' ').pop()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
