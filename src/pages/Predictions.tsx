import { Layout } from '@/components/layout/Layout';
import { PredictionCard } from '@/components/cards/PredictionCard';
import { todaysPredictions } from '@/data/mockData';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Filter, Grid, List } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

type ViewMode = 'cards' | 'table';
type FilterMode = 'all' | 'qualified';

const Predictions = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('cards');
  const [filter, setFilter] = useState<FilterMode>('all');

  const filteredPredictions = filter === 'qualified'
    ? todaysPredictions.filter(p => p.isQualified)
    : todaysPredictions;

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Today's Predictions</h1>
          <p className="text-muted-foreground">
            {new Date().toLocaleDateString('en-US', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between mb-8">
          <div className="flex gap-2">
            <Button
              variant={filter === 'all' ? 'default' : 'outline'}
              onClick={() => setFilter('all')}
              size="sm"
            >
              All Games ({todaysPredictions.length})
            </Button>
            <Button
              variant={filter === 'qualified' ? 'default' : 'outline'}
              onClick={() => setFilter('qualified')}
              size="sm"
            >
              <Filter className="w-4 h-4 mr-1" />
              Qualified ({todaysPredictions.filter(p => p.isQualified).length})
            </Button>
          </div>

          <div className="flex gap-2">
            <Button
              variant={viewMode === 'cards' ? 'secondary' : 'ghost'}
              onClick={() => setViewMode('cards')}
              size="icon"
            >
              <Grid className="w-4 h-4" />
            </Button>
            <Button
              variant={viewMode === 'table' ? 'secondary' : 'ghost'}
              onClick={() => setViewMode('table')}
              size="icon"
            >
              <List className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Content */}
        {viewMode === 'cards' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredPredictions.map((prediction) => (
              <PredictionCard key={prediction.id} prediction={prediction} />
            ))}
          </div>
        ) : (
          <div className="glass-card overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="border-border hover:bg-transparent">
                  <TableHead>Matchup</TableHead>
                  <TableHead className="text-right">Win %</TableHead>
                  <TableHead className="text-right">Odds</TableHead>
                  <TableHead className="text-right">EV/€100</TableHead>
                  <TableHead className="text-right">Kelly %</TableHead>
                  <TableHead className="text-center">Pick</TableHead>
                  <TableHead className="text-center">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredPredictions.map((prediction) => (
                  <TableRow key={prediction.id} className="table-row-hover border-border">
                    <TableCell>
                      <div className="space-y-1">
                        <p className={cn(
                          "font-medium",
                          prediction.predictedWinner === prediction.homeTeam && "text-primary"
                        )}>
                          {prediction.homeTeam}
                        </p>
                        <p className={cn(
                          "text-sm",
                          prediction.predictedWinner === prediction.awayTeam && "text-primary"
                        )}>
                          @ {prediction.awayTeam}
                        </p>
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="space-y-1">
                        <p>{(prediction.homeWinProb * 100).toFixed(0)}%</p>
                        <p className="text-sm text-muted-foreground">
                          {(prediction.awayWinProb * 100).toFixed(0)}%
                        </p>
                      </div>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      <div className="space-y-1">
                        <p>{prediction.homeOdds.toFixed(2)}</p>
                        <p className="text-sm text-muted-foreground">
                          {prediction.awayOdds.toFixed(2)}
                        </p>
                      </div>
                    </TableCell>
                    <TableCell className={cn(
                      "text-right font-semibold",
                      prediction.evPer100 > 0 ? "stat-positive" : "stat-negative"
                    )}>
                      {prediction.evPer100 > 0 ? '+' : ''}{prediction.evPer100.toFixed(1)}€
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {prediction.kellyStake.toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-center">
                      <span className="text-primary font-semibold">
                        {prediction.predictedWinner.split(' ').pop()}
                      </span>
                    </TableCell>
                    <TableCell className="text-center">
                      {prediction.isQualified ? (
                        <span className="inline-flex items-center gap-1 bg-primary/10 text-primary text-xs font-medium px-2 py-1 rounded-full">
                          ✓ Qualified
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        {/* Legend */}
        <div className="mt-8 p-4 glass-card">
          <h3 className="font-semibold mb-3">Understanding the Data</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="font-medium text-primary">Win %</span>
              <p className="text-muted-foreground">Model's predicted probability of winning</p>
            </div>
            <div>
              <span className="font-medium text-primary">EV/€100</span>
              <p className="text-muted-foreground">Expected value per €100 wagered</p>
            </div>
            <div>
              <span className="font-medium text-primary">Kelly %</span>
              <p className="text-muted-foreground">Optimal stake based on Kelly criterion</p>
            </div>
            <div>
              <span className="font-medium text-primary">Qualified</span>
              <p className="text-muted-foreground">Meets EV threshold & confidence criteria</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Predictions;
