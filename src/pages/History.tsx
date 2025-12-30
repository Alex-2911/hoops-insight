import { Layout } from '@/components/layout/Layout';
import { StatCard } from '@/components/cards/StatCard';
import { historicalStats, summaryStats } from '@/data/mockData';
import { Target, TrendingUp, BarChart3, Calendar } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from 'recharts';

const History = () => {
  const chartData = historicalStats.map(stat => ({
    ...stat,
    accuracyPercent: (stat.accuracy * 100).toFixed(0),
    dateShort: stat.date.slice(5),
  }));

  const avgAccuracy = historicalStats.reduce((sum, s) => sum + s.accuracy, 0) / historicalStats.length;
  const totalCorrect = historicalStats.reduce((sum, s) => sum + s.correctPredictions, 0);
  const totalPredictions = historicalStats.reduce((sum, s) => sum + s.totalPredictions, 0);

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Historical Accuracy</h1>
          <p className="text-muted-foreground">
            Track record and performance metrics over time
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Average Accuracy"
            value={`${(avgAccuracy * 100).toFixed(1)}%`}
            subtitle="last 14 days"
            icon={<Target className="w-6 h-6" />}
          />
          <StatCard
            title="Correct Predictions"
            value={`${totalCorrect}/${totalPredictions}`}
            subtitle={`${((totalCorrect / totalPredictions) * 100).toFixed(0)}% hit rate`}
            icon={<TrendingUp className="w-6 h-6" />}
            trend="up"
          />
          <StatCard
            title="Best Day"
            value="80%"
            subtitle="Jan 6, 2026"
            icon={<BarChart3 className="w-6 h-6" />}
          />
          <StatCard
            title="Days Tracked"
            value={historicalStats.length}
            subtitle="consistent data"
            icon={<Calendar className="w-6 h-6" />}
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Accuracy Over Time */}
          <div className="glass-card p-6">
            <h3 className="font-semibold mb-6">Accuracy Trend</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(38, 92%, 50%)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(38, 92%, 50%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(25, 20%, 20%)" />
                  <XAxis 
                    dataKey="dateShort" 
                    stroke="hsl(30, 10%, 55%)"
                    fontSize={12}
                  />
                  <YAxis 
                    domain={[40, 100]}
                    stroke="hsl(30, 10%, 55%)"
                    fontSize={12}
                    tickFormatter={(value) => `${value}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(20, 8%, 12%)',
                      border: '1px solid hsl(25, 20%, 20%)',
                      borderRadius: '8px',
                    }}
                    labelStyle={{ color: 'hsl(40, 20%, 92%)' }}
                    formatter={(value: number) => [`${value}%`, 'Accuracy']}
                  />
                  <Area
                    type="monotone"
                    dataKey="accuracyPercent"
                    stroke="hsl(38, 92%, 50%)"
                    strokeWidth={2}
                    fill="url(#accuracyGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Daily Predictions */}
          <div className="glass-card p-6">
            <h3 className="font-semibold mb-6">Daily Predictions</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(25, 20%, 20%)" />
                  <XAxis 
                    dataKey="dateShort" 
                    stroke="hsl(30, 10%, 55%)"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="hsl(30, 10%, 55%)"
                    fontSize={12}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(20, 8%, 12%)',
                      border: '1px solid hsl(25, 20%, 20%)',
                      borderRadius: '8px',
                    }}
                    labelStyle={{ color: 'hsl(40, 20%, 92%)' }}
                  />
                  <Bar 
                    dataKey="correctPredictions" 
                    name="Correct"
                    fill="hsl(142, 76%, 36%)" 
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar 
                    dataKey="totalPredictions" 
                    name="Total"
                    fill="hsl(25, 30%, 30%)" 
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Profit Chart */}
        <div className="glass-card p-6">
          <h3 className="font-semibold mb-6">Daily Profit/Loss</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(25, 20%, 20%)" />
                <XAxis 
                  dataKey="dateShort" 
                  stroke="hsl(30, 10%, 55%)"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(30, 10%, 55%)"
                  fontSize={12}
                  tickFormatter={(value) => `€${value}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(20, 8%, 12%)',
                    border: '1px solid hsl(25, 20%, 20%)',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ color: 'hsl(40, 20%, 92%)' }}
                  formatter={(value: number) => [`€${value.toFixed(2)}`, 'Profit']}
                />
                <Bar 
                  dataKey="profit" 
                  name="Profit"
                  fill="hsl(38, 92%, 50%)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Performance Notes */}
        <div className="mt-8 glass-card p-6">
          <h3 className="font-semibold mb-4">Performance Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div className="space-y-2">
              <p className="text-primary font-medium">Strong Performance</p>
              <p className="text-muted-foreground">
                Model maintains 66%+ accuracy, outperforming random chance (50%) significantly.
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-primary font-medium">Value Betting</p>
              <p className="text-muted-foreground">
                Positive EV bets are identified when model probability exceeds implied odds probability.
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-primary font-medium">Kelly Criterion</p>
              <p className="text-muted-foreground">
                Stake sizes are optimized using Kelly formula to maximize long-term growth.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default History;
