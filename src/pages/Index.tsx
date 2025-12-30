import { Layout } from '@/components/layout/Layout';
import { Button } from '@/components/ui/button';
import { StatCard } from '@/components/cards/StatCard';
import { Link } from 'react-router-dom';
import { summaryStats, todaysPredictions } from '@/data/mockData';
import { TrendingUp, Target, DollarSign, Zap, ArrowRight, Star } from 'lucide-react';
import { PredictionCard } from '@/components/cards/PredictionCard';

const Index = () => {
  const qualifiedPicks = todaysPredictions.filter(p => p.isQualified);

  return (
    <Layout>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
        <div className="container mx-auto px-4 py-20 relative">
          <div className="max-w-3xl mx-auto text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
              <Zap className="w-4 h-4" />
              Basketball Prediction Pipeline 2026
            </div>
            
            <h1 className="text-4xl md:text-6xl font-bold mb-6 leading-tight">
              Data-Driven{' '}
              <span className="gradient-text">NBA Predictions</span>
              <br />for Smarter Betting
            </h1>
            
            <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
              Leverage machine learning models to identify value bets with positive expected value. 
              Get daily predictions, Kelly criterion stakes, and track your performance over time.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="gold" size="xl" asChild>
                <Link to="/predictions">
                  View Today's Predictions
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </Button>
              <Button variant="outline" size="xl" asChild>
                <Link to="/history">
                  See Track Record
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Overview */}
      <section className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Overall Accuracy"
            value={`${(summaryStats.overallAccuracy * 100).toFixed(0)}%`}
            subtitle={`${summaryStats.totalPredictions} predictions`}
            icon={<Target className="w-6 h-6" />}
          />
          <StatCard
            title="Total Profit"
            value={`+€${summaryStats.totalProfit.toFixed(0)}`}
            subtitle={`${summaryStats.roi}% ROI`}
            icon={<DollarSign className="w-6 h-6" />}
            trend="up"
          />
          <StatCard
            title="Win Streak"
            value={summaryStats.winStreak}
            subtitle="consecutive wins"
            icon={<TrendingUp className="w-6 h-6" />}
            trend="up"
          />
          <StatCard
            title="Today's Best Bets"
            value={summaryStats.qualifiedBetsToday}
            subtitle={`Avg EV: ${summaryStats.averageEV}%`}
            icon={<Star className="w-6 h-6" />}
          />
        </div>
      </section>

      {/* Featured Predictions */}
      <section className="container mx-auto px-4 py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold mb-2">Today's Top Picks</h2>
            <p className="text-muted-foreground">
              Qualified bets meeting our value criteria
            </p>
          </div>
          <Button variant="outline" asChild>
            <Link to="/predictions">
              View All
              <ArrowRight className="w-4 h-4" />
            </Link>
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {qualifiedPicks.map((prediction) => (
            <PredictionCard key={prediction.id} prediction={prediction} />
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="container mx-auto px-4 py-16">
        <h2 className="text-2xl font-bold text-center mb-12">How It Works</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            {
              step: '01',
              title: 'Data Collection',
              description: 'We gather comprehensive NBA statistics, team performance metrics, and historical betting odds daily.'
            },
            {
              step: '02',
              title: 'ML Prediction',
              description: 'Our models analyze patterns to predict game outcomes and calculate win probabilities.'
            },
            {
              step: '03',
              title: 'Value Identification',
              description: 'We compare predictions against market odds to find positive EV opportunities using Kelly criterion.'
            }
          ].map((item) => (
            <div key={item.step} className="glass-card p-6 text-center">
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                <span className="text-primary font-bold">{item.step}</span>
              </div>
              <h3 className="text-lg font-semibold mb-2">{item.title}</h3>
              <p className="text-muted-foreground text-sm">{item.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="glass-card p-8 md:p-12 text-center glow-border">
          <h2 className="text-3xl font-bold mb-4">Ready to Start?</h2>
          <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
            Explore today's predictions and discover value betting opportunities backed by data science.
          </p>
          <Button variant="gold" size="lg" asChild>
            <Link to="/predictions">
              Get Started
              <ArrowRight className="w-5 h-5" />
            </Link>
          </Button>
        </div>
      </section>
    </Layout>
  );
};

export default Index;
