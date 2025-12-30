import { ReactNode } from 'react';
import { Navigation } from './Navigation';

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main className="pt-16">
        {children}
      </main>
      <footer className="border-t border-border py-8 mt-16">
        <div className="container mx-auto px-4 text-center text-muted-foreground text-sm">
          <p>Basketball Prediction Pipeline 2026 • Data-driven NBA betting insights</p>
          <p className="mt-2 text-xs">For educational purposes only. Please gamble responsibly.</p>
        </div>
      </footer>
    </div>
  );
};
