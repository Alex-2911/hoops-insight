import { ReactNode } from 'react';
interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="min-h-screen bg-background">
      <main>{children}</main>
      <footer className="border-t border-border py-8 mt-16">
        <div className="container mx-auto px-4 text-center text-muted-foreground text-sm">
          <p>NBA Results & Statistics • Historical analysis only</p>
          <p className="mt-2 text-xs">No future predictions are provided. For educational purposes only.</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
