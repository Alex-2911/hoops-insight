import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Dashboard runtime error", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background text-foreground">
          <div className="container mx-auto px-4 py-16">
            <div className="glass-card p-8 text-center">
              <h1 className="text-2xl font-bold mb-2">Something went wrong</h1>
              <p className="text-sm text-muted-foreground">
                The dashboard hit an unexpected error. Please refresh or try again later.
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
