import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  componentStack?: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.setState({ componentStack: info.componentStack });
    console.error("Dashboard runtime error", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background px-4 py-10 text-foreground">
          <div className="mx-auto max-w-4xl rounded-xl border border-red-500/60 bg-red-950/30 p-6">
            <h1 className="mb-3 text-2xl font-bold text-red-300">Frontend render error</h1>
            <p className="mb-4 text-sm text-red-100">
              The application crashed during rendering, before or during dashboard data fetch.
            </p>
            <p className="mb-2 text-xs uppercase tracking-wide text-red-200/80">Error message</p>
            <pre className="mb-4 overflow-auto rounded bg-black/30 p-3 text-xs text-red-100">
              {this.state.error?.message ?? "Unknown error"}
            </pre>
            {this.state.componentStack && (
              <>
                <p className="mb-2 text-xs uppercase tracking-wide text-red-200/80">Component stack</p>
                <pre className="overflow-auto rounded bg-black/30 p-3 text-xs text-red-100">
                  {this.state.componentStack}
                </pre>
              </>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
