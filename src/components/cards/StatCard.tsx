import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { fmtNumber } from "@/lib/format";

interface StatCardProps {
  title: ReactNode;
  value: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  trend?: "up" | "down" | "neutral";
  className?: string;
}

export const StatCard = ({ title, value, subtitle, icon, trend, className }: StatCardProps) => {
  const displayValue =
    typeof value === "number"
      ? fmtNumber(value, 2)
      : value === null || value === undefined
        ? "—"
        : value;

  return (
    <div className={cn(
      "glass-card p-6 transition-all duration-300 hover:shadow-lg hover:border-primary/30",
      className
    )}>
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
          <p className={cn(
            "text-3xl font-extrabold leading-none text-foreground",
            trend === "up" && "stat-positive",
            trend === "down" && "stat-negative"
          )}>
            {displayValue}
          </p>
          {subtitle && (
            <div className="text-sm leading-6 text-muted-foreground">{subtitle}</div>
          )}
        </div>
        {icon && (
          <div className="w-12 h-12 rounded-xl bg-primary/15 flex items-center justify-center text-primary">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};
