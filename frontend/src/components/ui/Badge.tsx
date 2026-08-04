import { HTMLAttributes } from "react";
import { cn } from "@utils/cn";
import type { SeverityLevel } from "@types/prediction";

const severityStyles: Record<SeverityLevel, string> = {
  low: "bg-severity-low/15 text-severity-low",
  moderate: "bg-severity-moderate/15 text-severity-moderate",
  high: "bg-severity-high/15 text-severity-high",
  critical: "bg-severity-critical/15 text-severity-critical",
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  severity?: SeverityLevel;
}

export function Badge({ severity, className, children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium capitalize",
        severity ? severityStyles[severity] : "bg-canopy-100 text-canopy-700 dark:bg-canopy-800 dark:text-canopy-200",
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
