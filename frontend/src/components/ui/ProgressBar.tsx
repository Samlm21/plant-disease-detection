import { cn } from "@utils/cn";

interface ProgressBarProps {
  value: number; // 0–1
  colorClassName?: string;
  trackClassName?: string;
  label?: string;
}

export function ProgressBar({ value, colorClassName = "bg-canopy-500", trackClassName, label }: ProgressBarProps) {
  const percent = Math.round(Math.min(Math.max(value, 0), 1) * 100);

  return (
    <div className="w-full">
      <div
        className={cn("h-2 w-full overflow-hidden rounded-full bg-canopy-100 dark:bg-canopy-800", trackClassName)}
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div
          className={cn("h-full rounded-full transition-all duration-500 ease-out", colorClassName)}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
