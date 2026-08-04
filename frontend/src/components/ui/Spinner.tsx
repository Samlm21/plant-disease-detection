import { cn } from "@utils/cn";

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn(
        "inline-block h-6 w-6 animate-spin rounded-full border-2 border-canopy-300 border-t-canopy-600",
        className
      )}
    />
  );
}
