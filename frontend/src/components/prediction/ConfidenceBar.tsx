import { ProgressBar } from "@components/ui/ProgressBar";

interface ConfidenceBarProps {
  label: string;
  confidence: number; // 0–1
}

/** Confidence maps onto the same cool→hot scale used for the Grad-CAM legend,
 *  so a person learns one color language across the whole prediction card. */
function confidenceColor(confidence: number): string {
  if (confidence >= 0.75) return "bg-scan-hot";
  if (confidence >= 0.4) return "bg-scan-mid";
  return "bg-scan-cool";
}

export function ConfidenceBar({ label, confidence }: ConfidenceBarProps) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-32 shrink-0 truncate text-sm text-canopy-700 dark:text-canopy-200">{label}</span>
      <ProgressBar value={confidence} colorClassName={confidenceColor(confidence)} label={`${label} confidence`} />
      <span className="w-12 shrink-0 text-right font-mono text-sm text-canopy-600 dark:text-canopy-300">
        {Math.round(confidence * 100)}%
      </span>
    </div>
  );
}
