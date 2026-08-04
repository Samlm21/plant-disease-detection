import { ButtonHTMLAttributes, forwardRef } from "react";
import { cn } from "@utils/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  isLoading?: boolean;
}

const variantStyles: Record<Variant, string> = {
  primary: "bg-canopy-600 text-white hover:bg-canopy-700 disabled:bg-canopy-300",
  secondary:
    "bg-canopy-100 text-canopy-800 hover:bg-canopy-200 dark:bg-canopy-800 dark:text-canopy-100 dark:hover:bg-canopy-700",
  ghost: "bg-transparent text-canopy-700 hover:bg-canopy-100 dark:text-canopy-200 dark:hover:bg-canopy-800/50",
  danger: "bg-severity-high text-white hover:opacity-90",
};

const sizeStyles: Record<Size, string> = {
  sm: "text-sm px-3 py-1.5",
  md: "text-sm px-4 py-2.5",
  lg: "text-base px-6 py-3",
};

/** Base interactive control. Every button in the app should render through this
 *  component so hover/focus/disabled states stay consistent. */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", isLoading, disabled, children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors duration-150",
          "disabled:cursor-not-allowed disabled:opacity-60",
          variantStyles[variant],
          sizeStyles[size],
          className
        )}
        {...props}
      >
        {isLoading && (
          <span
            className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"
            aria-hidden="true"
          />
        )}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";
