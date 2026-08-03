/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Design rationale (see chat explanation): a botanical/diagnostic
        // palette rather than the default indigo-on-white SaaS look.
        canopy: {
          50: "#f2f7f3",
          100: "#dfebe1",
          200: "#b9d4bf",
          300: "#8cb896",
          400: "#5f9a6d",
          500: "#3f7d4d",
          600: "#2f643c",
          700: "#265032",
          800: "#1f4029",
          900: "#0e1f14", // near-black, tinted green — dark mode background
        },
        scan: {
          // "Scan" family maps to Grad-CAM heat scale: cool = low activation,
          // warm = high activation. Used for confidence bars & heatmap legends.
          cool: "#3a7ca5",
          mid: "#e9c46a",
          hot: "#e0623d",
        },
        severity: {
          low: "#4a8b5c",
          moderate: "#d99a2b",
          high: "#c94f3f",
          critical: "#9c2b2b",
        },
        surface: {
          light: "#f7f8f5",
          dark: "#0e1310",
        },
      },
      fontFamily: {
        // Display/UI face: geometric, confident, used for headings & nav.
        sans: ["'Inter'", "system-ui", "sans-serif"],
        // Data face: monospace for metrics, confidence %, inference time —
        // reinforces "instrument readout" feeling for prediction data.
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      backdropBlur: {
        xs: "2px",
      },
      boxShadow: {
        soft: "0 4px 24px -8px rgba(14, 31, 20, 0.12)",
        glass: "0 8px 32px 0 rgba(14, 31, 20, 0.15)",
      },
      borderRadius: {
        xl2: "1.25rem",
      },
      animation: {
        "fade-in": "fadeIn 0.4s ease-out",
        "slide-up": "slideUp 0.4s ease-out",
      },
      keyframes: {
        fadeIn: { "0%": { opacity: 0 }, "100%": { opacity: 1 } },
        slideUp: {
          "0%": { opacity: 0, transform: "translateY(12px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
