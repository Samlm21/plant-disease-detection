import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Leaf, Moon, Sun, Menu, X } from "lucide-react";
import { useTheme } from "@context/ThemeContext";
import { cn } from "@utils/cn";

const links = [
  { to: "/", label: "Home" },
  { to: "/predict", label: "Predict" },
  { to: "/history", label: "History" },
  { to: "/about", label: "About" },
  { to: "/contact", label: "Contact" },
];

export function Navbar() {
  const { theme, toggleTheme } = useTheme();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-canopy-200/40 bg-surface-light/80 backdrop-blur-md dark:border-canopy-800/40 dark:bg-surface-dark/80">
      <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6" aria-label="Primary">
        <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <Leaf className="h-5 w-5 text-canopy-600" aria-hidden="true" />
          Plant Disease Detector
        </NavLink>

        <div className="hidden items-center gap-1 md:flex">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                cn(
                  "rounded-md px-3 py-2 text-sm font-medium text-canopy-700 hover:bg-canopy-100 dark:text-canopy-200 dark:hover:bg-canopy-800/60",
                  isActive && "bg-canopy-100 text-canopy-900 dark:bg-canopy-800 dark:text-white"
                )
              }
            >
              {link.label}
            </NavLink>
          ))}
          <button
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            className="ml-2 rounded-md p-2 text-canopy-700 hover:bg-canopy-100 dark:text-canopy-200 dark:hover:bg-canopy-800/60"
          >
            {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>
        </div>

        <button
          className="p-2 md:hidden"
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
        >
          {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-canopy-200/40 px-4 py-3 md:hidden dark:border-canopy-800/40">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              onClick={() => setOpen(false)}
              className="block rounded-md px-3 py-2 text-sm font-medium text-canopy-700 hover:bg-canopy-100 dark:text-canopy-200 dark:hover:bg-canopy-800/60"
            >
              {link.label}
            </NavLink>
          ))}
          <button
            onClick={toggleTheme}
            className="mt-1 flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm font-medium text-canopy-700 hover:bg-canopy-100 dark:text-canopy-200 dark:hover:bg-canopy-800/60"
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            Toggle theme
          </button>
        </div>
      )}
    </header>
  );
}
