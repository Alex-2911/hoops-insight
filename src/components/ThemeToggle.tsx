import React from "react";

const ThemeToggle = () => {
  const [theme, setTheme] = React.useState<"light" | "dark">(() =>
    document.documentElement.classList.contains("dark") ? "dark" : "light",
  );

  const toggleTheme = () => {
    setTheme((prevTheme) => {
      const nextTheme = prevTheme === "light" ? "dark" : "light";
      document.documentElement.classList.toggle("dark", nextTheme === "dark");
      localStorage.setItem("theme", nextTheme);
      return nextTheme;
    });
  };

  return (
    <button
      className="rounded px-3 py-1 text-sm text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground"
      onClick={toggleTheme}
      type="button"
    >
      Switch to {theme === "light" ? "dark" : "light"} theme
    </button>
  );
};

export default ThemeToggle;
