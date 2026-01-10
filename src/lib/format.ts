export const fmtNumber = (value: unknown, decimals = 2): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(decimals);
};

export const fmtPercent = (value: unknown, decimals = 2): string => {
  const formatted = fmtNumber(value, decimals);
  if (formatted === "—") {
    return "—";
  }
  return `${formatted}%`;
};

export const fmtCurrencyEUR = (value: unknown, decimals = 0): string => {
  const formatted = fmtNumber(value, decimals);
  if (formatted === "—") {
    return "—";
  }
  return `€${formatted}`;
};

export const fmtMoney = fmtCurrencyEUR;
export const fmtPct = fmtPercent;
