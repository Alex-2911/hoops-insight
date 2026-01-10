export const fmtNumber = (value: unknown, decimals = 2): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(decimals);
};

export const fmtMoney = (
  value: unknown,
  decimals = 2,
  currency = "€",
): string => {
  const formatted = fmtNumber(value, decimals);
  if (formatted === "—") {
    return "—";
  }
  return `${currency}${formatted}`;
};

export const fmtPct = (value: unknown, decimals = 2): string => {
  const formatted = fmtNumber(value, decimals);
  if (formatted === "—") {
    return "—";
  }
  return `${formatted}%`;
};
