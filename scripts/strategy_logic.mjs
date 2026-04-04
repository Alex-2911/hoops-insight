import fs from "fs";

const normalizeKey = (key) =>
  key
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_")
    .replace(/[^a-z0-9_]/g, "")
    .replace(/_+/g, "_");

const coerceNumber = (value) => {
  if (value === null || value === undefined) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const parseDate = (value) => {
  if (!value) return null;
  const trimmed = String(value).trim();
  if (!trimmed) return null;
  const date = new Date(trimmed);
  if (Number.isNaN(date.getTime())) return null;
  return date;
};

export const DEFAULT_PARAMS = {
  home_win_rate_threshold: 0,
  odds_min: 1,
  odds_max: 3.2,
  prob_threshold: 0.5,
  min_ev: 0,
};

const extractParamDict = (payload) => {
  const candidates = [];
  ["params", "params_used", "active_params", "thresholds", "strategy", "filters"].forEach((key) => {
    const value = payload?.[key];
    if (value && typeof value === "object" && !Array.isArray(value)) {
      candidates.push(value);
    }
  });
  const meta = payload?.meta;
  if (meta && typeof meta === "object" && !Array.isArray(meta)) {
    ["params", "params_used", "active_params", "thresholds", "strategy"].forEach((key) => {
      const value = meta[key];
      if (value && typeof value === "object" && !Array.isArray(value)) {
        candidates.push(value);
      }
    });
  }

  const hasKnown = (obj) => {
    const keys = Object.keys(obj || {}).map((k) => normalizeKey(k));
    return ["home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold", "min_ev", "min_ev_per_100"].some((k) => keys.includes(k));
  };
  const matched = candidates.find(hasKnown);
  if (matched) return matched;

  return Object.fromEntries(
    Object.entries(payload ?? {}).filter(([k]) => {
      const nk = normalizeKey(k);
      return [
        "home_win_rate_threshold", "min_home_win_rate", "home_win_rate_min",
        "odds_min", "min_odds_1", "min_odds",
        "odds_max", "max_odds_1", "max_odds",
        "prob_threshold", "min_prob_used", "min_prob", "min_prob_iso",
        "min_ev", "min_ev_per_100", "min_ev_eur_per_100",
      ].includes(nk);
    }),
  );
};

export const loadStrategyParamsVersioned = (filePath) => {
  if (!fs.existsSync(filePath)) {
    return { version: 1, params: { ...DEFAULT_PARAMS }, source: "defaults" };
  }
  const payload = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  const version = Number(payload?.version ?? 1);
  let raw = extractParamDict(payload);
  const params = { ...DEFAULT_PARAMS };
  Object.entries(raw).forEach(([k, v]) => {
    const n = coerceNumber(v);
    if (n !== null) {
      let nk = normalizeKey(k);
      if (["min_ev_per_100", "min_ev_eur_per_100"].includes(nk)) nk = "min_ev";
      else if (["min_home_win_rate", "home_win_rate_min"].includes(nk)) nk = "home_win_rate_threshold";
      else if (["min_odds_1", "min_odds"].includes(nk)) nk = "odds_min";
      else if (["max_odds_1", "max_odds"].includes(nk)) nk = "odds_max";
      else if (["min_prob", "min_prob_iso", "min_prob_used"].includes(nk)) nk = "prob_threshold";
      params[nk] = n;
    }
  });
  return { version, params, source: filePath };
};

export const filterLocalMatchedGamesParams = (rows, paramsObj) => {
  const params = paramsObj?.params ?? DEFAULT_PARAMS;
  return rows
    .map((row) => {
      const prob = coerceNumber(row.prob_used);
      const odds = coerceNumber(row.odds_1 ?? row.closing_home_odds);
      const ev = coerceNumber(row.ev_eur_per_100 ?? row.ev_per_100);
      const hwr = coerceNumber(row.home_win_rate);
      const gap = prob !== null && odds ? prob - 1 / odds : null;
      const blocked = [];
      if (prob === null || prob < params.prob_threshold) blocked.push("prob_threshold");
      if (odds === null || odds < params.odds_min || odds > params.odds_max) blocked.push("odds_range");
      if (ev === null || ev <= params.min_ev) blocked.push("min_ev");
      if (hwr === null || hwr < params.home_win_rate_threshold) blocked.push("home_win_rate_threshold");
      return {
        ...row,
        prob_used: prob,
        model_market_gap: gap,
        model_market_gap_flag: gap !== null ? gap > 0 : false,
        blocked_by: blocked.join("|"),
      };
    })
    .filter((row) => row.blocked_by === "");
};

export const filterLocalMatchedGamesWindow = (rows, startDate, endDate) => {
  const start = new Date(startDate);
  const end = new Date(endDate);
  return rows.filter((row) => {
    const rowDate = parseDate(row.date);
    if (!rowDate) return false;
    return rowDate >= start && rowDate <= end;
  });
};
