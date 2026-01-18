import fs from "fs";
import path from "path";

const dataDir = process.argv[2] ?? path.join(process.cwd(), "public", "data");
const statePath = path.join(dataDir, "dashboard_state.json");
const strategyParamsPath = path.join(dataDir, "strategy_params.json");
const combinedPath = path.join(dataDir, "combined_latest.csv");
const localMatchedPath = path.join(dataDir, "local_matched_games_latest.csv");

const loadJson = (filePath) => {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing required file: ${filePath}`);
  }
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
};

const normalizeKey = (key) =>
  key
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_")
    .replace(/[^a-z0-9_]/g, "")
    .replace(/_+/g, "_");

const parseCsvLine = (line) => {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (char === "," && !inQuotes) {
      result.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  result.push(current);
  return result;
};

const parseCsv = (filePath) => {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing required file: ${filePath}`);
  }
  const text = fs.readFileSync(filePath, "utf-8").trim();
  if (!text) {
    return [];
  }
  const lines = text.split(/\r?\n/);
  const headers = parseCsvLine(lines[0]).map((h) => normalizeKey(h));
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = parseCsvLine(line);
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = values[idx] ?? "";
    });
    return row;
  });
};

const parseDate = (value) => {
  if (!value) {
    return null;
  }
  const trimmed = String(value).trim();
  if (!trimmed) {
    return null;
  }
  const date = new Date(trimmed);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date;
};

const formatDate = (date) => date.toISOString().slice(0, 10);

const coerceNumber = (value) => {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const trimmed = String(value).trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeParams = (params) => {
  if (!params || typeof params !== "object") {
    return {};
  }
  return Object.fromEntries(
    Object.entries(params).map(([key, value]) => [normalizeKey(key), value]),
  );
};

const loadStrategyParams = (filePath) => {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  const raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  if (!raw || typeof raw !== "object") {
    return {};
  }
  const candidates = [raw.params_used, raw.params, raw];
  const selected = candidates.find((candidate) => candidate && typeof candidate === "object");
  return normalizeParams(selected ?? {});
};

const getParam = (params, ...names) => {
  for (const name of names) {
    const normalized = normalizeKey(name);
    if (Object.prototype.hasOwnProperty.call(params, normalized)) {
      return coerceNumber(params[normalized]);
    }
  }
  return null;
};

const filterLocalMatchedGamesWindow = (rows, windowStart, windowEnd) => {
  if (!rows.length || !windowStart || !windowEnd) {
    return rows;
  }
  const startDate = parseDate(windowStart);
  const endDate = parseDate(windowEnd);
  if (!startDate || !endDate) {
    return rows;
  }
  const rowDates = rows
    .map((row) => parseDate(row.date))
    .filter((date) => date !== null);
  if (rowDates.length) {
    const minDate = new Date(Math.min(...rowDates.map((date) => date.getTime())));
    const maxDate = new Date(Math.max(...rowDates.map((date) => date.getTime())));
    if (endDate < minDate || startDate > maxDate) {
      return rows;
    }
  }
  return rows.filter((row) => {
    const rowDate = parseDate(row.date);
    return rowDate && rowDate >= startDate && rowDate <= endDate;
  });
};

const filterLocalMatchedGamesParams = (rows, params) => {
  if (!rows.length) {
    return rows;
  }
  const minProbUsed = getParam(
    params,
    "prob_threshold",
    "min_prob_used",
    "min_prob",
    "min_prob_iso",
  );
  const minOdds = getParam(params, "odds_min", "min_odds_1", "min_odds");
  let maxOdds = getParam(params, "odds_max", "max_odds_1", "max_odds");
  if (maxOdds === null) {
    maxOdds = 3.2;
  }
  const minEv = getParam(params, "min_ev", "min_ev_eur_per_100", "min_ev_per_100");
  const minHomeWinRate = getParam(params, "home_win_rate_threshold", "min_home_win_rate");

  let current = rows;
  if (minProbUsed !== null) {
    current = current.filter((row) => {
      const probUsed = coerceNumber(row.prob_used);
      return probUsed !== null && probUsed >= minProbUsed;
    });
  }
  if (minOdds !== null) {
    current = current.filter((row) => {
      const odds = coerceNumber(row.odds_1);
      return odds !== null && odds >= minOdds;
    });
  }
  if (maxOdds !== null) {
    current = current.filter((row) => {
      const odds = coerceNumber(row.odds_1);
      return odds !== null && odds <= maxOdds;
    });
  }
  if (minEv !== null) {
    current = current.filter((row) => {
      const ev = coerceNumber(row.ev_eur_per_100);
      return ev !== null && ev > minEv;
    });
  }
  if (minHomeWinRate !== null) {
    current = current.filter((row) => {
      const winRate = coerceNumber(row.home_win_rate);
      return winRate !== null && winRate >= minHomeWinRate;
    });
  }
  return current;
};

const dashboardState = loadJson(statePath);
const strategyParams = loadStrategyParams(strategyParamsPath);
const combinedRows = parseCsv(combinedPath);
const localRows = parseCsv(localMatchedPath);

const playedRows = combinedRows
  .map((row) => {
    const result = row.result || row.result_raw;
    if (!result || String(result).trim() === "" || String(result).trim() === "0") {
      return null;
    }
    const date = parseDate(row.game_date || row.date);
    if (!date) {
      return null;
    }
    return { date };
  })
  .filter(Boolean);

if (!playedRows.length) {
  throw new Error("No played games found in combined_latest.csv.");
}

playedRows.sort((a, b) => a.date.getTime() - b.date.getTime());
const windowSize = dashboardState.window_size;
const windowRows = playedRows.slice(-windowSize);
if (!windowRows.length) {
  throw new Error("Window selection produced zero rows.");
}

const computedWindowStart = formatDate(windowRows[0].date);
const computedWindowEnd = formatDate(windowRows[windowRows.length - 1].date);

if (dashboardState.window_start !== computedWindowStart) {
  throw new Error(
    `window_start mismatch: state=${dashboardState.window_start} computed=${computedWindowStart}`,
  );
}

if (dashboardState.window_end !== computedWindowEnd) {
  throw new Error(
    `window_end mismatch: state=${dashboardState.window_end} computed=${computedWindowEnd}`,
  );
}

const startDate = new Date(computedWindowStart);
const endDate = new Date(computedWindowEnd);
const windowFilteredLocalRows = filterLocalMatchedGamesWindow(
  localRows.map((row) => ({
    ...row,
    date: row.date || row.game_date || row.event_date,
  })),
  computedWindowStart,
  computedWindowEnd,
);
const strategyMatches = filterLocalMatchedGamesParams(
  windowFilteredLocalRows,
  strategyParams,
).length;

if (dashboardState.strategy_matches_window !== strategyMatches) {
  throw new Error(
    `strategy_matches_window mismatch: state=${dashboardState.strategy_matches_window} computed=${strategyMatches}`,
  );
}

console.log("✔ Dashboard state parity validated.");
