import fs from "fs";
import path from "path";

const dataDir = process.argv[2] ?? path.join(process.cwd(), "public", "data");
const statePath = path.join(dataDir, "dashboard_state.json");
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

const dashboardState = loadJson(statePath);
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
const strategyMatches = localRows.filter((row) => {
  const date = parseDate(row.date || row.game_date || row.event_date);
  if (!date) {
    return false;
  }
  return date >= startDate && date <= endDate;
}).length;

if (dashboardState.strategy_matches_window !== strategyMatches) {
  throw new Error(
    `strategy_matches_window mismatch: state=${dashboardState.strategy_matches_window} computed=${strategyMatches}`,
  );
}

console.log("✔ Dashboard state parity validated.");
