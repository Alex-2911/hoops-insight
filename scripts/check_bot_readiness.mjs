import fs from "fs";
import path from "path";

const args = new Set(process.argv.slice(2));
const strict = args.has("--strict");
const dataDirArg = process.argv.find((arg) => arg.startsWith("--data-dir="));
const dataDir = path.resolve(dataDirArg?.split("=")[1] ?? "public/data");
const today = process.env.READINESS_TODAY || new Date().toISOString().slice(0, 10);

const requiredDashboardFiles = [
  "summary.json",
  "tables.json",
  "dashboard_state.json",
  "dashboard_payload.json",
  "last_run.json",
  "strategy_params.json",
  "combined_latest.csv",
  "local_matched_games_latest.csv",
  "actual_bets_manual.csv",
];

const expectedPipelineArtifacts = [
  "metrics_snapshot.json",
  "stage1_daily_snapshot_latest.csv",
  "stage1_daily_snapshot_latest.json",
  "setup_profitability_scan_latest.csv",
  "setup_profitability_scan_latest.json",
  "script11_watchlist_history_latest.csv",
  "script11_watchlist_history_latest.json",
];

const readJson = (fileName) => {
  const filePath = path.join(dataDir, fileName);
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
};

const missingRequired = requiredDashboardFiles.filter((fileName) => !fs.existsSync(path.join(dataDir, fileName)));
const missingPipelineArtifacts = expectedPipelineArtifacts.filter((fileName) => !fs.existsSync(path.join(dataDir, fileName)));
const summary = readJson("summary.json");
const dashboardState = readJson("dashboard_state.json");
const asOfDate = dashboardState?.as_of_date || summary?.as_of_date || null;
const backendConfigured = Boolean(process.env.HOOPS_AGENT_API_URL || process.env.OPENAI_API_KEY);
const frontendConfigured = Boolean(process.env.VITE_HOOPS_AGENT_API_URL) || fs.existsSync(path.resolve("api/agent.ts"));

const issues = [];
if (missingRequired.length > 0) issues.push(`Missing required dashboard files: ${missingRequired.join(", ")}`);
if (missingPipelineArtifacts.length > 0) issues.push(`Missing optional pipeline artifacts: ${missingPipelineArtifacts.join(", ")}`);
if (!asOfDate) issues.push("No as_of_date found in summary/dashboard_state.");
if (asOfDate && asOfDate < today) issues.push(`Dashboard data is stale: as_of_date=${asOfDate}, today=${today}`);
if (!frontendConfigured) issues.push("Agent frontend has no VITE_HOOPS_AGENT_API_URL and no local api/agent.ts fallback.");
if (!backendConfigured) issues.push("Agent backend credentials are not configured: set HOOPS_AGENT_API_URL or OPENAI_API_KEY.");

console.log("Hoops bot readiness report");
console.log(`- data_dir: ${dataDir}`);
console.log(`- today: ${today}`);
console.log(`- as_of_date: ${asOfDate ?? "missing"}`);
console.log(`- required dashboard files: ${missingRequired.length === 0 ? "ok" : `missing ${missingRequired.length}`}`);
console.log(`- optional pipeline artifacts: ${missingPipelineArtifacts.length === 0 ? "ok" : `missing ${missingPipelineArtifacts.length}`}`);
console.log(`- agent frontend endpoint: ${frontendConfigured ? "configured" : "missing"}`);
console.log(`- agent backend credentials: ${backendConfigured ? "configured" : "missing"}`);

if (issues.length > 0) {
  console.log("\nIssues:");
  for (const issue of issues) console.log(`- ${issue}`);
}

if (strict && issues.length > 0) {
  process.exit(1);
}
