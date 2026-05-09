const endpoint = process.env.AGENT_API_URL || "http://localhost:5173/api/agent";

const response = await fetch(endpoint, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "who is playing today?",
    capability: "read_only",
    context: {},
  }),
});

const text = await response.text();
let data;
try {
  data = JSON.parse(text);
} catch (error) {
  console.error(text);
  throw new Error(`Agent smoke test expected JSON from ${endpoint}, received status ${response.status}.`);
}

const requiredFields = ["answer", "used_sources", "warnings"];
const missingFields = requiredFields.filter((field) => !(field in data));
if (missingFields.length > 0) {
  throw new Error(`Agent smoke test response is missing fields: ${missingFields.join(", ")}`);
}

if ("reply" in data) {
  throw new Error("Agent smoke test response must use answer, not reply.");
}

if (typeof data.answer !== "string" || !Array.isArray(data.used_sources) || !Array.isArray(data.warnings)) {
  throw new Error("Agent smoke test response has invalid answer, used_sources, or warnings types.");
}

if (!response.ok) {
  console.error(JSON.stringify(data, null, 2));
  throw new Error(`Agent smoke test failed with HTTP ${response.status}.`);
}

console.log(JSON.stringify(data, null, 2));
