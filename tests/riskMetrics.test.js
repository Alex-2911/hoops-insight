import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { RISK_METRICS_MIN_BETS, shouldShowRiskMetrics } from "../src/lib/riskMetrics.js";

describe("shouldShowRiskMetrics", () => {
  it("hides metrics for 0 bets", () => {
    assert.equal(shouldShowRiskMetrics(0), false);
  });

  it("hides metrics for 1 bet", () => {
    assert.equal(shouldShowRiskMetrics(1), false);
  });

  it("hides metrics for 2 bets", () => {
    assert.equal(shouldShowRiskMetrics(2), false);
  });

  it("shows metrics at the minimum threshold", () => {
    assert.equal(shouldShowRiskMetrics(RISK_METRICS_MIN_BETS), true);
  });

  it("shows metrics for larger samples", () => {
    assert.equal(shouldShowRiskMetrics(20), true);
  });
});
