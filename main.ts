// gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase earn
// rate at the case's target, as five quantiles, scored by pinball loss on the
// logit scale.
//
// Model: per-trophy logit trend in log-time, recency-weighted, shrunk toward
// the tier mean, with a calibrated spread.
//
// Why this beats persistence: Exophase rates are not static. A trophy's rate
// is (players who earned it) / (players tracked). As the launch week goes on,
// two forces move every rate: progression (tracked players reach the trophy,
// pushing its rate up) and dilution (new players join Exophase tracking
// having earned nothing yet, pushing every rate down). Early, easy trophies
// rise because progression wins (Chapter 1 went 1.25% on launch night to
// 18% two days later); late, hard trophies fall because dilution wins
// (Chapter 3 went 1.10% to 0.07%, the platinum 0.63% to 0.00%). A forecast
// that holds the last rate (the house baseline) misses both moves. A line
// fitted to logit(rate) against ln(1 + hours since launch) captures the net
// drift, and ln(1 + hours) already damps the trend in linear time, so the
// early surge does not extrapolate without bound.
//
// Fit: weighted least squares on the case's history, weights decaying with a
// 24-hour half-life so recent reads dominate, over the full history (not just
// the last 24 hours, which the house curve baseline uses). Slopes shrink
// toward zero (persistence), so a noisy trophy does not chase its own noise
// and a tier's mean is never used as a prior, since a tier mixes rising
// early trophies with falling late ones. The median is the shrunk line read
// at the target, blended slightly toward the last rate. The spread combines the house random-walk term (0.5 logit units
// per square-root day), the slope's standard error times the extrapolation
// distance, and the fit's residual spread; trophies at 0% get a wider spread
// because they may emerge by the target. Quantiles are the median shifted by
// the standard normal's quantiles times the spread, mapped back to percent.
//
// Input: every *.in case in the input directory (MAKEFUN_INPUT_DIR or
// /input). Output: one *.out per case in the output directory
// (MAKEFUN_OUTPUT_DIR or /output): {"trophies":[{"id":n,
// "q":[q05,q25,q50,q75,q95]}, ...]}. Deterministic: no randomness, reads
// are processed in name order, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The game released 19 November 2026; hours are counted from its launch day. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Recency half-life for the trend fit, in hours: recent reads dominate. */
const HALF_LIFE_HOURS = 24;
/** Prior strength, in reads, for shrinking a trophy's slope to its tier mean. */
const SHRINK_K = 4;
/** The house random-walk spread, in logit units per square-root day. */
const BASE_SIGMA = 0.5;
/** Spread floor and ceiling, in logit units. */
const SPREAD_MIN = 0.4;
const SPREAD_MAX = 1.8;
/** Trophies at or below this last rate get a wider spread: they may emerge. */
const EMERGE_RATE = 0.05;
const EMERGE_SPREAD = 1.0;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => {
  const p = 1 / (1 + Math.exp(-l));
  return Math.min(100, Math.max(0, p * 100));
};
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from launch to an ISO timestamp, floored at 0. */
const hoursSinceLaunch = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - LAUNCH_MS) / 3600000);

interface Fit {
  /** Intercept of the weighted line. */
  a: number;
  /** Slope of the weighted line. */
  b: number;
  /** Standard error of the slope. */
  seB: number;
  /** Residual standard deviation around the line. */
  residSd: number;
  /** Weighted mean of x, the line's anchor. */
  xBar: number;
  /** Weighted mean of y. */
  yBar: number;
  /** Number of reads used. */
  n: number;
  /** Effective number of reads (sum of weights). */
  nEff: number;
}

/** Weighted least squares of y on x. Returns null when the fit is degenerate. */
function wls(xs: number[], ys: number[], ws: number[]): Fit | null {
  const n = xs.length;
  if (n < 2) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i]!;
    sx += ws[i]! * xs[i]!;
    sy += ws[i]! * ys[i]!;
  }
  if (sw <= 0) return null;
  const xBar = sx / sw;
  const yBar = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i]! - xBar;
    sxx += ws[i]! * dx * dx;
    sxy += ws[i]! * dx * (ys[i]! - yBar);
  }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  const a = yBar - b * xBar;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i]! - (a + b * xs[i]!);
    sse += ws[i]! * r * r;
  }
  const dof = Math.max(1, n - 2);
  const residSd = Math.sqrt(sse / (sw * (dof / n)));
  const seB = residSd / Math.sqrt(sxx);
  return { a, b, seB, residSd, xBar, yBar, n, nEff: sw };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  // Every read must carry one rate per trophy; a short or long row would
  // silently misalign the trophies that follow it.
  for (let ri = 0; ri < c.history.length; ri++) {
    if (c.history[ri]!.rates.length !== c.trophies.length) {
      throw new Error(`${file} read ${ri} has ${c.history[ri]!.rates.length} rates, not ${c.trophies.length}`);
    }
  }
  const checkpointMs = Date.parse(c.at);
  const targetHours = hoursSinceLaunch(c.target);
  const xTarget = Math.log(1 + targetHours);
  const horizonDays = c.horizon_minutes / 1440;

  // First pass: fit every trophy's trend.
  interface TrophyFit {
    trophy: Trophy;
    lastRate: number;
    fit: Fit | null;
    xLast: number;
  }
  const fits: TrophyFit[] = [];
  for (let ti = 0; ti < c.trophies.length; ti++) {
    const trophy = c.trophies[ti]!;
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    let lastRate: number | null = null;
    let xLast = 0;
    for (const read of c.history) {
      const rate = read.rates[ti];
      if (rate === null || rate === undefined) continue;
      const h = hoursSinceLaunch(read.at);
      const x = Math.log(1 + h);
      const ageHours = Math.max(0, (checkpointMs - Date.parse(read.at)) / 3600000);
      const w = Math.pow(0.5, ageHours / HALF_LIFE_HOURS);
      xs.push(x);
      ys.push(logit(rate));
      ws.push(w);
      lastRate = rate;
      xLast = x;
    }
    if (lastRate === null) lastRate = 0;
    fits.push({ trophy, lastRate, fit: wls(xs, ys, ws), xLast });
  }

  // Second pass: shrunk forecast and spread per trophy.
  const trophies = fits.map((f) => {
    const { trophy, lastRate } = f;
    // A trophy everyone already holds stays held.
    if (lastRate >= 99.99) {
      return { id: trophy.id, q: [100, 100, 100, 100, 100] };
    }
    const lLast = logit(lastRate);
    let lMed = lLast;
    let spread = BASE_SIGMA * Math.sqrt(horizonDays);
    if (f.fit !== null) {
      const fit = f.fit;
      // Shrink the slope toward zero (persistence), not toward a tier mean:
      // a tier mixes rising early trophies with falling late ones, so its
      // mean is a bad prior for either. The data must earn the trend.
      const bShrunk = (fit.nEff * fit.b) / (fit.nEff + SHRINK_K);
      const aShrunk = fit.yBar - bShrunk * fit.xBar;
      const lTrend = aShrunk + bShrunk * xTarget;
      // Blend the trend toward the last rate: the trend is the signal, the
      // last rate anchors it against over-extrapolation.
      lMed = 0.9 * lTrend + 0.1 * lLast;
      const dx = Math.max(0, xTarget - f.xLast);
      const sSlope = fit.seB * dx;
      const sResid = fit.residSd;
      spread = Math.sqrt(
        spread * spread + sSlope * sSlope + sResid * sResid,
      );
    }
    // A trophy nobody holds yet may emerge by the target: widen the spread.
    if (lastRate <= EMERGE_RATE) {
      spread = Math.max(spread, EMERGE_SPREAD);
    }
    spread = Math.min(SPREAD_MAX, Math.max(SPREAD_MIN, spread));
    const q = Z.map((z) => round4(invLogitPct(lMed + z * spread)));
    // Guard the shape: finite, 0 to 100, non-decreasing.
    for (let j = 0; j < q.length; j++) {
      if (!Number.isFinite(q[j])) q[j] = j === 0 ? 0 : q[j - 1]!;
      q[j] = Math.min(100, Math.max(0, q[j]!));
      if (j > 0 && q[j]! < q[j - 1]!) q[j] = q[j - 1]!;
    }
    return { id: trophy.id, q };
  });

  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
  // Per-case diagnostic on stderr: median q50 and median implied spread.
  const median = (xs: number[]): number => xs.sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;
  const medQ50 = median(trophies.map((t) => t.q[2]!));
  const medSpread = median(trophies.map((t) => (logit(t.q[3]!) - logit(t.q[1]!)) / (2 * 0.674)));
  console.error(`case ${id}: ${trophies.length} trophies, median q50 ${round4(medQ50)}%, median spread ${round4(medSpread)} logit`);
}
