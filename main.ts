// gta6-twitch-forecast — main.ts (model v3)
//
// Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category
// for hours 2..24 of a launch day, as five quantiles per hour, from the
// case's hour-1 print plus the launch-day shapes of past big launches.
//
// Model v3 (pooled pinball-median regression with a smooth hour-varying
// slope and a viewers-per-channel level):
//   For each comparable launch, form log-ratios r_{i,h} = ln(viewers_h /
//   viewers_1) for h = 2..24.  A launch only counts when it is a real
//   rolling-midnight console launch: tier "comparable" (no early-access
//   contamination, no free-to-play, no slow-burn weak launches), no PC in
//   its platforms, and an hour-1 print already at least 10% of its own
//   release-day peak (so hour 1 is a post-unlock hour, not a near-zero
//   pre-unlock reading — simultaneous global unlocks fail this and would
//   otherwise blow the ratios up).
//   Then fit ONE pooled model over every (launch, hour) point:
//     r_{i,h} = s_h + (b0 + b1*u_h + b2*u_h^2) * ln(h1_i)
//                 + c0 * (ln vpc_i - mean ln vpc)
//     u_h = max(0, h - 5) / 19   (ramps from the US-East unlock, 05:00Z,
//     to the end of the day), vpc_i = hour-1 viewers / hour-1 channels.
//   The 23 hour intercepts s_h carry the shared unlock-wave shape (the
//   US-East 05:00Z and US-West 08:00Z jumps, the evening prime-time
//   peak); the slope basis lets the scale effect grow through the day
//   instead of estimating 23 separate noisy per-hour slopes from ~7
//   launches each; the viewers-per-channel term is a level shift —
//   broad grassroots audiences (low viewers per channel) run hotter all
//   day relative to hour 1 than concentrated big-streamer audiences.
//   THE V3 CHANGE — the median curve is fitted by direct pinball loss,
//   not least squares: the 27 coefficients are the minimiser of
//   sum rho_0.5(y - x'beta) over all fitted points (iteratively
//   reweighted least squares from the least-squares solution, fixed 60
//   iterations, smoothing eps 0.05, deterministic).  The scored metric
//   IS the pinball loss, so the median that minimises it on the training
//   launches beats the least-squares median: least squares is pulled by
//   the extreme launches (a small launch's +1.7 evening surge, a
//   mega-launch's flat evening), while the pinball median tracks the
//   typical launch.  The spread is unchanged from v2: the POOLED
//   least-squares residual standard deviation x 1.3, and the five
//   quantiles are median + z_tau * spread (z = -1.645, -0.674, 0, 0.674,
//   1.645).  Direct pinball fits of the tail quantiles were tried and
//   rejected (they overfit: practice 0.192, CV-11 0.161); the pooled
//   residual distribution is close to Gaussian (fitted spread
//   multipliers -1.64/-0.52/+0.08/+0.50/+1.72 vs Gaussian z), so the
//   Gaussian spread stays.
//   Selected by leave-one-launch-out cross-validation over the console
//   comparables AND the four practice cases (a variant had to win on
//   both): on the four practice cases v3 scores a mean pinball loss of
//   ~0.177 against v2's ~0.179 and the house baseline's ~0.387; per
//   practice case: Wolverine 0.254, Red Dead Redemption 2 0.091, God of
//   War Ragnarok 0.136, Spider-Man 2 0.228.  Leave-one-launch-out over
//   the 11 console launches (the 7 comparables plus the 4 practice
//   launches, as the final case has them): 0.150 vs v2's 0.156 — v3 wins
//   on 8 of the 11 launches, most on the surge launches (Wolverine
//   -0.027, Saros -0.026, Zelda -0.018).  The final case trains on all
//   11, so the same robustness applies there.
//   Variants tried and rejected on cross-validation: per-quantile
//   pinball fits, pinball-fitted spread multipliers, per-hour spreads,
//   Huber medians (between least squares and pinball), smoothed hour
//   intercepts, pre-launch history features (f48, pre-launch peak —
//   they win on the practice cases but lose badly on
//   leave-one-launch-out), wider (PC-inclusive) training pool,
//   quadratic in ln hour-1, recency weighting, t / empirical residual
//   quantiles, asymmetric spreads, and the ln h1 x vpc interaction
//   (practice 0.176 but CV-11 0.167 — classic overfitting).
//   Fallbacks: if fewer than 5 console comparables have an hour, widen
//   to every tier-comparable launch with a real hour 1; if still fewer
//   than 5, fall back to the flat-at-hour-1 forecast with the baseline
//   spread.
//
// Deterministic: no clock, no randomness, no network, stdlib only.
// Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input/), in name
// order, and writes one /output/<id>.out per case ($MAKEFUN_OUTPUT_DIR
// else /output/).  Two runs on the same case write identical bytes.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

type HourlyPoint = { from: string; viewers: number | null; channels?: number | null };
type Comparable = {
  name: string;
  day_start: string;
  unlock?: string | null;
  platforms?: string | null;
  hourly: HourlyPoint[];
};

type CaseFile = {
  id: string;
  known: { hour: number; viewers: number; channels?: number | null }[];
  forecast?: { hour: number }[];
  comparables: Comparable[];
};

const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const SIGMA_MULT = 1.3;
const MIN_COMPARABLES = 5;
const VPC_CLAMP = 1.5;
// Pinball-median fit: fixed 60 IRLS iterations, smoothing eps 0.05.
const PB_ITERS = 60;
const PB_EPS = 0.05;

// A launch's release-day hours 1..24 (viewers > 0), keyed by hour.
function dayHours(comp: Comparable): Map<number, number> {
  const out = new Map<number, number>();
  const t0 = Date.parse(comp.day_start);
  if (!Number.isFinite(t0)) return out;
  for (const pt of comp.hourly || []) {
    if (pt.viewers == null || pt.viewers <= 0) continue;
    const t = Date.parse(pt.from);
    if (!Number.isFinite(t)) continue;
    const dh = (t - t0) / 3600000;
    if (dh >= 0 && dh < 24 && dh === Math.floor(dh)) {
      out.set(dh + 1, pt.viewers);
    }
  }
  return out;
}

// Hour-1 channels of a comparable (null when the source has none).
function hour1Channels(comp: Comparable): number | null {
  const t0 = Date.parse(comp.day_start);
  if (!Number.isFinite(t0)) return null;
  for (const pt of comp.hourly || []) {
    if (pt.viewers == null || pt.viewers <= 0) continue;
    const t = Date.parse(pt.from);
    if (!Number.isFinite(t)) continue;
    if (t === t0 && pt.channels != null && pt.channels > 0) return pt.channels;
  }
  return null;
}

function tierOf(comp: Comparable): string {
  const m = /Tier:\s*([a-z-]+)/.exec(comp.unlock || "");
  return m ? m[1] : "";
}

// A comparable counts for the primary set when it is a clean,
// rolling-midnight console launch with a real (post-unlock) hour 1.
function isPrimary(comp: Comparable, dh: Map<number, number>): boolean {
  if (tierOf(comp) !== "comparable") return false;
  if ((comp.platforms || "").includes("PC")) return false;
  const h1 = dh.get(1);
  if (h1 == null) return false;
  let peak = 0;
  for (const v of dh.values()) if (v > peak) peak = v;
  return peak > 0 && h1 >= 0.1 * peak;
}

// The wider set: any clean launch with a real hour 1 (any platform).
function isWider(comp: Comparable, dh: Map<number, number>): boolean {
  if (tierOf(comp) !== "comparable") return false;
  const h1 = dh.get(1);
  if (h1 == null) return false;
  let peak = 0;
  for (const v of dh.values()) if (v > peak) peak = v;
  return peak > 0 && h1 >= 0.1 * peak;
}

// Slope basis: ramps from the US-East unlock (05:00Z, hour 5) to the
// end of the day.  b(u) = b0 + b1*u + b2*u^2 is the log-ratio slope on
// ln(hour-1 print) at hour h.
function uOf(h: number): number {
  return Math.max(0, h - 5) / 19;
}

// One design row of the pooled model: 23 hour intercepts (s_2..s_24),
// then ln(h1), ln(h1)*u, ln(h1)*u^2, centred ln viewers-per-channel.
function designRow(h: number, x: number, v: number): number[] {
  const row = new Array<number>(27).fill(0);
  row[h - 2] = 1;
  const u = uOf(h);
  row[23] = x;
  row[24] = x * u;
  row[25] = x * u * u;
  row[26] = v;
  return row;
}

type FitPt = { h: number; x: number; v: number; y: number };

// The pinball-median (tau = 0.5) fit: minimise sum rho_0.5(y - x'beta)
// by iteratively reweighted least squares from the least-squares
// solution.  The pinball loss rho_0.5(r) = 0.5*|r| has gradient
// 0.5*sign(r), so the IRLS weight is 0.5 / sqrt(r^2 + eps^2) (the eps
// smooths the kink at r = 0).  Fixed iteration count, fixed eps, no
// randomness, no clock: the same input always gives the same output.
// Returns the least-squares coefficients unchanged when a step fails.
function pinballMedianFit(pts: FitPt[], init: number[]): number[] {
  const NPAR = 27;
  let beta = [...init];
  const rows = pts.map((pt) => ({ row: designRow(pt.h, pt.x, pt.v), y: pt.y }));
  for (let iter = 0; iter < PB_ITERS; iter++) {
    const A: number[][] = Array.from({ length: NPAR }, () => new Array<number>(NPAR).fill(0));
    const rhs = new Array<number>(NPAR).fill(0);
    for (const { row, y } of rows) {
      let pred = 0;
      for (let i = 0; i < NPAR; i++) pred += row[i] * beta[i];
      const r = y - pred;
      const w = 0.5 / Math.sqrt(r * r + PB_EPS * PB_EPS);
      for (let i = 0; i < NPAR; i++) {
        if (row[i] === 0) continue;
        rhs[i] += w * row[i] * y;
        for (let j = 0; j < NPAR; j++) {
          if (row[j] !== 0) A[i][j] += w * row[i] * row[j];
        }
      }
    }
    for (let i = 0; i < NPAR; i++) A[i][i] += 1e-8;
    const next = solveLinear(A, rhs);
    if (next == null) break;
    let delta = 0;
    for (let i = 0; i < NPAR; i++) {
      const d = Math.abs(next[i] - beta[i]);
      if (d > delta) delta = d;
    }
    beta = next;
    if (delta < 1e-10) break;
  }
  return beta;
}

// Solve A x = b by Gaussian elimination with partial pivoting.
// A is modified in place; returns null when the system is singular.
function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M: number[][] = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
    }
    if (Math.abs(M[piv][i]) < 1e-12) return null;
    if (piv !== i) {
      const tmp = M[i];
      M[i] = M[piv];
      M[piv] = tmp;
    }
    for (let r = i + 1; r < n; r++) {
      const f = M[r][i] / M[i][i];
      if (f === 0) continue;
      for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

function forecastCase(c: CaseFile): { hours: { hour: number; q: number[] }[] } {
  const known1 = (c.known && c.known[0] && c.known[0].viewers) || 0;
  const p = Math.max(known1, 1);
  const lp = Math.log(p);
  const ch1 =
    c.known && c.known[0] && c.known[0].channels != null && c.known[0].channels > 0
      ? (c.known[0].channels as number)
      : null;

  // Precompute each comparable's day curve once.
  const curves: { comp: Comparable; dh: Map<number, number>; ch1: number | null }[] = [];
  for (const comp of c.comparables || []) {
    const dh = dayHours(comp);
    if (dh.get(1) != null) curves.push({ comp, dh, ch1: hour1Channels(comp) });
  }

  // The primary set is the console-only rolling-midnight launches.  In the
  // final case the four practice launches join it (they are all console
  // rolling-midnight single-player launches), taking it from 7 to 11.
  const useWider =
    curves.filter(({ comp, dh }) => isPrimary(comp, dh)).length < MIN_COMPARABLES;
  const select = useWider ? isWider : isPrimary;
  const use = curves.filter(({ comp, dh }) => select(comp, dh));

  const hours: { hour: number; q: number[] }[] = [];
  if (use.length < MIN_COMPARABLES) {
    // Too thin: flat at hour 1 with the baseline spread.
    for (let h = 2; h <= 24; h++) {
      const s = 0.15 * Math.sqrt(h - 1);
      hours.push({ hour: h, q: Z.map((z) => p * Math.exp(z * s)) });
    }
  } else {
    // Mean log viewers-per-channel over the training launches that have
    // hour-1 channels; the vpc term is centred on it.
    let vpcSum = 0;
    let vpcN = 0;
    for (const { dh, ch1: c1 } of use) {
      const h1 = dh.get(1) as number;
      if (c1 != null) {
        vpcSum += Math.log(h1 / c1);
        vpcN++;
      }
    }
    const mvpc = vpcN > 0 ? vpcSum / vpcN : 0;

    // One pooled fit over every (launch, hour) point.
    // Columns: s_2..s_24 (23), b0, b1, b2, c0  =>  27 parameters.
    const NPAR = 27;
    const A: number[][] = Array.from({ length: NPAR }, () => new Array<number>(NPAR).fill(0));
    const rhs = new Array<number>(NPAR).fill(0);
    type Pt = { h: number; x: number; v: number; y: number };
    const pts: Pt[] = [];
    for (const { dh, ch1: c1 } of use) {
      const h1 = dh.get(1) as number;
      const x = Math.log(h1);
      const v = c1 != null ? Math.log(h1 / c1) - mvpc : 0;
      for (let h = 2; h <= 24; h++) {
        const vh = dh.get(h);
        if (vh == null) continue;
        const y = Math.log(vh / h1);
        pts.push({ h, x, v, y });
        const u = uOf(h);
        const row = new Array<number>(NPAR).fill(0);
        row[h - 2] = 1;
        row[23] = x;
        row[24] = x * u;
        row[25] = x * u * u;
        row[26] = v;
        for (let i = 0; i < NPAR; i++) {
          if (row[i] === 0) continue;
          rhs[i] += row[i] * y;
          for (let j = 0; j < NPAR; j++) {
            if (row[j] !== 0) A[i][j] += row[i] * row[j];
          }
        }
      }
    }
    // Tiny ridge for numerical stability.
    for (let i = 0; i < NPAR; i++) A[i][i] += 1e-9;
    const coef = solveLinear(A, rhs);

    if (coef == null || pts.length < NPAR) {
      // Singular fit: flat at hour 1 with the baseline spread.
      for (let h = 2; h <= 24; h++) {
        const s = 0.15 * Math.sqrt(h - 1);
        hours.push({ hour: h, q: Z.map((z) => p * Math.exp(z * s)) });
      }
    } else {
      // The median curve is the pinball (least-absolute-deviation) fit:
      // it minimises the scored loss itself on the training launches,
      // where the least-squares median is pulled by the extreme ones.
      const medCoef = pinballMedianFit(pts, coef);
      const s = medCoef.slice(0, 23);
      const b0 = medCoef[23];
      const b1 = medCoef[24];
      const b2 = medCoef[25];
      const c0 = medCoef[26];
      // Pooled residual SD over every fitted point (least-squares fit,
      // as in v2 — the spread is unchanged).
      let ss = 0;
      for (const pt of pts) {
        const u = uOf(pt.h);
        const pred =
          coef[pt.h - 2] + (coef[23] + coef[24] * u + coef[25] * u * u) * pt.x + coef[26] * pt.v;
        const r = pt.y - pred;
        ss += r * r;
      }
      const sig = Math.max(Math.sqrt(ss / pts.length), 0.05) * SIGMA_MULT;
      // Target's centred log viewers-per-channel, clamped so an extreme
      // hour-1 channel count cannot swing the forecast wildly.
      let vt = 0;
      if (ch1 != null) {
        vt = Math.log(p / ch1) - mvpc;
        if (vt > VPC_CLAMP) vt = VPC_CLAMP;
        if (vt < -VPC_CLAMP) vt = -VPC_CLAMP;
      }
      for (let h = 2; h <= 24; h++) {
        const u = uOf(h);
        const med = s[h - 2] + (b0 + b1 * u + b2 * u * u) * lp + c0 * vt;
        hours.push({ hour: h, q: Z.map((z) => p * Math.exp(med + z * sig)) });
      }
    }
  }

  // Round to 2 decimals, clamp to (0, 1e9], keep non-decreasing.
  for (const row of hours) {
    const rounded: number[] = [];
    for (let i = 0; i < 5; i++) {
      let v = Math.round(row.q[i] * 100) / 100;
      if (!Number.isFinite(v) || v <= 0) v = 0.01;
      if (v > 1e9) v = 1e9;
      if (i > 0 && v < rounded[i - 1]) v = rounded[i - 1];
      rounded.push(v);
    }
    row.q = rounded;
  }
  return { hours };
}

function fallbackOutput(): { hours: { hour: number; q: number[] }[] } {
  const hours: { hour: number; q: number[] }[] = [];
  for (let h = 2; h <= 24; h++) {
    hours.push({ hour: h, q: [1, 1, 1, 1, 1] });
  }
  return { hours };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const id = file.slice(0, -3);
    const outPath = join(outDir, id + ".out");
    try {
      const parsed = JSON.parse(readFileSync(join(inDir, file), "utf8")) as CaseFile;
      const out = forecastCase(parsed);
      writeFileSync(outPath, JSON.stringify(out));
    } catch {
      // Never leave a case without a valid-shaped output.
      writeFileSync(outPath, JSON.stringify(fallbackOutput()));
    }
  }
}

main();
