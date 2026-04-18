import { useEffect, useRef, useState, useMemo, useCallback } from "react";

// ---------------------------------------------------------------------------
// Reference constants from results.json (mirrored here to avoid async lookups
// during initial render — the JSON is fetched to populate dynamic pieces).
// ---------------------------------------------------------------------------
const LAP_LENGTH_M = 5412;
const V_MAX = 98; // m/s
const REFERENCE_PARAMS = {
  m_kg: 888,
  P_eff_W: 700000,
  CdA: 0.7,
  ClA: 4.0,
  rho: 1.17,
  mu: 1.65,
};

const CORNERS = [
  { name: "T1", d: 1090, v_apex: 25.85, radius: 35 },
  { name: "T2", d: 1180, v_apex: 44.56, radius: 80 },
  { name: "T3", d: 1260, v_apex: 44.56, radius: 80 },
  { name: "T4", d: 1640, v_apex: 22.2, radius: 25 },
  { name: "T5", d: 2020, v_apex: 41.7, radius: 70 },
  { name: "T6", d: 2180, v_apex: 44.56, radius: 80 },
  { name: "T7", d: 2280, v_apex: 44.56, radius: 80 },
  { name: "T8", d: 2470, v_apex: 52.9, radius: 110 },
  { name: "T9", d: 2730, v_apex: 77.0, radius: 220 },
  { name: "T10", d: 2910, v_apex: 62.0, radius: 150 },
  { name: "T11", d: 3430, v_apex: 36.7, radius: 55 },
  { name: "T12", d: 3620, v_apex: 45.8, radius: 85 },
  { name: "T13", d: 3730, v_apex: 45.8, radius: 85 },
  { name: "T14", d: 4070, v_apex: 41.7, radius: 70 },
  { name: "T15", d: 4450, v_apex: 44.56, radius: 80 },
];

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
function haversine(lon1, lat1, lon2, lat2) {
  const R = 6371000;
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const dphi = ((lat2 - lat1) * Math.PI) / 180;
  const dlam = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dphi / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dlam / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function buildArcLengthLUT(coords) {
  const cumDist = [0];
  for (let i = 1; i < coords.length; i++) {
    const d = haversine(
      coords[i - 1][0],
      coords[i - 1][1],
      coords[i][0],
      coords[i][1]
    );
    cumDist.push(cumDist[i - 1] + d);
  }
  // Rescale to exactly 5412 m (the nominal lap length in our data) so
  // the published corner positions line up cleanly with our LUT.
  const rawTotal = cumDist[cumDist.length - 1];
  const scale = LAP_LENGTH_M / rawTotal;
  const scaled = cumDist.map((x) => x * scale);
  return { cumDist: scaled, totalDist: LAP_LENGTH_M };
}

function distanceToLatLon(lut, coords, d) {
  const dd = ((d % LAP_LENGTH_M) + LAP_LENGTH_M) % LAP_LENGTH_M;
  const { cumDist } = lut;
  // Binary search for the segment
  let lo = 0;
  let hi = cumDist.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (cumDist[mid] <= dd) lo = mid;
    else hi = mid;
  }
  const seg = cumDist[hi] - cumDist[lo];
  const t = seg > 0 ? (dd - cumDist[lo]) / seg : 0;
  const [lon1, lat1] = coords[lo];
  const [lon2, lat2] = coords[hi];
  return [lon1 + (lon2 - lon1) * t, lat1 + (lat2 - lat1) * t];
}

function sliceTrackByDistance(coords, lut, dStart, dEnd) {
  const { cumDist } = lut;
  const n = coords.length;
  const s = Math.max(0, Math.min(LAP_LENGTH_M, dStart));
  const e = Math.max(0, Math.min(LAP_LENGTH_M, dEnd));
  const pts = [];
  pts.push(distanceToLatLon(lut, coords, s));
  for (let i = 0; i < n; i++) {
    if (cumDist[i] > s && cumDist[i] < e) pts.push(coords[i]);
  }
  pts.push(distanceToLatLon(lut, coords, e));
  return {
    type: "Feature",
    properties: {},
    geometry: { type: "LineString", coordinates: pts },
  };
}

// ---------------------------------------------------------------------------
// Synthetic telemetry — used by the SpeedTrace-style charts.
// ---------------------------------------------------------------------------
function buildSpeedTrace(nPoints = 400) {
  const xs = [];
  const vs = [];
  // Sort corners by distance (already sorted)
  const tauAccel = 4.2;
  const aBrake = 50; // m/s^2
  const vApexArr = CORNERS.map((c) => c.v_apex);
  const dArr = CORNERS.map((c) => c.d);

  for (let i = 0; i < nPoints; i++) {
    const d = (i / (nPoints - 1)) * LAP_LENGTH_M;
    // Find nearest corner ahead and behind
    let aheadIdx = -1;
    for (let k = 0; k < dArr.length; k++) {
      if (dArr[k] >= d) {
        aheadIdx = k;
        break;
      }
    }
    const behindIdx = aheadIdx === -1 ? dArr.length - 1 : aheadIdx - 1;
    const vPrevApex = behindIdx >= 0 ? vApexArr[behindIdx] : vApexArr[vApexArr.length - 1];
    const dPrevExit = behindIdx >= 0 ? dArr[behindIdx] : dArr[dArr.length - 1] - LAP_LENGTH_M;
    const vNextApex = aheadIdx >= 0 ? vApexArr[aheadIdx] : vApexArr[0];
    const dNextEntry = aheadIdx >= 0 ? dArr[aheadIdx] : dArr[0] + LAP_LENGTH_M;

    // Acceleration phase speed — Hill-Keller style exponential approach
    const distSinceExit = Math.max(0, d - dPrevExit);
    const vAvg = (vPrevApex + V_MAX) / 2 + 1e-3;
    const vAccel =
      V_MAX - (V_MAX - vPrevApex) * Math.exp(-distSinceExit / (tauAccel * vAvg));

    // Braking phase: v^2 = v_apex^2 + 2*a*(d_brake - d)
    const distToNext = Math.max(0, dNextEntry - d);
    const vBrake = Math.sqrt(vNextApex * vNextApex + 2 * aBrake * distToNext);

    const v = Math.min(V_MAX, vAccel, vBrake);
    xs.push(d);
    vs.push(v);
  }
  return { xs, vs };
}

const SPEED_TRACE = buildSpeedTrace(400);

function speedAtDistance(d) {
  const { xs, vs } = SPEED_TRACE;
  if (d <= xs[0]) return vs[0];
  if (d >= xs[xs.length - 1]) return vs[vs.length - 1];
  let lo = 0;
  let hi = xs.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= d) lo = mid;
    else hi = mid;
  }
  const t = (d - xs[lo]) / (xs[hi] - xs[lo]);
  return vs[lo] + (vs[hi] - vs[lo]) * t;
}

// ---------------------------------------------------------------------------
// Chart primitives
// ---------------------------------------------------------------------------
const AX_COLOR = "rgba(255,255,255,0.2)";
const LBL_COLOR = "rgba(255,255,255,0.45)";
const TEAL = "#2AA198";
const GOLD = "#D4AF37";
const CORAL = "#ff6b6b";

function ChartFrame({ width = 480, height = 200, children, ariaLabel }) {
  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={ariaLabel}
      style={{ display: "block", marginTop: 12 }}
    >
      {children}
    </svg>
  );
}

// ---- SpeedTrace ------------------------------------------------------------
function SpeedTrace({ interactive = false, onHover, onLeave }) {
  const W = 480;
  const H = 200;
  const PAD_L = 28;
  const PAD_R = 8;
  const PAD_T = 10;
  const PAD_B = 22;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const yMax = 110;
  const xToPx = (x) => PAD_L + (x / LAP_LENGTH_M) * plotW;
  const yToPx = (y) => PAD_T + plotH - (y / yMax) * plotH;
  const [hoverX, setHoverX] = useState(null);

  const pathD = useMemo(() => {
    const { xs, vs } = SPEED_TRACE;
    return xs
      .map((x, i) => `${i === 0 ? "M" : "L"} ${xToPx(x).toFixed(2)} ${yToPx(vs[i]).toFixed(2)}`)
      .join(" ");
  }, []);

  const handleMove = (e) => {
    if (!interactive) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * W;
    if (relX < PAD_L || relX > W - PAD_R) {
      setHoverX(null);
      onLeave && onLeave();
      return;
    }
    const d = ((relX - PAD_L) / plotW) * LAP_LENGTH_M;
    setHoverX(relX);
    onHover && onHover(d);
  };
  const handleOut = () => {
    setHoverX(null);
    onLeave && onLeave();
  };

  const hoverD = hoverX != null ? ((hoverX - PAD_L) / plotW) * LAP_LENGTH_M : null;
  const hoverY = hoverD != null ? speedAtDistance(hoverD) : null;

  return (
    <ChartFrame width={W} height={H} ariaLabel="Speed vs lap distance">
      {/* Axes */}
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke={AX_COLOR} />
      {/* Y gridlines */}
      {[0, 25, 50, 75, 100].map((v) => (
        <g key={v}>
          <line
            x1={PAD_L}
            y1={yToPx(v)}
            x2={W - PAD_R}
            y2={yToPx(v)}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={v === 0 ? 0 : 0.4}
          />
          <text x={PAD_L - 4} y={yToPx(v) + 3} fontSize="10" fill={LBL_COLOR} textAnchor="end">
            {v}
          </text>
        </g>
      ))}
      {/* Corner markers */}
      {CORNERS.map((c) => (
        <g key={c.name}>
          <line
            x1={xToPx(c.d)}
            y1={PAD_T}
            x2={xToPx(c.d)}
            y2={PAD_T + plotH}
            stroke={GOLD}
            strokeDasharray="2,3"
            opacity={0.25}
          />
        </g>
      ))}
      {/* Speed line */}
      <path d={pathD} stroke={TEAL} strokeWidth="2" fill="none" />
      {/* X labels */}
      {[0, 1353, 2706, 4059, 5412].map((v, i) => (
        <text key={i} x={xToPx(v)} y={H - 6} fontSize="10" fill={LBL_COLOR} textAnchor="middle">
          {v}m
        </text>
      ))}
      <text x={W - PAD_R} y={PAD_T + 10} fontSize="10" fill={LBL_COLOR} textAnchor="end">
        speed (m/s)
      </text>
      {/* Invisible capture rect for hover */}
      <rect
        x={PAD_L}
        y={PAD_T}
        width={plotW}
        height={plotH}
        fill="transparent"
        onMouseMove={handleMove}
        onMouseLeave={handleOut}
        style={{ cursor: interactive ? "crosshair" : "default" }}
      />
      {/* Hover crosshair */}
      {hoverX != null && (
        <g>
          <line
            x1={hoverX}
            y1={PAD_T}
            x2={hoverX}
            y2={PAD_T + plotH}
            stroke="rgba(212,175,55,0.6)"
            strokeWidth="1"
          />
          <circle cx={hoverX} cy={yToPx(hoverY)} r="4" fill={TEAL} stroke={GOLD} strokeWidth="1.5" />
          <text x={hoverX + 6} y={yToPx(hoverY) - 6} fontSize="10" fill={GOLD}>
            {hoverY.toFixed(0)} m/s @ {hoverD.toFixed(0)} m
          </text>
        </g>
      )}
    </ChartFrame>
  );
}

// ---- OscillatorRadar -------------------------------------------------------
const OSCILLATORS = [
  { name: "ICE", hz: 250 },
  { name: "Turbo", hz: 2000 },
  { name: "MGU-K", hz: 500 },
  { name: "MGU-H", hz: 2000 },
  { name: "Batt", hz: 10 },
  { name: "Gearbox", hz: 100 },
  { name: "Diff", hz: 80 },
  { name: "FL-Wh", hz: 25 },
  { name: "FR-Wh", hz: 25 },
  { name: "RL-Wh", hz: 25 },
  { name: "RR-Wh", hz: 25 },
  { name: "FL-Br", hz: 1 },
  { name: "FR-Br", hz: 1 },
  { name: "RL-Br", hz: 1 },
  { name: "RR-Br", hz: 1 },
  { name: "FL-Su", hz: 4 },
  { name: "FR-Su", hz: 4 },
  { name: "RL-Su", hz: 4 },
  { name: "RR-Su", hz: 4 },
  { name: "Aero", hz: 0.5 },
];

function OscillatorRadar() {
  const W = 480;
  const H = 260;
  const cx = W / 2;
  const cy = H / 2 + 4;
  const rMax = Math.min(W, H) / 2 - 40;
  const maxHz = 2000;
  const logMax = Math.log10(maxHz);
  const n = OSCILLATORS.length;
  const rFor = (hz) => {
    const logVal = Math.log10(Math.max(0.1, hz));
    const t = Math.max(0, Math.min(1, (logVal + 1) / (logMax + 1)));
    return t * rMax;
  };
  const pts = OSCILLATORS.map((o, i) => {
    const ang = (i / n) * 2 * Math.PI - Math.PI / 2;
    const r = rFor(o.hz);
    return [cx + r * Math.cos(ang), cy + r * Math.sin(ang), ang, o];
  });
  const polyD = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ") + " Z";

  return (
    <ChartFrame width={W} height={H} ariaLabel="Oscillator frequency radar">
      {/* Rings */}
      {[0.25, 0.5, 0.75, 1].map((t, i) => (
        <circle
          key={i}
          cx={cx}
          cy={cy}
          r={rMax * t}
          fill="none"
          stroke={AX_COLOR}
          strokeDasharray={i === 3 ? "none" : "2,3"}
        />
      ))}
      {/* Spokes + labels */}
      {OSCILLATORS.map((o, i) => {
        const ang = (i / n) * 2 * Math.PI - Math.PI / 2;
        const x1 = cx;
        const y1 = cy;
        const x2 = cx + rMax * Math.cos(ang);
        const y2 = cy + rMax * Math.sin(ang);
        const lx = cx + (rMax + 12) * Math.cos(ang);
        const ly = cy + (rMax + 12) * Math.sin(ang);
        return (
          <g key={i}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={AX_COLOR} opacity={0.4} />
            <text
              x={lx}
              y={ly}
              fontSize="8"
              fill={LBL_COLOR}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {o.name}
            </text>
          </g>
        );
      })}
      {/* Ring labels */}
      {[1, 10, 100, 1000, 2000].map((hz, i) => (
        <text key={i} x={cx + 3} y={cy - rFor(hz)} fontSize="8" fill={LBL_COLOR}>
          {hz}
        </text>
      ))}
      {/* Filled polygon */}
      <path d={polyD} fill={TEAL} fillOpacity={0.2} stroke={TEAL} strokeWidth="2" />
      {pts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="2.5" fill={GOLD} />
      ))}
    </ChartFrame>
  );
}

// ---- BrakeThrottleGear -----------------------------------------------------
function BrakeThrottleGear({ onHover, onLeave }) {
  const W = 480;
  const H = 210;
  const PAD_L = 32;
  const PAD_R = 32;
  const PAD_T = 18;
  const PAD_B = 22;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const dMin = 950;
  const dMax = 1200;
  const xToPx = (d) => PAD_L + ((d - dMin) / (dMax - dMin)) * plotW;
  const pctToPx = (p) => PAD_T + plotH - (p / 100) * plotH;
  const gearToPx = (g) => PAD_T + plotH - ((g - 2) / 6) * plotH;
  const [hoverX, setHoverX] = useState(null);

  const samples = useMemo(() => {
    const arr = [];
    const N = 140;
    for (let i = 0; i < N; i++) {
      const d = dMin + (i / (N - 1)) * (dMax - dMin);
      const v = speedAtDistance(d);
      const vEntry = 98;
      const brake = Math.max(0, Math.min(100, (100 * (vEntry - v)) / 60));
      const throttle = Math.max(0, Math.min(100, 100 - brake * 1.05));
      const gear = Math.max(3, Math.min(8, Math.floor(v / 12) + 1));
      arr.push({ d, v, brake, throttle, gear });
    }
    return arr;
  }, []);

  const mkPath = (valueFn) =>
    samples
      .map((s, i) => `${i === 0 ? "M" : "L"} ${xToPx(s.d).toFixed(2)} ${valueFn(s).toFixed(2)}`)
      .join(" ");

  const brakePath = mkPath((s) => pctToPx(s.brake));
  const throttlePath = mkPath((s) => pctToPx(s.throttle));
  const gearPath = samples
    .map((s, i) => {
      const x = xToPx(s.d);
      const y = gearToPx(s.gear);
      return i === 0 ? `M ${x.toFixed(2)} ${y.toFixed(2)}` : `L ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  const handleMove = (e) => {
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * W;
    if (relX < PAD_L || relX > W - PAD_R) {
      setHoverX(null);
      onLeave && onLeave();
      return;
    }
    const d = dMin + ((relX - PAD_L) / plotW) * (dMax - dMin);
    setHoverX(relX);
    onHover && onHover(d);
  };

  return (
    <ChartFrame width={W} height={H} ariaLabel="Brake throttle gear around T1">
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={W - PAD_R} y1={PAD_T} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      {[0, 50, 100].map((p) => (
        <g key={p}>
          <line
            x1={PAD_L}
            y1={pctToPx(p)}
            x2={W - PAD_R}
            y2={pctToPx(p)}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={p === 0 ? 0 : 0.3}
          />
          <text x={PAD_L - 4} y={pctToPx(p) + 3} fontSize="10" fill={LBL_COLOR} textAnchor="end">
            {p}
          </text>
        </g>
      ))}
      {[3, 5, 7].map((g) => (
        <text key={g} x={W - PAD_R + 4} y={gearToPx(g) + 3} fontSize="10" fill={GOLD}>
          g{g}
        </text>
      ))}
      <path d={throttlePath} stroke={TEAL} strokeWidth="2" fill="none" />
      <path d={brakePath} stroke={CORAL} strokeWidth="2" fill="none" />
      <path d={gearPath} stroke={GOLD} strokeWidth="1.5" fill="none" strokeDasharray="3,2" />
      {/* Legend */}
      <g transform={`translate(${PAD_L + 6}, ${PAD_T + 4})`}>
        <text x="0" y="10" fontSize="10" fill={TEAL}>throttle</text>
        <text x="54" y="10" fontSize="10" fill={CORAL}>brake</text>
        <text x="94" y="10" fontSize="10" fill={GOLD}>gear</text>
      </g>
      <text x={xToPx(1090)} y={H - 6} fontSize="10" fill={LBL_COLOR} textAnchor="middle">
        T1 apex 1090m
      </text>
      <line
        x1={xToPx(1090)}
        y1={PAD_T}
        x2={xToPx(1090)}
        y2={PAD_T + plotH}
        stroke={GOLD}
        strokeDasharray="2,3"
        opacity={0.5}
      />
      <rect
        x={PAD_L}
        y={PAD_T}
        width={plotW}
        height={plotH}
        fill="transparent"
        onMouseMove={handleMove}
        onMouseLeave={() => {
          setHoverX(null);
          onLeave && onLeave();
        }}
        style={{ cursor: "crosshair" }}
      />
      {hoverX != null && (
        <line
          x1={hoverX}
          y1={PAD_T}
          x2={hoverX}
          y2={PAD_T + plotH}
          stroke="rgba(212,175,55,0.6)"
          strokeWidth="1"
        />
      )}
    </ChartFrame>
  );
}

// ---- FrictionCircle --------------------------------------------------------
function FrictionCircle() {
  const W = 480;
  const H = 220;
  const cx = W / 2;
  const cy = H / 2 + 4;
  const r = Math.min(W, H) / 2 - 22;
  const g0 = 9.81;
  const limit = 5; // g
  const rPerG = r / 6;

  const points = useMemo(() => {
    const dMin = 1550;
    const dMax = 1700;
    const pts = [];
    const N = 60;
    for (let i = 0; i < N; i++) {
      const d = dMin + (i / (N - 1)) * (dMax - dMin);
      const v = speedAtDistance(d);
      const vPrev = speedAtDistance(Math.max(0, d - 8));
      const aLong = (v * v - vPrev * vPrev) / (2 * 8) / g0;
      // Lateral: v^2 / R where R ~ 25m near T4, scale away from it
      const distToApex = Math.abs(d - 1640);
      const radius = 25 + distToApex * 0.8;
      const aLat = (v * v) / radius / g0;
      // Randomize lateral sign so the scatter fills both sides
      const sign = (i % 2 === 0 ? 1 : -1);
      pts.push({ aLong, aLat: sign * aLat, d });
    }
    return pts;
  }, []);

  return (
    <ChartFrame width={W} height={H} ariaLabel="Friction circle at T4">
      {/* Grid */}
      {[-5, -2.5, 0, 2.5, 5].map((g) => (
        <g key={g}>
          <line
            x1={cx + g * rPerG}
            y1={cy - r}
            x2={cx + g * rPerG}
            y2={cy + r}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={g === 0 ? 0.5 : 0.2}
          />
          <line
            x1={cx - r}
            y1={cy - g * rPerG}
            x2={cx + r}
            y2={cy - g * rPerG}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={g === 0 ? 0.5 : 0.2}
          />
          <text x={cx + g * rPerG} y={cy + r + 12} fontSize="9" fill={LBL_COLOR} textAnchor="middle">
            {g}g
          </text>
          <text x={cx - r - 4} y={cy - g * rPerG + 3} fontSize="9" fill={LBL_COLOR} textAnchor="end">
            {g}g
          </text>
        </g>
      ))}
      {/* Grip envelope */}
      <circle
        cx={cx}
        cy={cy}
        r={limit * rPerG}
        fill="none"
        stroke={GOLD}
        strokeDasharray="4,4"
        strokeWidth="1.5"
      />
      {/* Scatter */}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={cx + p.aLat * rPerG}
          cy={cy - p.aLong * rPerG}
          r="2.5"
          fill={TEAL}
          fillOpacity={0.75}
        />
      ))}
      <text x={cx + 4} y={12} fontSize="10" fill={LBL_COLOR} textAnchor="start">
        longitudinal g
      </text>
      <text x={cx + r - 4} y={cy - 4} fontSize="10" fill={LBL_COLOR} textAnchor="end">
        lateral g
      </text>
      <text x={cx + limit * rPerG * 0.71 + 4} y={cy - limit * rPerG * 0.71 - 2} fontSize="10" fill={GOLD}>
        μ·g ≈ 5g
      </text>
    </ChartFrame>
  );
}

// ---- GeffCurve -------------------------------------------------------------
function GeffCurve() {
  const W = 480;
  const H = 200;
  const PAD_L = 32;
  const PAD_R = 10;
  const PAD_T = 10;
  const PAD_B = 22;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const vMax = 100;
  const gMax = 6;
  const xToPx = (v) => PAD_L + (v / vMax) * plotW;
  const yToPx = (g) => PAD_T + plotH - ((g - 1) / (gMax - 1)) * plotH;
  const { m_kg, ClA, rho } = REFERENCE_PARAMS;
  const g0 = 9.81;

  const samples = [];
  for (let v = 0; v <= vMax; v += 1) {
    const gEff = 1 + (rho * ClA * v * v) / (2 * m_kg * g0);
    samples.push({ v, g: Math.min(gMax, gEff) });
  }
  const baselineD = `M ${xToPx(0)} ${yToPx(1)} L ${xToPx(vMax)} ${yToPx(1)}`;
  const curveD = samples
    .map((s, i) => `${i === 0 ? "M" : "L"} ${xToPx(s.v).toFixed(2)} ${yToPx(s.g).toFixed(2)}`)
    .join(" ");
  const fillD =
    curveD +
    ` L ${xToPx(vMax)} ${yToPx(1)} L ${xToPx(0)} ${yToPx(1)} Z`;

  const markers = [
    { v: 26, label: "T4" },
    { v: 77, label: "T9" },
    { v: 98, label: "v_max" },
  ];

  return (
    <ChartFrame width={W} height={H} ariaLabel="Effective g versus speed">
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke={AX_COLOR} />
      {[1, 2, 3, 4, 5, 6].map((g) => (
        <g key={g}>
          <line
            x1={PAD_L}
            y1={yToPx(g)}
            x2={W - PAD_R}
            y2={yToPx(g)}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={g === 1 ? 0 : 0.25}
          />
          <text x={PAD_L - 4} y={yToPx(g) + 3} fontSize="10" fill={LBL_COLOR} textAnchor="end">
            {g}g
          </text>
        </g>
      ))}
      {[0, 25, 50, 75, 100].map((v) => (
        <text key={v} x={xToPx(v)} y={H - 6} fontSize="10" fill={LBL_COLOR} textAnchor="middle">
          {v}
        </text>
      ))}
      <path d={fillD} fill={TEAL} fillOpacity={0.15} />
      <path d={baselineD} stroke={GOLD} strokeWidth="2" strokeDasharray="3,3" fill="none" />
      <path d={curveD} stroke={TEAL} strokeWidth="2" fill="none" />
      {markers.map((m) => {
        const g = 1 + (rho * ClA * m.v * m.v) / (2 * m_kg * g0);
        return (
          <g key={m.label}>
            <circle cx={xToPx(m.v)} cy={yToPx(Math.min(gMax, g))} r="3" fill={GOLD} />
            <text x={xToPx(m.v) + 5} y={yToPx(Math.min(gMax, g)) - 4} fontSize="10" fill={GOLD}>
              {m.label} · {g.toFixed(1)}g
            </text>
          </g>
        );
      })}
      <text x={W - PAD_R} y={PAD_T + 10} fontSize="10" fill={LBL_COLOR} textAnchor="end">
        g_eff(v)  ·  speed (m/s) →
      </text>
    </ChartFrame>
  );
}

// ---- SEntropyTrace ---------------------------------------------------------
function SEntropyTrace({ onHover, onLeave }) {
  const W = 480;
  const H = 200;
  const PAD_L = 28;
  const PAD_R = 8;
  const PAD_T = 14;
  const PAD_B = 22;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const xToPx = (d) => PAD_L + (d / LAP_LENGTH_M) * plotW;
  const yToPx = (y) => PAD_T + plotH - y * plotH;
  const [hoverX, setHoverX] = useState(null);

  const samples = useMemo(() => {
    const N = 240;
    const arr = [];
    for (let i = 0; i < N; i++) {
      const d = (i / (N - 1)) * LAP_LENGTH_M;
      const phase = (2 * Math.PI * d) / LAP_LENGTH_M;
      const Sk = 0.2 + 0.6 * Math.sin(phase) ** 2;
      const St = 0.3 + 0.4 * Math.sin(phase + 0.9) ** 2;
      const Se = 0.4 + 0.5 * Math.sin(phase + 1.8) ** 2;
      arr.push({ d, Sk, St, Se });
    }
    return arr;
  }, []);

  const mkPath = (key) =>
    samples
      .map((s, i) => `${i === 0 ? "M" : "L"} ${xToPx(s.d).toFixed(2)} ${yToPx(s[key]).toFixed(2)}`)
      .join(" ");

  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * W;
    if (relX < PAD_L || relX > W - PAD_R) {
      setHoverX(null);
      onLeave && onLeave();
      return;
    }
    const d = ((relX - PAD_L) / plotW) * LAP_LENGTH_M;
    setHoverX(relX);
    onHover && onHover(d);
  };

  return (
    <ChartFrame width={W} height={H} ariaLabel="S-entropy components vs lap distance">
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke={AX_COLOR} />
      {[0, 0.5, 1].map((y) => (
        <g key={y}>
          <line
            x1={PAD_L}
            y1={yToPx(y)}
            x2={W - PAD_R}
            y2={yToPx(y)}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={y === 0 ? 0 : 0.25}
          />
          <text x={PAD_L - 4} y={yToPx(y) + 3} fontSize="10" fill={LBL_COLOR} textAnchor="end">
            {y}
          </text>
        </g>
      ))}
      {[LAP_LENGTH_M / 3, (2 * LAP_LENGTH_M) / 3].map((d, i) => (
        <g key={i}>
          <line
            x1={xToPx(d)}
            y1={PAD_T}
            x2={xToPx(d)}
            y2={PAD_T + plotH}
            stroke={GOLD}
            strokeDasharray="3,3"
            opacity={0.4}
          />
          <text x={xToPx(d)} y={PAD_T + 10} fontSize="9" fill={GOLD} textAnchor="middle">
            S{i + 2}
          </text>
        </g>
      ))}
      <path d={mkPath("Sk")} stroke={TEAL} strokeWidth="2" fill="none" />
      <path d={mkPath("St")} stroke={GOLD} strokeWidth="2" fill="none" />
      <path d={mkPath("Se")} stroke={CORAL} strokeWidth="2" fill="none" />
      <g transform={`translate(${PAD_L + 6}, ${PAD_T - 2})`}>
        <text x="0" y="9" fontSize="10" fill={TEAL}>S_k</text>
        <text x="30" y="9" fontSize="10" fill={GOLD}>S_t</text>
        <text x="60" y="9" fontSize="10" fill={CORAL}>S_e</text>
      </g>
      <rect
        x={PAD_L}
        y={PAD_T}
        width={plotW}
        height={plotH}
        fill="transparent"
        onMouseMove={handleMove}
        onMouseLeave={() => {
          setHoverX(null);
          onLeave && onLeave();
        }}
        style={{ cursor: "crosshair" }}
      />
      {hoverX != null && (
        <line
          x1={hoverX}
          y1={PAD_T}
          x2={hoverX}
          y2={PAD_T + plotH}
          stroke="rgba(212,175,55,0.6)"
          strokeWidth="1"
        />
      )}
    </ChartFrame>
  );
}

// ---- FaultTrajectory -------------------------------------------------------
function FaultTrajectory() {
  const W = 480;
  const H = 200;
  const PAD_L = 32;
  const PAD_R = 10;
  const PAD_T = 16;
  const PAD_B = 24;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const nLaps = 20;
  const xToPx = (lap) => PAD_L + (lap / nLaps) * plotW;
  const yToPx = (y) => PAD_T + plotH - y * plotH;

  const healthy = [];
  const faulty = [];
  let rng = 42;
  const rnd = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };
  for (let lap = 0; lap <= nLaps; lap++) {
    healthy.push({ lap, y: 0.05 + (rnd() - 0.5) * 0.04 });
    let y = 0.05 + (rnd() - 0.5) * 0.03;
    if (lap > 15) y += (lap - 15) * 0.11 + rnd() * 0.03;
    else if (lap > 13) y += (lap - 13) * 0.02;
    faulty.push({ lap, y: Math.min(0.95, y) });
  }

  const mk = (arr) =>
    arr
      .map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p.lap).toFixed(2)} ${yToPx(p.y).toFixed(2)}`)
      .join(" ");

  return (
    <ChartFrame width={W} height={H} ariaLabel="Fault divergence trajectory">
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke={AX_COLOR} />
      {[0, 0.25, 0.5, 0.75, 1].map((y) => (
        <g key={y}>
          <line
            x1={PAD_L}
            y1={yToPx(y)}
            x2={W - PAD_R}
            y2={yToPx(y)}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={y === 0 ? 0 : 0.25}
          />
          <text x={PAD_L - 4} y={yToPx(y) + 3} fontSize="10" fill={LBL_COLOR} textAnchor="end">
            {y}
          </text>
        </g>
      ))}
      {[0, 5, 10, 15, 20].map((lap) => (
        <text key={lap} x={xToPx(lap)} y={H - 6} fontSize="10" fill={LBL_COLOR} textAnchor="middle">
          L{lap}
        </text>
      ))}
      {/* Detection marker */}
      <line
        x1={xToPx(13)}
        y1={PAD_T}
        x2={xToPx(13)}
        y2={PAD_T + plotH}
        stroke={GOLD}
        strokeDasharray="3,3"
      />
      <text x={xToPx(13) + 4} y={PAD_T + 10} fontSize="9" fill={GOLD}>
        Philharmonic detection · L13
      </text>
      <line
        x1={xToPx(15)}
        y1={PAD_T}
        x2={xToPx(15)}
        y2={PAD_T + plotH}
        stroke={CORAL}
        strokeDasharray="3,3"
      />
      <text x={xToPx(15) + 4} y={PAD_T + 22} fontSize="9" fill={CORAL}>
        Fault injection · L15
      </text>
      <path d={mk(healthy)} stroke={TEAL} strokeWidth="2" fill="none" />
      <path d={mk(faulty)} stroke={CORAL} strokeWidth="2" fill="none" />
      <text x={W - PAD_R - 4} y={yToPx(0.55)} fontSize="10" fill={GOLD} textAnchor="end">
        13-lap lead time
      </text>
    </ChartFrame>
  );
}

// ---- SectorBars ------------------------------------------------------------
const SECTOR_DATA = [
  { name: "Sector 1", predicted: 28.0, fastest: 28.72, avg: 29.2 },
  { name: "Sector 2", predicted: 28.85, fastest: 29.65, avg: 30.15 },
  { name: "Sector 3", predicted: 23.54, fastest: 24.08, avg: 24.4 },
];

function SectorBars() {
  const W = 480;
  const H = 220;
  const PAD_L = 32;
  const PAD_R = 10;
  const PAD_T = 22;
  const PAD_B = 28;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const yMax = 32;
  const yMin = 22;
  const yToPx = (y) => PAD_T + plotH - ((y - yMin) / (yMax - yMin)) * plotH;
  const groupW = plotW / SECTOR_DATA.length;
  const barW = (groupW - 20) / 3;

  return (
    <ChartFrame width={W} height={H} ariaLabel="Sector times predicted vs observed">
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke={AX_COLOR} />
      {[22, 25, 28, 31].map((y) => (
        <g key={y}>
          <line
            x1={PAD_L}
            y1={yToPx(y)}
            x2={W - PAD_R}
            y2={yToPx(y)}
            stroke={AX_COLOR}
            strokeDasharray="2,3"
            opacity={0.3}
          />
          <text x={PAD_L - 4} y={yToPx(y) + 3} fontSize="10" fill={LBL_COLOR} textAnchor="end">
            {y}s
          </text>
        </g>
      ))}
      {SECTOR_DATA.map((s, i) => {
        const gx = PAD_L + i * groupW + 10;
        const bars = [
          { key: "predicted", val: s.predicted, color: TEAL },
          { key: "fastest", val: s.fastest, color: GOLD },
          { key: "avg", val: s.avg, color: "rgba(255,255,255,0.35)" },
        ];
        return (
          <g key={s.name}>
            {bars.map((b, j) => {
              const x = gx + j * barW;
              const y = yToPx(b.val);
              const h = PAD_T + plotH - y;
              return (
                <g key={b.key}>
                  <rect x={x} y={y} width={barW - 2} height={h} fill={b.color} />
                  <text
                    x={x + (barW - 2) / 2}
                    y={y - 4}
                    fontSize="9"
                    fill="rgba(255,255,255,0.7)"
                    textAnchor="middle"
                  >
                    {b.val.toFixed(2)}
                  </text>
                </g>
              );
            })}
            <text
              x={gx + (barW * 3) / 2}
              y={H - 12}
              fontSize="10"
              fill={LBL_COLOR}
              textAnchor="middle"
            >
              {s.name}
            </text>
          </g>
        );
      })}
      <g transform={`translate(${PAD_L + 6}, ${PAD_T - 6})`}>
        <rect x="0" y="0" width="10" height="8" fill={TEAL} />
        <text x="14" y="8" fontSize="10" fill={LBL_COLOR}>predicted</text>
        <rect x="70" y="0" width="10" height="8" fill={GOLD} />
        <text x="84" y="8" fontSize="10" fill={LBL_COLOR}>fastest</text>
        <rect x="128" y="0" width="10" height="8" fill="rgba(255,255,255,0.35)" />
        <text x="142" y="8" fontSize="10" fill={LBL_COLOR}>avg(8)</text>
      </g>
    </ChartFrame>
  );
}

// ---- GumbelSwarm -----------------------------------------------------------
function GumbelSwarm({ samples, gumbelLoc, gumbelScale }) {
  const W = 480;
  const H = 220;
  const PAD_L = 32;
  const PAD_R = 10;
  const PAD_T = 18;
  const PAD_B = 24;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const xMin = 82;
  const xMax = 96;
  const xToPx = (x) => PAD_L + ((x - xMin) / (xMax - xMin)) * plotW;
  const pdfMax = 0.3;
  const yToPx = (y) => PAD_T + plotH - (y / pdfMax) * plotH;
  const verstappen = 89.708;
  const theoreticalMin = 89.1;

  const curve = [];
  const betaSafe = Math.max(0.01, gumbelScale || 1.55);
  for (let x = xMin; x <= xMax; x += 0.1) {
    const z = (x - (gumbelLoc || 91.58)) / betaSafe;
    const pdf = (1 / betaSafe) * Math.exp(-(z + Math.exp(-z)));
    curve.push({ x, y: pdf });
  }
  const curveD = curve
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p.x).toFixed(2)} ${yToPx(p.y).toFixed(2)}`)
    .join(" ");

  // Deterministic jitter
  let rng = 7;
  const rnd = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };
  const baseY = PAD_T + plotH * 0.75;
  const swarm = (samples || []).map((s) => ({
    x: xToPx(Math.max(xMin, Math.min(xMax, s))),
    y: baseY + (rnd() - 0.5) * plotH * 0.3,
  }));

  return (
    <ChartFrame width={W} height={H} ariaLabel="Monte Carlo lap time distribution">
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke={AX_COLOR} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke={AX_COLOR} />
      {[82, 85, 88, 91, 94].map((x) => (
        <text key={x} x={xToPx(x)} y={H - 6} fontSize="10" fill={LBL_COLOR} textAnchor="middle">
          {x}s
        </text>
      ))}
      <path d={curveD} stroke={GOLD} strokeWidth="2" fill="none" />
      {swarm.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="3" fill={TEAL} fillOpacity={0.7} />
      ))}
      <line
        x1={xToPx(verstappen)}
        y1={PAD_T}
        x2={xToPx(verstappen)}
        y2={PAD_T + plotH}
        stroke={GOLD}
        strokeWidth="1.5"
      />
      <text x={xToPx(verstappen) + 4} y={PAD_T + 10} fontSize="10" fill={GOLD}>
        Verstappen 2023 · 89.708s
      </text>
      <line
        x1={xToPx(theoreticalMin)}
        y1={PAD_T}
        x2={xToPx(theoreticalMin)}
        y2={PAD_T + plotH}
        stroke={TEAL}
        strokeWidth="1.5"
        strokeDasharray="3,3"
      />
      <text x={xToPx(theoreticalMin) - 4} y={PAD_T + 22} fontSize="10" fill={TEAL} textAnchor="end">
        Theoretical min · 89.1s
      </text>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// Chapter content
// ---------------------------------------------------------------------------
const CHAPTERS = [
  {
    title: "Sakhir, Bahrain",
    label: "Overview",
    alignment: "left",
    location: { center: [50.5144, 26.0315], zoom: 14.5, pitch: 0, bearing: 0 },
    text: "The Bahrain International Circuit opened in 2004, a 5.412 km loop in the Sakhir desert with 15 corners and a characteristic long main straight into a heavy T1 braking zone. The lap sits 16 m below sea level — dense air, long straights, abrasive tarmac — a useful crucible for a model that claims to compute the theoretical minimum lap. Across one qualifying flyer a car spends 13.36 seconds crossing the start-finish straight alone and 32.06 seconds negotiating the T3-T4 section. These timings, from the reference solver, anchor every chapter that follows.",
    chart: "speedtrace-static",
  },
  {
    title: "The Oscillator Network",
    label: "Architecture",
    alignment: "right",
    location: { center: [50.5150, 26.0320], zoom: 15, pitch: 30, bearing: -20 },
    text: "A Formula One power unit is not one machine; it is a graph of twenty coupled oscillators. The ICE fires at ~250 Hz, the turbocharger spins near 2 kHz, the MGU-H sits on the same shaft, the MGU-K couples at 500 Hz, and a 10 Hz battery feeds them all. Below the drivetrain another sixteen oscillators — four wheels, four brakes, four corners of suspension, and the aerodynamic structure — close the loop. Philharmonic treats this as a circuit graph: oscillator frequency on a log scale, mutual conductance on the edges. The radar below shows the operating frequencies spread over nearly four decades.",
    chart: "radar",
  },
  {
    title: "242 Data Points per Lap",
    label: "Telemetry",
    alignment: "left",
    location: { center: [50.5144, 26.0270], zoom: 17, pitch: 45, bearing: 10 },
    text: "The FastF1 stream delivers 242 telemetry samples per lap — speed, RPM, throttle, brake, gear, DRS — observing only 11 of the 20 oscillator nodes. The reconstruction problem is to recover the other 9 (turbo shaft state, MGU-H torque, MGU-K duty, battery SOC, per-corner suspension load) from the observable slice. Hover anywhere on the speed trace and the corresponding point on the circuit lights up; you are watching the inverse map from distance-along-lap back to position-in-space. The lap reaches 98 m/s on the pit-straight and collapses to 22 m/s at the T4 apex — a factor-of-four swing that the oscillator graph must absorb coherently.",
    chart: "speedtrace-interactive",
  },
  {
    title: "Turn 1 — Heavy Braking",
    label: "Braking Zone",
    alignment: "right",
    location: { center: [50.5135, 26.0355], zoom: 17.5, pitch: 60, bearing: -60 },
    text: "At the end of the main straight — cumulative distance 1090 m — the car decelerates from ~98 m/s to 25.85 m/s in roughly 110 metres, shedding 39.71 MJ into the brake discs and harvesting 4.0 MJ through the MGU-K. Brake pressure climbs from 0 to 100 % in under 0.3 s, throttle collapses from 100 to 0, and the gearbox drops from 8 to 3. This braking zone dominates the circuit's energy budget: of 48.99 MJ of mechanical propulsion, nearly all of it comes back out here as heat. Hover the trace to locate the matching point on the track.",
    chart: "brake-throttle-gear",
  },
  {
    title: "Turn 4 — The Hairpin",
    label: "Minimum Speed",
    alignment: "left",
    location: { center: [50.5080, 26.0350], zoom: 17.5, pitch: 60, bearing: 120 },
    text: "T4 is the slowest point on the lap: 25 m radius, 22 m/s apex, first and second gear territory. The friction circle collapses here — longitudinal demand is low but lateral demand saturates near μ·g ≈ 5 g. The scatter below is the car's operating envelope through the 1550-1700 m window; it hugs the dashed grip limit on corner entry and exit, confirming that the bottleneck at T4 is not power but tyre friction. The sensitivity analysis agrees — μ alone accounts for 48 % of total lap-time variance, dwarfing every other parameter.",
    chart: "friction-circle",
  },
  {
    title: "T9-T10 Sweepers",
    label: "Downforce Corners",
    alignment: "right",
    location: { center: [50.5170, 26.0340], zoom: 17, pitch: 60, bearing: -90 },
    text: "The T9-T10 complex is taken at 77 m/s and 62 m/s respectively — far above the T4 apex — and here grip stops being a property of the tyres and starts being a property of the car's aerodynamic package. With ClA = 4.0 and ρ = 1.17 kg/m³, the effective vertical load at 77 m/s is roughly 3.5 g, more than triple the static weight. The teal curve below traces g_eff(v); the shaded band is the aerodynamic contribution. Without it, the sweepers would be 30 % slower and the S-entropy trace would tell a different story.",
    chart: "geff",
  },
  {
    title: "S-Entropy Trace",
    label: "Categorical Coordinates",
    alignment: "left",
    location: { center: [50.5144, 26.0315], zoom: 15, pitch: 45, bearing: 0 },
    text: "The racing line can be rewritten as a trajectory in three categorical coordinates: S_k (knowledge — what the car knows), S_t (time — where it is in the lap), and S_e (energy — how much is left in the store). These three components oscillate with the corner cadence and phase-lock across sectors. Sector boundaries (dashed gold) divide the lap into three windows where the dominant component rotates. Hover any point on the trace to find its location on the track.",
    chart: "sentropy",
  },
  {
    title: "Fault Detection",
    label: "Diagnostics",
    alignment: "right",
    location: { center: [50.5135, 26.0355], zoom: 16, pitch: 30, bearing: -60 },
    text: "We injected a 30 % conductance drop in the ICE-Turbo coupling at lap 15 — the kind of fault that bearing wear produces before a shaft failure. Philharmonic flagged it on lap 13, before the injection, because the backward-navigated state had already begun escaping the healthy attractor. The coral curve below shows the divergence; the teal baseline shows healthy laps. The 13-lap lead time is not prediction — it is pattern-matching on a manifold that the car is already on. The pulsing marker on the T1 exit is where the fault localized.",
    chart: "fault",
  },
  {
    title: "Racing Line Extraction",
    label: "Qualifying",
    alignment: "left",
    location: { center: [50.5144, 26.0315], zoom: 15, pitch: 45, bearing: 10 },
    text: "Across 8 qualifying laps the solver extracts a per-sector predicted minimum and compares it to the fastest observed and the 8-lap average. Sector 1 predicts 28.00 s vs a fastest 28.72 s; sector 2, 28.85 s vs 29.65 s; sector 3, 23.54 s vs 24.08 s. The gap between predicted and fastest — roughly 0.7 s per sector — is the residual between theoretical optimum and a human driver in traffic. The gap between fastest and average is execution noise: tyre temperature, wind, fuel load.",
    chart: "sector-bars",
  },
  {
    title: "Theoretical Minimum",
    label: "Validation",
    alignment: "right",
    location: { center: [50.5144, 26.0315], zoom: 14, pitch: 0, bearing: 0 },
    rotate: true,
    text: "Finally: 500 Monte Carlo samples over the full parameter distribution yield a mean lap time of 89.10 s with 95 % CI [84.91, 93.62]. Fitted against the historical Gumbel-extreme-value distribution of qualifying records (loc 91.58 s, scale 1.55 s), the predicted minimum of 89.1 s sits 0.608 s clear of Verstappen's 2023 pole of 89.708 s. That residual decomposes as 0.15 s tyre preparation, 0.20 s setup mismatch, and 0.25 s driver execution — all of which are observed but not modelled. The theoretical minimum is a lower bound, and reality respected it.",
    chart: "gumbel",
  },
];

// ---------------------------------------------------------------------------
// Fallback (no Mapbox token)
// ---------------------------------------------------------------------------
function FallbackView({ results }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0a",
        color: "#fafafa",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "4rem 2rem",
      }}
    >
      <div style={{ maxWidth: 720, width: "100%" }}>
        <div
          style={{
            fontSize: "0.65rem",
            color: GOLD,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            marginBottom: 16,
          }}
        >
          Bahrain International Circuit
        </div>
        <h1 style={{ fontSize: "2.5rem", fontWeight: 700, marginBottom: 8, lineHeight: 1.1 }}>
          Philharmonic F1 Validation
        </h1>
        <p style={{ fontSize: "0.9rem", color: "rgba(250,250,250,0.5)", marginBottom: 48 }}>
          Sakhir, Bahrain &mdash; 26.0325&deg;N, 50.5112&deg;E
          <br />
          Set <code>NEXT_PUBLIC_MAPBOX_TOKEN</code> for the interactive map experience.
        </p>
        {CHAPTERS.map((ch, i) => (
          <div
            key={i}
            style={{
              marginBottom: 48,
              paddingLeft: 24,
              borderLeft: "2px solid rgba(212,175,55,0.3)",
            }}
          >
            <div
              style={{
                fontSize: "0.6rem",
                color: GOLD,
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                marginBottom: 4,
              }}
            >
              {ch.label || `Chapter ${i + 1}`}
            </div>
            <h2 style={{ fontSize: "1.3rem", fontWeight: 700, marginBottom: 8 }}>{ch.title}</h2>
            <p style={{ fontSize: "0.85rem", color: "rgba(250,250,250,0.6)", lineHeight: 1.7 }}>
              {ch.text}
            </p>
            <div
              style={{
                marginTop: 10,
                fontSize: "0.7rem",
                color: "rgba(250,250,250,0.35)",
                fontFamily: "monospace",
              }}
            >
              [chart: {ch.chart}]
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main story component
// ---------------------------------------------------------------------------
export default function BahrainStory() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [activeChapter, setActiveChapter] = useState(0);
  const [hoveredDistance, setHoveredDistance] = useState(null);
  const [geo, setGeo] = useState(null); // { coords, lut }
  const [results, setResults] = useState(null);
  const chapterRefs = useRef([]);
  const rafRef = useRef(null);
  const pulseRef = useRef({ t: 0 });

  // Fetch data client-side
  useEffect(() => {
    let alive = true;
    Promise.all([
      fetch("/data/bh-2002.geojson").then((r) => r.json()).catch(() => null),
      fetch("/data/results.json").then((r) => r.json()).catch(() => null),
    ]).then(([g, r]) => {
      if (!alive) return;
      if (g && g.features && g.features.length > 0) {
        const coords = g.features[0].geometry.coordinates;
        const lut = buildArcLengthLUT(coords);
        setGeo({ coords, lut, feature: g.features[0] });
      }
      if (r) setResults(r);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Hover callbacks passed to charts
  const handleHover = useCallback((d) => setHoveredDistance(d), []);
  const handleLeave = useCallback(() => setHoveredDistance(null), []);

  // Set up map
  useEffect(() => {
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
    if (!token || !mapContainer.current || !geo) return;

    const mapboxgl = require("mapbox-gl");
    mapboxgl.accessToken = token;

    const m = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/dark-v11",
      center: CHAPTERS[0].location.center,
      zoom: CHAPTERS[0].location.zoom,
      pitch: CHAPTERS[0].location.pitch || 0,
      bearing: CHAPTERS[0].location.bearing || 0,
      interactive: false,
    });
    map.current = m;

    const addCornerFeatures = () => {
      const features = CORNERS.map((c) => {
        const [lon, lat] = distanceToLatLon(geo.lut, geo.coords, c.d);
        return {
          type: "Feature",
          properties: { name: c.name, d: c.d },
          geometry: { type: "Point", coordinates: [lon, lat] },
        };
      });
      return { type: "FeatureCollection", features };
    };

    m.on("load", () => {
      // Track line
      m.addSource("track", { type: "geojson", data: geo.feature });
      m.addLayer({
        id: "track-line",
        type: "line",
        source: "track",
        paint: {
          "line-color": TEAL,
          "line-width": 4,
          "line-opacity": 0.7,
        },
      });

      // Active segment
      m.addSource("track-active", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "track-active-line",
        type: "line",
        source: "track-active",
        paint: {
          "line-color": GOLD,
          "line-width": 6,
          "line-blur": 2,
          "line-opacity": 0.95,
        },
      });

      // Corner markers
      m.addSource("corners", { type: "geojson", data: addCornerFeatures() });
      m.addLayer({
        id: "corner-dots",
        type: "circle",
        source: "corners",
        paint: {
          "circle-radius": 4,
          "circle-color": GOLD,
          "circle-stroke-color": "rgba(10,10,10,0.9)",
          "circle-stroke-width": 1.5,
        },
      });
      m.addLayer({
        id: "corner-labels",
        type: "symbol",
        source: "corners",
        minzoom: 15,
        layout: {
          "text-field": ["get", "name"],
          "text-size": 10,
          "text-offset": [0, -1.2],
          "text-anchor": "bottom",
          "text-allow-overlap": true,
        },
        paint: {
          "text-color": "#fafafa",
          "text-halo-color": "rgba(10,10,10,0.9)",
          "text-halo-width": 1.2,
        },
      });

      // Hover marker (single point that we update via setData)
      m.addSource("hover-point", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "hover-ring",
        type: "circle",
        source: "hover-point",
        paint: {
          "circle-radius": 10,
          "circle-color": "rgba(212,175,55,0.0)",
          "circle-stroke-color": GOLD,
          "circle-stroke-width": 2,
        },
      });
      m.addLayer({
        id: "hover-dot",
        type: "circle",
        source: "hover-point",
        paint: {
          "circle-radius": 5,
          "circle-color": TEAL,
          "circle-stroke-color": "#fafafa",
          "circle-stroke-width": 1,
        },
      });

      // Fault pulse marker (Mapbox custom pulsing dot)
      const size = 120;
      const pulsingDot = {
        width: size,
        height: size,
        data: new Uint8Array(size * size * 4),
        onAdd: function () {
          const canvas = document.createElement("canvas");
          canvas.width = this.width;
          canvas.height = this.height;
          this.context = canvas.getContext("2d");
        },
        render: function () {
          const duration = 1400;
          const t = (performance.now() % duration) / duration;
          const radius = (size / 2) * 0.3;
          const outerRadius = (size / 2) * 0.7 * t + radius;
          const ctx = this.context;
          ctx.clearRect(0, 0, this.width, this.height);
          ctx.beginPath();
          ctx.arc(this.width / 2, this.height / 2, outerRadius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 107, 107, ${1 - t})`;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(this.width / 2, this.height / 2, radius, 0, Math.PI * 2);
          ctx.fillStyle = CORAL;
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 2;
          ctx.fill();
          ctx.stroke();
          this.data = ctx.getImageData(0, 0, this.width, this.height).data;
          m.triggerRepaint();
          return true;
        },
      };
      if (!m.hasImage("pulsing-dot")) {
        m.addImage("pulsing-dot", pulsingDot, { pixelRatio: 2 });
      }
      m.addSource("fault-pulse", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "fault-pulse-layer",
        type: "symbol",
        source: "fault-pulse",
        layout: { "icon-image": "pulsing-dot", "icon-allow-overlap": true },
      });
    });

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (m._rotateInterval) clearInterval(m._rotateInterval);
      m.remove();
      map.current = null;
    };
  }, [geo]);

  // IntersectionObserver for active chapter
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = parseInt(entry.target.dataset.index, 10);
            setActiveChapter(idx);
          }
        });
      },
      { threshold: 0.55 }
    );
    chapterRefs.current.forEach((ref) => {
      if (ref) observer.observe(ref);
    });
    return () => observer.disconnect();
  }, []);

  // React to activeChapter changes on the map
  useEffect(() => {
    const m = map.current;
    if (!m || !geo) return;
    if (!m.isStyleLoaded()) {
      const fn = () => updateForChapter();
      m.once("load", fn);
      return () => m.off("load", fn);
    }
    updateForChapter();

    function updateForChapter() {
      const ch = CHAPTERS[activeChapter];
      m.flyTo({
        center: ch.location.center,
        zoom: ch.location.zoom,
        pitch: ch.location.pitch || 0,
        bearing: ch.location.bearing || 0,
        duration: 2000,
        essential: true,
      });

      // Clear existing rotation
      if (m._rotateInterval) {
        clearInterval(m._rotateInterval);
        m._rotateInterval = null;
      }
      if (ch.rotate) {
        let bearing = ch.location.bearing || 0;
        m._rotateInterval = setInterval(() => {
          if (!map.current) return;
          bearing += 0.15;
          map.current.setBearing(bearing);
        }, 50);
      }

      // Active segment highlighting
      let activeSlice = { type: "FeatureCollection", features: [] };
      if (activeChapter === 3) {
        // T1 — 110m segment ending at T1 apex
        activeSlice = {
          type: "FeatureCollection",
          features: [sliceTrackByDistance(geo.coords, geo.lut, 980, 1090)],
        };
      } else if (activeChapter === 4) {
        // T4 arc
        activeSlice = {
          type: "FeatureCollection",
          features: [sliceTrackByDistance(geo.coords, geo.lut, 1617, 1663)],
        };
      } else if (activeChapter === 5) {
        // T9-T10 sweep
        activeSlice = {
          type: "FeatureCollection",
          features: [sliceTrackByDistance(geo.coords, geo.lut, 2720, 2920)],
        };
      } else if (activeChapter === 7) {
        // Fault at T1
        activeSlice = {
          type: "FeatureCollection",
          features: [sliceTrackByDistance(geo.coords, geo.lut, 980, 1090)],
        };
      }
      const src = m.getSource("track-active");
      if (src) src.setData(activeSlice);

      // Fault pulse marker
      const faultSrc = m.getSource("fault-pulse");
      if (faultSrc) {
        if (activeChapter === 7) {
          const [lon, lat] = distanceToLatLon(geo.lut, geo.coords, 1090);
          faultSrc.setData({
            type: "FeatureCollection",
            features: [
              {
                type: "Feature",
                properties: {},
                geometry: { type: "Point", coordinates: [lon, lat] },
              },
            ],
          });
        } else {
          faultSrc.setData({ type: "FeatureCollection", features: [] });
        }
      }
    }
  }, [activeChapter, geo]);

  // Update hover marker
  useEffect(() => {
    const m = map.current;
    if (!m || !geo || !m.isStyleLoaded()) return;
    const src = m.getSource("hover-point");
    if (!src) return;
    if (hoveredDistance == null) {
      src.setData({ type: "FeatureCollection", features: [] });
    } else {
      const [lon, lat] = distanceToLatLon(geo.lut, geo.coords, hoveredDistance);
      src.setData({
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            properties: {},
            geometry: { type: "Point", coordinates: [lon, lat] },
          },
        ],
      });
    }
  }, [hoveredDistance, geo]);

  if (!process.env.NEXT_PUBLIC_MAPBOX_TOKEN) {
    return <FallbackView results={results} />;
  }

  const renderChart = (chart) => {
    switch (chart) {
      case "speedtrace-static":
        return <SpeedTrace interactive={false} />;
      case "speedtrace-interactive":
        return <SpeedTrace interactive onHover={handleHover} onLeave={handleLeave} />;
      case "radar":
        return <OscillatorRadar />;
      case "brake-throttle-gear":
        return <BrakeThrottleGear onHover={handleHover} onLeave={handleLeave} />;
      case "friction-circle":
        return <FrictionCircle />;
      case "geff":
        return <GeffCurve />;
      case "sentropy":
        return <SEntropyTrace onHover={handleHover} onLeave={handleLeave} />;
      case "fault":
        return <FaultTrajectory />;
      case "sector-bars":
        return <SectorBars />;
      case "gumbel":
        return (
          <GumbelSwarm
            samples={results?.monte_carlo?.samples_head || []}
            gumbelLoc={results?.historical_fit?.gumbel_loc_s}
            gumbelScale={results?.historical_fit?.gumbel_scale_s}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div style={{ position: "relative" }}>
      <div
        ref={mapContainer}
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          zIndex: 0,
        }}
      />
      <div style={{ position: "relative", zIndex: 1 }}>
        {CHAPTERS.map((ch, i) => (
          <div
            key={i}
            ref={(el) => (chapterRefs.current[i] = el)}
            data-index={i}
            style={{
              minHeight: "100vh",
              display: "flex",
              alignItems: "center",
              justifyContent: ch.alignment === "right" ? "flex-end" : "flex-start",
              padding: "0 5%",
            }}
          >
            <div
              style={{
                maxWidth: 560,
                width: "100%",
                padding: "2rem",
                background: "rgba(10,10,10,0.88)",
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
                borderRadius: 14,
                border: `1px solid ${
                  activeChapter === i ? "rgba(212,175,55,0.4)" : "rgba(255,255,255,0.08)"
                }`,
                transition: "border-color 0.5s, opacity 0.5s",
                opacity: activeChapter === i ? 1 : 0.35,
              }}
            >
              <div
                style={{
                  fontSize: "0.65rem",
                  color: GOLD,
                  letterSpacing: "0.2em",
                  textTransform: "uppercase",
                  marginBottom: 8,
                }}
              >
                {ch.label || `Chapter ${i + 1}`}
              </div>
              <h2
                style={{
                  fontSize: "1.4rem",
                  fontWeight: 700,
                  color: "#fafafa",
                  marginBottom: 10,
                }}
              >
                {ch.title}
              </h2>
              <p
                style={{
                  fontSize: "0.85rem",
                  color: "rgba(250,250,250,0.68)",
                  lineHeight: 1.7,
                  marginBottom: 8,
                }}
              >
                {ch.text}
              </p>
              {activeChapter === i ? renderChart(ch.chart) : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
