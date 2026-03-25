// Munich coordinates (TUM campus area)
export const MUNICH_CENTER = { lat: 48.1351, lng: 11.5820 };
export const MUNICH_BBOX = { west: 11.4, south: 48.0, east: 11.7, north: 48.2 };

// Color tokens matching tailwind.config.js
export const COLORS = {
  dark: "#0a0a0a",
  light: "#fafafa",
  gold: "#D4AF37",
  primary: "#C6A962",
  teal: "#2AA198",
  membrane: "#2AA198",
  primaryDark: "#58E6D9",
};

// API endpoints
export const API = {
  weather: "/api/weather",
  traffic: "/api/traffic",
  celltowers: "/api/celltowers",
};

// Mapbox style
export const MAPBOX_STYLE = "mapbox://styles/mapbox/dark-v11";

// Key metrics used across pages
export const METRICS = {
  papers: "30+",
  opsPerSecond: "10²⁸",
  validations: "18/18",
  freeParameters: "0",
  nightDetectionRange: "100m",
  brakeWarning: "240ms",
  aroundCorner: "10-20s",
  optimalPathError: "0.15m",
  convoyReduction: "92%",
  market: "$2.3T",
};
