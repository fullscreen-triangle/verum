import { useEffect, useRef } from "react";

// Munich route: Leopoldstraße → Ludwigstraße → Odeonsplatz
const ROUTE = [
  [11.5847, 48.1620], [11.5845, 48.1590], [11.5842, 48.1560],
  [11.5840, 48.1530], [11.5838, 48.1500], [11.5835, 48.1470],
  [11.5833, 48.1440], [11.5831, 48.1410], [11.5830, 48.1380],
  [11.5770, 48.1370], [11.5740, 48.1365], [11.5710, 48.1360],
];

// Generate synthetic trail with heat/diffusion
function generateTrailData() {
  const features = [];
  const now = Date.now();
  // Multiple "past vehicles" with decaying trails
  for (let v = 0; v < 20; v++) {
    const ageMinutes = v * 8 + Math.random() * 5;
    const offset = (Math.random() - 0.5) * 0.0003;
    for (let i = 0; i < ROUTE.length - 1; i++) {
      const spread = 0.0001 + ageMinutes * 0.00001; // trail widens with age
      const intensity = Math.max(0.1, 1 - ageMinutes / 180); // fades over 3 hours
      for (let j = 0; j < 5; j++) {
        features.push({
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [
              ROUTE[i][0] + offset + (Math.random() - 0.5) * spread * 2,
              ROUTE[i][1] + (Math.random() - 0.5) * spread * 2,
            ],
          },
          properties: { intensity, age: ageMinutes },
        });
      }
    }
  }
  return { type: "FeatureCollection", features };
}

export default function MapContainer() {
  const container = useRef(null);
  const map = useRef(null);
  const animFrame = useRef(null);
  const carMarker = useRef(null);
  const routeIdx = useRef(0);

  useEffect(() => {
    if (map.current || !container.current) return;
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

    if (!token) {
      container.current.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:center;height:100%;background:#111;color:#888;font-size:14px;text-align:center;padding:40px;">
          <div>
            <div style="font-size:48px;margin-bottom:16px;">🗺️</div>
            <div style="margin-bottom:8px;font-weight:bold;color:#D4AF37;">Mapbox Token Required</div>
            <div>Add NEXT_PUBLIC_MAPBOX_TOKEN to .env.local</div>
            <div style="margin-top:12px;color:#2AA198;">Munich · Molecular Trail Visualization</div>
          </div>
        </div>
      `;
      return;
    }

    const mapboxgl = require("mapbox-gl");
    mapboxgl.accessToken = token;

    map.current = new mapboxgl.Map({
      container: container.current,
      style: "mapbox://styles/mapbox/dark-v11",
      center: [11.5835, 48.1470],
      zoom: 14,
      pitch: 45,
      bearing: -10,
    });

    map.current.on("load", () => {
      const trailData = generateTrailData();

      // Exhaust trail heatmap
      map.current.addSource("exhaust-trail", { type: "geojson", data: trailData });
      map.current.addLayer({
        id: "trail-heat",
        type: "heatmap",
        source: "exhaust-trail",
        paint: {
          "heatmap-weight": ["get", "intensity"],
          "heatmap-intensity": 1.2,
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 12, 10, 16, 30],
          "heatmap-color": [
            "interpolate", ["linear"], ["heatmap-density"],
            0, "rgba(42,161,152,0)",
            0.2, "rgba(42,161,152,0.3)",
            0.5, "rgba(198,169,98,0.5)",
            0.8, "rgba(212,175,55,0.7)",
            1, "rgba(212,175,55,0.9)",
          ],
          "heatmap-opacity": 0.8,
        },
      });

      // Route line
      map.current.addSource("route", {
        type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: ROUTE } },
      });
      map.current.addLayer({
        id: "route-line",
        type: "line",
        source: "route",
        paint: {
          "line-color": "#D4AF37",
          "line-width": 2,
          "line-opacity": 0.4,
          "line-dasharray": [2, 4],
        },
      });

      // Animated car marker
      const el = document.createElement("div");
      el.style.cssText = "width:16px;height:16px;background:#D4AF37;border-radius:50%;box-shadow:0 0 12px rgba(212,175,55,0.6);border:2px solid #0a0a0a;";
      carMarker.current = new mapboxgl.Marker(el).setLngLat(ROUTE[0]).addTo(map.current);

      // Sensing radius
      const radiusEl = document.createElement("div");
      radiusEl.style.cssText = "width:80px;height:80px;border:1px solid rgba(42,161,152,0.3);border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;animation:pulse 2s infinite;";
      const style = document.createElement("style");
      style.textContent = "@keyframes pulse{0%,100%{opacity:0.3;transform:translate(-50%,-50%) scale(1)}50%{opacity:0.6;transform:translate(-50%,-50%) scale(1.2)}}";
      document.head.appendChild(style);
      new mapboxgl.Marker({ element: radiusEl }).setLngLat(ROUTE[0]).addTo(map.current);

      // Animate car along route
      let idx = 0;
      const animate = () => {
        idx = (idx + 0.02) % (ROUTE.length - 1);
        const i = Math.floor(idx);
        const t = idx - i;
        const lng = ROUTE[i][0] + t * (ROUTE[i + 1][0] - ROUTE[i][0]);
        const lat = ROUTE[i][1] + t * (ROUTE[i + 1][1] - ROUTE[i][1]);
        carMarker.current.setLngLat([lng, lat]);
        radiusEl.parentElement && new mapboxgl.Marker({ element: radiusEl }).setLngLat([lng, lat]);
        animFrame.current = requestAnimationFrame(animate);
      };
      animFrame.current = requestAnimationFrame(animate);
    });

    return () => {
      if (animFrame.current) cancelAnimationFrame(animFrame.current);
      if (map.current) map.current.remove();
      map.current = null;
    };
  }, []);

  return <div ref={container} style={{ width: "100%", height: "100%" }} />;
}
