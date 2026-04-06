import { useEffect, useRef } from "react";

export default function NetworkMap({ towers, traffic }) {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (map.current || !mapContainer.current) return;
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "";

    // If no token, show fallback
    if (!token) {
      mapContainer.current.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:center;height:100%;background:#111;color:#888;font-size:14px;text-align:center;padding:40px;">
          <div>
            <div style="font-size:48px;margin-bottom:16px;">🗺️</div>
            <div style="margin-bottom:8px;font-weight:bold;color:#D4AF37;">Mapbox Token Required</div>
            <div>Add NEXT_PUBLIC_MAPBOX_TOKEN to .env.local</div>
            <div style="margin-top:12px;color:#555;">Munich · 48.1351°N, 11.5820°E</div>
            <div style="margin-top:8px;color:#2AA198;">${towers?.features?.length || 200} cell towers · Distributed mesh ready</div>
          </div>
        </div>
      `;
      return;
    }

    const mapboxgl = require("mapbox-gl");
    mapboxgl.accessToken = token;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/dark-v11",
      center: [11.5820, 48.1351],
      zoom: 11,
    });

    map.current.on("load", () => {
      // Add cell tower heatmap
      if (towers?.features?.length > 0) {
        map.current.addSource("towers", { type: "geojson", data: towers });

        map.current.addLayer({
          id: "tower-heat",
          type: "heatmap",
          source: "towers",
          paint: {
            "heatmap-weight": 1,
            "heatmap-intensity": 1.5,
            "heatmap-radius": 30,
            "heatmap-color": [
              "interpolate", ["linear"], ["heatmap-density"],
              0, "rgba(42,161,152,0)",
              0.3, "rgba(42,161,152,0.4)",
              0.6, "rgba(198,169,98,0.6)",
              1, "rgba(212,175,55,0.9)",
            ],
            "heatmap-opacity": 0.7,
          },
        });

        map.current.addLayer({
          id: "tower-points",
          type: "circle",
          source: "towers",
          minzoom: 13,
          paint: {
            "circle-radius": 4,
            "circle-color": "#2AA198",
            "circle-stroke-width": 1,
            "circle-stroke-color": "#0a0a0a",
            "circle-opacity": 0.8,
          },
        });
      }
    });

    return () => {
      if (map.current) map.current.remove();
      map.current = null;
    };
  }, [towers]);

  return <div ref={mapContainer} style={{ width: "100%", height: "100%" }} />;
}
