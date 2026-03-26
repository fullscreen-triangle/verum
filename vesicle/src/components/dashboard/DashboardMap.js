import { useEffect, useRef, useState, useCallback } from "react";
import mapboxgl from "mapbox-gl";
import { Deck } from "@deck.gl/core";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import LayerPanel from "./LayerPanel";
import InfoPanel from "./InfoPanel";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

// Munich center
const INITIAL_VIEW = { lng: 11.582, lat: 48.1351, zoom: 12.5, pitch: 45, bearing: -17.6 };

// Munich route segments for exhaust trails
const EXHAUST_ROUTES = [
  { name: "Leopoldstrasse", coords: [[11.5820,48.1550],[11.5818,48.1500],[11.5815,48.1450],[11.5810,48.1400],[11.5808,48.1351]] },
  { name: "Ludwigstrasse", coords: [[11.5795,48.1420],[11.5798,48.1380],[11.5800,48.1350],[11.5802,48.1310],[11.5805,48.1280]] },
  { name: "Maximilianstrasse", coords: [[11.5750,48.1380],[11.5800,48.1378],[11.5850,48.1375],[11.5900,48.1372],[11.5950,48.1370]] },
  { name: "Mittlerer Ring N", coords: [[11.5400,48.1550],[11.5500,48.1560],[11.5600,48.1565],[11.5700,48.1560],[11.5800,48.1555]] },
  { name: "Mittlerer Ring E", coords: [[11.6100,48.1500],[11.6105,48.1450],[11.6110,48.1400],[11.6108,48.1350],[11.6100,48.1300]] },
  { name: "Altstadtring", coords: [[11.5700,48.1380],[11.5730,48.1400],[11.5780,48.1410],[11.5830,48.1400],[11.5860,48.1380]] },
  { name: "Donnersbergerbruecke", coords: [[11.5380,48.1430],[11.5420,48.1420],[11.5470,48.1410],[11.5520,48.1400],[11.5570,48.1395]] },
  { name: "Landshuter Allee", coords: [[11.5360,48.1600],[11.5365,48.1550],[11.5370,48.1500],[11.5375,48.1450],[11.5380,48.1400]] },
];

// Synthetic computing nodes around Munich
const COMPUTING_NODES = Array.from({ length: 60 }, (_, i) => {
  const angle = (i / 60) * 2 * Math.PI;
  const r = 0.01 + Math.random() * 0.06;
  return {
    position: [11.582 + r * Math.cos(angle), 48.1351 + r * Math.sin(angle) * 0.7],
    capacity: 20 + Math.random() * 80,
    load: Math.random(),
    name: `Node ${i + 1}`,
  };
});

function NoTokenFallback() {
  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "#0a0a0a",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 20,
        padding: 40,
      }}
    >
      <div
        style={{
          width: 60,
          height: 60,
          borderRadius: 12,
          background: "linear-gradient(135deg, #D4AF37 0%, #2AA198 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 28,
        }}
      >
        V
      </div>
      <h1 style={{ color: "#fafafa", fontSize: 24, fontWeight: 700, margin: 0 }}>
        Vesicle Dashboard
      </h1>
      <p style={{ color: "rgba(250,250,250,0.5)", maxWidth: 400, textAlign: "center", lineHeight: 1.7 }}>
        No Mapbox token detected. To enable the full map dashboard, add your token to{" "}
        <code
          style={{
            background: "rgba(255,255,255,0.08)",
            padding: "2px 6px",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          .env.local
        </code>
        :
      </p>
      <pre
        style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 8,
          padding: "12px 20px",
          color: "#2AA198",
          fontSize: 13,
          fontFamily: "monospace",
        }}
      >
        NEXT_PUBLIC_MAPBOX_TOKEN=pk.your_token_here
      </pre>
      <p style={{ color: "rgba(250,250,250,0.3)", fontSize: 12 }}>
        Get a free token at{" "}
        <a href="https://mapbox.com" style={{ color: "#D4AF37" }}>
          mapbox.com
        </a>
      </p>
    </div>
  );
}

export default function DashboardMap() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);
  const deckRef = useRef(null);
  const deckCanvasRef = useRef(null);

  const [activeLayers, setActiveLayers] = useState(["buildings"]);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [infoData, setInfoData] = useState(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Data state
  const [celltowerData, setCelltowerData] = useState(null);
  const [weatherData, setWeatherData] = useState(null);
  const [trafficData, setTrafficData] = useState(null);
  const [isochroneData, setIsochroneData] = useState(null);
  const [directionsData, setDirectionsData] = useState(null);
  const [directionsPoints, setDirectionsPoints] = useState([]);
  const [exhaust, setExhaust] = useState([]);

  // Toggle layer
  const handleToggle = useCallback((id) => {
    setActiveLayers((prev) =>
      prev.includes(id) ? prev.filter((l) => l !== id) : [...prev, id]
    );
  }, []);

  // Initialize map
  useEffect(() => {
    if (!MAPBOX_TOKEN || mapRef.current) return;
    mapboxgl.accessToken = MAPBOX_TOKEN;

    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/dark-v11",
      center: [INITIAL_VIEW.lng, INITIAL_VIEW.lat],
      zoom: INITIAL_VIEW.zoom,
      pitch: INITIAL_VIEW.pitch,
      bearing: INITIAL_VIEW.bearing,
      antialias: true,
    });

    map.addControl(new mapboxgl.NavigationControl({ showCompass: true }), "bottom-right");
    map.addControl(new mapboxgl.ScaleControl({ maxWidth: 150, unit: "metric" }), "bottom-left");

    map.on("load", () => {
      mapRef.current = map;
      setMapLoaded(true);
    });

    map.on("click", (e) => {
      const coords = [e.lngLat.lng, e.lngLat.lat];
      setSelectedPoint(coords);
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Initialize DeckGL overlay
  useEffect(() => {
    if (!mapLoaded || !mapRef.current || deckRef.current) return;
    const map = mapRef.current;

    const deck = new Deck({
      canvas: deckCanvasRef.current,
      width: "100%",
      height: "100%",
      initialViewState: {
        longitude: INITIAL_VIEW.lng,
        latitude: INITIAL_VIEW.lat,
        zoom: INITIAL_VIEW.zoom,
        pitch: INITIAL_VIEW.pitch,
        bearing: INITIAL_VIEW.bearing,
      },
      controller: false,
      layers: [],
      getTooltip: ({ object }) => {
        if (!object) return null;
        if (object.properties?.radio) {
          return `${object.properties.operator} - ${object.properties.radio}\nRange: ${Math.round(object.properties.range)}m\nSignal: ${Math.round(object.properties.signal)} dBm`;
        }
        if (object.name && object.capacity !== undefined) {
          return `${object.name}\nCapacity: ${Math.round(object.capacity)} TFLOPS\nLoad: ${Math.round(object.load * 100)}%`;
        }
        return null;
      },
    });

    deckRef.current = deck;

    // Sync deck viewport with map
    const syncDeck = () => {
      const center = map.getCenter();
      deck.setProps({
        viewState: {
          longitude: center.lng,
          latitude: center.lat,
          zoom: map.getZoom(),
          pitch: map.getPitch(),
          bearing: map.getBearing(),
        },
      });
    };

    map.on("move", syncDeck);
    map.on("resize", () => {
      deckCanvasRef.current.width = map.getCanvas().width;
      deckCanvasRef.current.height = map.getCanvas().height;
      syncDeck();
    });
    syncDeck();

    return () => {
      deck.finalize();
      deckRef.current = null;
    };
  }, [mapLoaded]);

  // -- Data fetchers --

  // Cell towers
  useEffect(() => {
    if (!activeLayers.includes("celltowers")) return;
    if (celltowerData) return;
    fetch("/api/celltowers")
      .then((r) => r.json())
      .then(setCelltowerData)
      .catch(console.error);
  }, [activeLayers, celltowerData]);

  // Weather
  useEffect(() => {
    if (!activeLayers.includes("weather")) return;
    if (weatherData) return;
    fetch("/api/weather?lat=48.1351&lon=11.582")
      .then((r) => r.json())
      .then(setWeatherData)
      .catch(console.error);
  }, [activeLayers, weatherData]);

  // Traffic
  useEffect(() => {
    if (!activeLayers.includes("traffic")) return;
    if (trafficData) return;
    fetch("/api/traffic?bbox=11.4,48.0,11.7,48.2")
      .then((r) => r.json())
      .then(setTrafficData)
      .catch(console.error);
  }, [activeLayers, trafficData]);

  // Isochrone -- fetch when layer is active and a point is selected
  useEffect(() => {
    if (!activeLayers.includes("isochrone") || !selectedPoint) return;
    fetch(`/api/isochrone?lat=${selectedPoint[1]}&lon=${selectedPoint[0]}&minutes=15`)
      .then((r) => r.json())
      .then(setIsochroneData)
      .catch(console.error);
  }, [activeLayers, selectedPoint]);

  // Directions -- collect two points
  useEffect(() => {
    if (!activeLayers.includes("directions") || !selectedPoint) return;
    setDirectionsPoints((prev) => {
      const next = [...prev, selectedPoint];
      if (next.length > 2) return [selectedPoint]; // reset on 3rd click
      return next;
    });
  }, [activeLayers, selectedPoint]);

  useEffect(() => {
    if (directionsPoints.length !== 2) return;
    const [o, d] = directionsPoints;
    fetch(`/api/directions?origin=${o[0]},${o[1]}&destination=${d[0]},${d[1]}`)
      .then((r) => r.json())
      .then(setDirectionsData)
      .catch(console.error);
  }, [directionsPoints]);

  // Exhaust trails -- animate
  useEffect(() => {
    if (!activeLayers.includes("exhaust")) {
      setExhaust([]);
      return;
    }
    // Generate particles along routes
    const particles = [];
    EXHAUST_ROUTES.forEach((route) => {
      for (let i = 0; i < route.coords.length - 1; i++) {
        const [x1, y1] = route.coords[i];
        const [x2, y2] = route.coords[i + 1];
        const numParticles = 8;
        for (let p = 0; p < numParticles; p++) {
          const t = p / numParticles;
          const spread = (Math.random() - 0.5) * 0.002;
          particles.push({
            position: [x1 + (x2 - x1) * t + spread, y1 + (y2 - y1) * t + spread * 0.7],
            intensity: 0.3 + Math.random() * 0.7,
            size: 40 + Math.random() * 80,
          });
        }
      }
    });
    setExhaust(particles);

    // Animate drift
    const interval = setInterval(() => {
      setExhaust((prev) =>
        prev.map((p) => ({
          ...p,
          position: [
            p.position[0] + (Math.random() - 0.5) * 0.0001,
            p.position[1] + Math.random() * 0.00005,
          ],
          intensity: Math.max(0.1, p.intensity - 0.01 + Math.random() * 0.02),
        }))
      );
    }, 200);

    return () => clearInterval(interval);
  }, [activeLayers]);

  // -- Build info panel data when point is clicked --
  useEffect(() => {
    if (!selectedPoint) {
      setInfoData(null);
      return;
    }
    const layers = {};

    if (activeLayers.includes("weather") && weatherData) {
      layers["Weather"] = {
        Temperature: `${weatherData.main?.temp ?? "--"}°C`,
        Humidity: `${weatherData.main?.humidity ?? "--"}%`,
        Wind: `${weatherData.wind?.speed ?? "--"} m/s`,
        Conditions: weatherData.weather?.[0]?.description ?? "--",
      };
    }

    if (activeLayers.includes("traffic") && trafficData?.flowSegments) {
      const nearest = trafficData.flowSegments[0];
      layers["Traffic"] = {
        "Nearest Road": nearest?.name ?? "--",
        "Free Flow": `${nearest?.freeFlow ?? "--"} km/h`,
        "Current Speed": `${nearest?.current ?? "--"} km/h`,
        Confidence: `${((nearest?.confidence ?? 0) * 100).toFixed(0)}%`,
      };
    }

    if (activeLayers.includes("isochrone") && isochroneData) {
      layers["Isochrone"] = {
        "Travel Time": "15 min",
        Profile: "Driving",
        Status: isochroneData._fallback ? "Simulated" : "Live",
      };
    }

    if (activeLayers.includes("directions") && directionsData?.routes?.[0]) {
      const route = directionsData.routes[0];
      layers["Directions"] = {
        Distance: `${(route.distance / 1000).toFixed(1)} km`,
        Duration: `${Math.round(route.duration / 60)} min`,
        Status: directionsData._fallback ? "Simulated" : "Live",
      };
    }

    if (activeLayers.includes("computing")) {
      // Find nearest computing node
      let minDist = Infinity;
      let nearest = null;
      COMPUTING_NODES.forEach((node) => {
        const d = Math.hypot(node.position[0] - selectedPoint[0], node.position[1] - selectedPoint[1]);
        if (d < minDist) {
          minDist = d;
          nearest = node;
        }
      });
      if (nearest) {
        layers["Computing"] = {
          Node: nearest.name,
          Capacity: `${Math.round(nearest.capacity)} TFLOPS`,
          Load: `${Math.round(nearest.load * 100)}%`,
          Latency: `${Math.round(2 + Math.random() * 8)} ms`,
        };
      }
    }

    setInfoData({
      coordinates: selectedPoint,
      timestamp: Date.now(),
      layers,
    });
  }, [selectedPoint, activeLayers, weatherData, trafficData, isochroneData, directionsData]);

  // -- Mapbox native layers --

  // 3D Buildings
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    const layerId = "3d-buildings";

    if (activeLayers.includes("buildings")) {
      if (!map.getLayer(layerId)) {
        const layers = map.getStyle().layers;
        let labelLayerId;
        for (const layer of layers) {
          if (layer.type === "symbol" && layer.layout?.["text-field"]) {
            labelLayerId = layer.id;
            break;
          }
        }
        map.addLayer(
          {
            id: layerId,
            source: "composite",
            "source-layer": "building",
            filter: ["==", "extrude", "true"],
            type: "fill-extrusion",
            minzoom: 12,
            paint: {
              "fill-extrusion-color": [
                "interpolate",
                ["linear"],
                ["get", "height"],
                0, "#1a1a2e",
                50, "#16213e",
                100, "#0f3460",
                200, "#533483",
              ],
              "fill-extrusion-height": ["get", "height"],
              "fill-extrusion-base": ["get", "min_height"],
              "fill-extrusion-opacity": 0.7,
            },
          },
          labelLayerId
        );
      }
    } else {
      if (map.getLayer(layerId)) map.removeLayer(layerId);
    }
  }, [activeLayers, mapLoaded]);

  // Isochrone polygon
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    const sourceId = "isochrone-source";
    const layerId = "isochrone-fill";
    const outlineId = "isochrone-outline";

    if (activeLayers.includes("isochrone") && isochroneData?.features?.length) {
      if (map.getSource(sourceId)) {
        map.getSource(sourceId).setData(isochroneData);
      } else {
        map.addSource(sourceId, { type: "geojson", data: isochroneData });
        map.addLayer({
          id: layerId,
          type: "fill",
          source: sourceId,
          paint: {
            "fill-color": "#2AA198",
            "fill-opacity": 0.15,
          },
        });
        map.addLayer({
          id: outlineId,
          type: "line",
          source: sourceId,
          paint: {
            "line-color": "#2AA198",
            "line-width": 2,
            "line-opacity": 0.8,
          },
        });
      }
    } else {
      if (map.getLayer(outlineId)) map.removeLayer(outlineId);
      if (map.getLayer(layerId)) map.removeLayer(layerId);
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    }
  }, [activeLayers, isochroneData, mapLoaded]);

  // Directions route line
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    const sourceId = "directions-source";
    const layerId = "directions-line";
    const glowId = "directions-glow";

    if (activeLayers.includes("directions") && directionsData?.routes?.[0]?.geometry) {
      const geojson = {
        type: "Feature",
        geometry: directionsData.routes[0].geometry,
        properties: {},
      };

      if (map.getSource(sourceId)) {
        map.getSource(sourceId).setData(geojson);
      } else {
        map.addSource(sourceId, { type: "geojson", data: geojson });
        // Glow layer
        map.addLayer({
          id: glowId,
          type: "line",
          source: sourceId,
          paint: {
            "line-color": "#D4AF37",
            "line-width": 10,
            "line-opacity": 0.15,
            "line-blur": 8,
          },
        });
        // Main route line
        map.addLayer({
          id: layerId,
          type: "line",
          source: sourceId,
          paint: {
            "line-color": "#D4AF37",
            "line-width": 3,
            "line-opacity": 0.9,
          },
        });
      }
    } else {
      if (map.getLayer(glowId)) map.removeLayer(glowId);
      if (map.getLayer(layerId)) map.removeLayer(layerId);
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    }
  }, [activeLayers, directionsData, mapLoaded]);

  // Traffic visualization (color-coded road segments)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    if (activeLayers.includes("traffic") && trafficData?.flowSegments) {
      // Add colored markers for traffic segments
      const sourceId = "traffic-points";
      const layerId = "traffic-circles";

      const features = trafficData.flowSegments.map((seg, i) => {
        const angle = (i / trafficData.flowSegments.length) * 2 * Math.PI;
        const r = 0.02 + i * 0.005;
        const ratio = seg.current / seg.freeFlow;
        return {
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [11.582 + r * Math.cos(angle), 48.1351 + r * Math.sin(angle) * 0.7],
          },
          properties: {
            name: seg.name,
            speed: seg.current,
            freeFlow: seg.freeFlow,
            ratio,
            color: ratio > 0.7 ? "#4ade80" : ratio > 0.4 ? "#fbbf24" : "#ef4444",
          },
        };
      });

      const geojson = { type: "FeatureCollection", features };

      if (map.getSource(sourceId)) {
        map.getSource(sourceId).setData(geojson);
      } else {
        map.addSource(sourceId, { type: "geojson", data: geojson });
        map.addLayer({
          id: layerId,
          type: "circle",
          source: sourceId,
          paint: {
            "circle-radius": 8,
            "circle-color": ["get", "color"],
            "circle-opacity": 0.8,
            "circle-blur": 0.3,
            "circle-stroke-width": 1,
            "circle-stroke-color": "rgba(255,255,255,0.2)",
          },
        });
      }
    } else {
      const sourceId = "traffic-points";
      const layerId = "traffic-circles";
      if (map.getLayer(layerId)) map.removeLayer(layerId);
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    }
  }, [activeLayers, trafficData, mapLoaded]);

  // Weather overlay (atmospheric haze effect)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    if (activeLayers.includes("weather") && weatherData) {
      map.setFog({
        color: "rgba(30, 40, 60, 0.8)",
        "high-color": "rgba(20, 30, 50, 0.6)",
        "horizon-blend": 0.08,
        "space-color": "rgba(10, 10, 20, 1)",
        "star-intensity": 0.4,
      });
    } else {
      map.setFog({});
    }
  }, [activeLayers, weatherData, mapLoaded]);

  // -- DeckGL layers --
  useEffect(() => {
    if (!deckRef.current) return;

    const layers = [];

    // Cell tower heatmap
    if (activeLayers.includes("celltowers") && celltowerData?.features) {
      layers.push(
        new HeatmapLayer({
          id: "celltower-heat",
          data: celltowerData.features,
          getPosition: (d) => d.geometry.coordinates,
          getWeight: (d) => d.properties.range / 2000,
          radiusPixels: 40,
          intensity: 1.2,
          threshold: 0.1,
          colorRange: [
            [0, 50, 50],
            [0, 100, 100],
            [88, 230, 217],
            [88, 230, 217],
            [200, 255, 240],
            [255, 255, 255],
          ],
        })
      );

      layers.push(
        new ScatterplotLayer({
          id: "celltower-points",
          data: celltowerData.features,
          getPosition: (d) => d.geometry.coordinates,
          getFillColor: (d) => {
            const r = d.properties.radio;
            if (r === "NR") return [88, 230, 217, 200];
            if (r === "LTE") return [42, 161, 152, 180];
            if (r === "UMTS") return [198, 169, 98, 160];
            return [136, 136, 136, 140];
          },
          getRadius: (d) => Math.max(30, d.properties.range / 20),
          radiusMinPixels: 2,
          radiusMaxPixels: 12,
          pickable: true,
          opacity: 0.7,
        })
      );
    }

    // Exhaust trails
    if (activeLayers.includes("exhaust") && exhaust.length > 0) {
      layers.push(
        new ScatterplotLayer({
          id: "exhaust-particles",
          data: exhaust,
          getPosition: (d) => d.position,
          getFillColor: (d) => [255, 159, 67, Math.round(d.intensity * 120)],
          getRadius: (d) => d.size,
          radiusMinPixels: 3,
          radiusMaxPixels: 20,
          opacity: 0.5,
        })
      );

      // Route paths
      layers.push(
        new PathLayer({
          id: "exhaust-paths",
          data: EXHAUST_ROUTES,
          getPath: (d) => d.coords,
          getColor: [255, 159, 67, 60],
          getWidth: 30,
          widthMinPixels: 2,
          widthMaxPixels: 15,
          capRounded: true,
          jointRounded: true,
        })
      );
    }

    // Computing nodes
    if (activeLayers.includes("computing")) {
      layers.push(
        new ScatterplotLayer({
          id: "computing-nodes",
          data: COMPUTING_NODES,
          getPosition: (d) => d.position,
          getFillColor: (d) => {
            const load = d.load;
            if (load < 0.3) return [42, 161, 152, 200];
            if (load < 0.6) return [212, 175, 55, 200];
            return [255, 107, 107, 200];
          },
          getRadius: (d) => 40 + d.capacity * 2,
          radiusMinPixels: 4,
          radiusMaxPixels: 18,
          pickable: true,
          stroked: true,
          lineWidthMinPixels: 1,
          getLineColor: [255, 255, 255, 40],
          opacity: 0.8,
        })
      );
    }

    // Direction waypoint markers
    if (activeLayers.includes("directions") && directionsPoints.length > 0) {
      layers.push(
        new ScatterplotLayer({
          id: "direction-waypoints",
          data: directionsPoints.map((p, i) => ({ position: p, index: i })),
          getPosition: (d) => d.position,
          getFillColor: (d) => (d.index === 0 ? [42, 161, 152, 240] : [212, 175, 55, 240]),
          getRadius: 100,
          radiusMinPixels: 8,
          radiusMaxPixels: 14,
          stroked: true,
          lineWidthMinPixels: 2,
          getLineColor: [255, 255, 255, 200],
        })
      );
    }

    deckRef.current.setProps({ layers });
  }, [activeLayers, celltowerData, exhaust, directionsPoints]);

  if (!MAPBOX_TOKEN) return <NoTokenFallback />;

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden", background: "#0a0a0a" }}>
      {/* Mapbox container */}
      <div ref={mapContainer} style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0 }} />

      {/* DeckGL overlay canvas */}
      <canvas
        ref={deckCanvasRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />

      {/* Top bar */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 56,
          zIndex: 20,
          background: "linear-gradient(180deg, rgba(10,10,10,0.95) 0%, rgba(10,10,10,0) 100%)",
          display: "flex",
          alignItems: "center",
          padding: "0 24px",
          gap: 16,
        }}
      >
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: 6,
            background: "linear-gradient(135deg, #D4AF37 0%, #2AA198 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 14,
            fontWeight: 800,
            color: "#0a0a0a",
          }}
        >
          V
        </div>
        <span style={{ color: "#fafafa", fontSize: 15, fontWeight: 700, letterSpacing: "0.04em" }}>
          Vesicle
        </span>
        <span style={{ color: "rgba(250,250,250,0.3)", fontSize: 12, marginLeft: 4 }}>Dashboard</span>
        <div style={{ flex: 1 }} />
        <span
          style={{
            fontSize: 11,
            color: "rgba(250,250,250,0.3)",
            background: "rgba(255,255,255,0.05)",
            padding: "4px 10px",
            borderRadius: 20,
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          Munich, DE
        </span>
        {activeLayers.length > 0 && (
          <span
            style={{
              fontSize: 11,
              color: "#D4AF37",
              background: "rgba(212,175,55,0.1)",
              padding: "4px 10px",
              borderRadius: 20,
              border: "1px solid rgba(212,175,55,0.2)",
            }}
          >
            {activeLayers.length} layer{activeLayers.length !== 1 ? "s" : ""} active
          </span>
        )}
      </div>

      {/* Layer panel */}
      <LayerPanel activeLayers={activeLayers} onToggle={handleToggle} />

      {/* Info panel */}
      <InfoPanel data={infoData} onClose={() => { setSelectedPoint(null); setInfoData(null); }} />

      {/* Directions hint */}
      {activeLayers.includes("directions") && directionsPoints.length < 2 && (
        <div
          style={{
            position: "absolute",
            bottom: 40,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 15,
            background: "rgba(10,10,10,0.9)",
            backdropFilter: "blur(8px)",
            padding: "8px 20px",
            borderRadius: 20,
            border: "1px solid rgba(212,175,55,0.3)",
            color: "#D4AF37",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          {directionsPoints.length === 0
            ? "Click to set origin"
            : "Click to set destination"}
        </div>
      )}

      {/* Isochrone hint */}
      {activeLayers.includes("isochrone") && !selectedPoint && (
        <div
          style={{
            position: "absolute",
            bottom: 40,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 15,
            background: "rgba(10,10,10,0.9)",
            backdropFilter: "blur(8px)",
            padding: "8px 20px",
            borderRadius: 20,
            border: "1px solid rgba(42,161,152,0.3)",
            color: "#2AA198",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          Click map to show 15-min reach
        </div>
      )}
    </div>
  );
}
