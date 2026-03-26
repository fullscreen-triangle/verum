import { motion } from "framer-motion";

const LAYERS = [
  { id: "traffic", name: "Traffic Flow", color: "#ff6b6b", desc: "Real-time speed data from TomTom" },
  { id: "isochrone", name: "Isochrone", color: "#2AA198", desc: "Reachability from selected point" },
  { id: "directions", name: "Directions", color: "#D4AF37", desc: "Route between two points" },
  { id: "celltowers", name: "Cell Towers", color: "#58E6D9", desc: "Tower density heatmap" },
  { id: "weather", name: "Weather", color: "#C6A962", desc: "Temperature & wind overlay" },
  { id: "exhaust", name: "Exhaust Trails", color: "#ff9f43", desc: "Molecular trail simulation" },
  { id: "buildings", name: "3D Buildings", color: "#888", desc: "Extruded building footprints" },
  { id: "computing", name: "Computing Power", color: "#2AA198", desc: "Distributed mesh capacity" },
];

export default function LayerPanel({ activeLayers, onToggle }) {
  return (
    <motion.div
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.4 }}
      style={{
        position: "absolute",
        top: 80,
        left: 16,
        zIndex: 10,
        width: 260,
        background: "rgba(10,10,10,0.9)",
        backdropFilter: "blur(12px)",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.08)",
        padding: 16,
        maxHeight: "calc(100vh - 120px)",
        overflowY: "auto",
      }}
    >
      <div
        style={{
          fontSize: 11,
          color: "#D4AF37",
          letterSpacing: "0.15em",
          textTransform: "uppercase",
          marginBottom: 12,
          fontWeight: 700,
        }}
      >
        Map Layers
      </div>
      {LAYERS.map((layer) => {
        const active = activeLayers.includes(layer.id);
        return (
          <button
            key={layer.id}
            onClick={() => onToggle(layer.id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              width: "100%",
              padding: "8px 10px",
              marginBottom: 4,
              borderRadius: 8,
              border: "none",
              background: active ? "rgba(255,255,255,0.06)" : "transparent",
              cursor: "pointer",
              textAlign: "left",
              transition: "background 0.2s",
            }}
            onMouseEnter={(e) => {
              if (!active) e.currentTarget.style.background = "rgba(255,255,255,0.03)";
            }}
            onMouseLeave={(e) => {
              if (!active) e.currentTarget.style.background = "transparent";
            }}
          >
            <div
              style={{
                width: 14,
                height: 14,
                borderRadius: 3,
                background: active ? layer.color : "transparent",
                border: `2px solid ${layer.color}`,
                transition: "background 0.2s",
                flexShrink: 0,
                boxShadow: active ? `0 0 8px ${layer.color}44` : "none",
              }}
            />
            <div>
              <div style={{ fontSize: 13, color: "#fafafa", fontWeight: 600 }}>{layer.name}</div>
              <div style={{ fontSize: 10, color: "rgba(250,250,250,0.4)" }}>{layer.desc}</div>
            </div>
          </button>
        );
      })}
    </motion.div>
  );
}
