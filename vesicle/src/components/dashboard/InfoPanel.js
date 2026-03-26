import { motion, AnimatePresence } from "framer-motion";

export default function InfoPanel({ data, onClose }) {
  if (!data) return null;

  return (
    <AnimatePresence>
      <motion.div
        key="info-panel"
        initial={{ x: 300, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 300, opacity: 0 }}
        transition={{ type: "spring", damping: 24, stiffness: 200 }}
        style={{
          position: "absolute",
          top: 80,
          right: 16,
          zIndex: 10,
          width: 300,
          background: "rgba(10,10,10,0.9)",
          backdropFilter: "blur(12px)",
          borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.08)",
          padding: 20,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 16,
          }}
        >
          <span
            style={{
              fontSize: 11,
              color: "#D4AF37",
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              fontWeight: 700,
            }}
          >
            Location Data
          </span>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#888",
              cursor: "pointer",
              fontSize: 18,
              lineHeight: 1,
              padding: "4px 8px",
              borderRadius: 4,
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#fff")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#888")}
          >
            &times;
          </button>
        </div>

        {data.coordinates && (
          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 10,
                color: "rgba(250,250,250,0.4)",
                textTransform: "uppercase",
                marginBottom: 4,
                letterSpacing: "0.08em",
              }}
            >
              Coordinates
            </div>
            <div style={{ fontSize: 13, color: "#fafafa", fontFamily: "monospace" }}>
              {data.coordinates[1].toFixed(4)}&deg;N, {data.coordinates[0].toFixed(4)}&deg;E
            </div>
          </div>
        )}

        {data.address && (
          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 10,
                color: "rgba(250,250,250,0.4)",
                textTransform: "uppercase",
                marginBottom: 4,
                letterSpacing: "0.08em",
              }}
            >
              Address
            </div>
            <div style={{ fontSize: 13, color: "#fafafa" }}>{data.address}</div>
          </div>
        )}

        {data.layers &&
          Object.entries(data.layers).map(([key, value]) => (
            <div
              key={key}
              style={{
                marginBottom: 10,
                padding: "8px 0",
                borderTop: "1px solid rgba(255,255,255,0.05)",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: "rgba(250,250,250,0.4)",
                  textTransform: "uppercase",
                  marginBottom: 4,
                  letterSpacing: "0.08em",
                }}
              >
                {key}
              </div>
              {typeof value === "object" && value !== null ? (
                Object.entries(value).map(([k, v]) => (
                  <div
                    key={k}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: 12,
                      marginBottom: 2,
                      padding: "2px 0",
                    }}
                  >
                    <span style={{ color: "rgba(250,250,250,0.6)" }}>{k}</span>
                    <span style={{ color: "#fafafa", fontWeight: 600 }}>{String(v)}</span>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: 13, color: "#fafafa" }}>{String(value)}</div>
              )}
            </div>
          ))}

        {data.timestamp && (
          <div
            style={{
              marginTop: 12,
              paddingTop: 8,
              borderTop: "1px solid rgba(255,255,255,0.05)",
              fontSize: 10,
              color: "rgba(250,250,250,0.3)",
              textAlign: "right",
            }}
          >
            {new Date(data.timestamp).toLocaleTimeString()}
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
