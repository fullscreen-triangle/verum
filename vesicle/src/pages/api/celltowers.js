import path from "path";
import { promises as fs } from "fs";

export default async function handler(req, res) {
  try {
    const filePath = path.join(process.cwd(), "public", "data", "munich-celltowers.json");
    const data = await fs.readFile(filePath, "utf-8");
    res.setHeader("Cache-Control", "s-maxage=86400, stale-while-revalidate");
    res.status(200).json(JSON.parse(data));
  } catch {
    // Fallback: generate synthetic Munich cell tower data
    const towers = generateMunichTowers();
    res.status(200).json(towers);
  }
}

function generateMunichTowers() {
  const center = { lat: 48.1351, lng: 11.5820 };
  const features = [];
  // Generate ~200 synthetic towers around Munich
  for (let i = 0; i < 200; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const radius = Math.random() * 0.08; // ~8km
    features.push({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [
          center.lng + radius * Math.cos(angle),
          center.lat + radius * Math.sin(angle) * 0.7,
        ],
      },
      properties: {
        radio: ["LTE", "UMTS", "GSM", "NR"][Math.floor(Math.random() * 4)],
        range: 200 + Math.random() * 2000,
        signal: -50 - Math.random() * 60,
        operator: ["Telekom", "Vodafone", "O2"][Math.floor(Math.random() * 3)],
      },
    });
  }
  return { type: "FeatureCollection", features, _fallback: true };
}
