export default async function handler(req, res) {
  const bbox = req.query.bbox || "11.4,48.0,11.7,48.2";
  const key = process.env.TOMTOM_API_KEY;

  if (!key) {
    // Fallback: synthetic Munich traffic data
    return res.status(200).json({
      flowSegments: [
        { name: "Leopoldstraße", freeFlow: 50, current: 35, confidence: 0.9 },
        { name: "Ludwigstraße", freeFlow: 50, current: 42, confidence: 0.85 },
        { name: "Maximilianstraße", freeFlow: 40, current: 28, confidence: 0.88 },
        { name: "Mittlerer Ring", freeFlow: 70, current: 45, confidence: 0.92 },
        { name: "Altstadtring", freeFlow: 30, current: 18, confidence: 0.87 },
        { name: "Prinzregentenstraße", freeFlow: 50, current: 40, confidence: 0.9 },
        { name: "Donnersbergerbrücke", freeFlow: 60, current: 30, confidence: 0.83 },
        { name: "Landshuter Allee", freeFlow: 70, current: 55, confidence: 0.91 },
      ],
      _fallback: true,
    });
  }

  try {
    const [west, south, east, north] = bbox.split(",");
    const lat = ((+south + +north) / 2).toFixed(4);
    const lon = ((+west + +east) / 2).toFixed(4);
    const url = `https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=${lat},${lon}&key=${key}`;
    const response = await fetch(url);
    const data = await response.json();
    res.setHeader("Cache-Control", "s-maxage=120, stale-while-revalidate");
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch traffic data" });
  }
}
