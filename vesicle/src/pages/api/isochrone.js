export default async function handler(req, res) {
  const { lat = 48.1351, lon = 11.582, minutes = 15, profile = "driving" } = req.query;
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

  if (!token) {
    // Fallback: synthetic isochrone polygon around the requested point
    const center = [parseFloat(lon), parseFloat(lat)];
    const mins = parseInt(minutes, 10);
    const radius = mins * 0.004; // rough scaling
    const steps = 36;
    const coords = [];
    for (let i = 0; i <= steps; i++) {
      const angle = (i / steps) * 2 * Math.PI;
      const jitter = 0.85 + Math.random() * 0.3;
      coords.push([
        center[0] + radius * Math.cos(angle) * jitter,
        center[1] + radius * Math.sin(angle) * 0.7 * jitter,
      ]);
    }
    coords[coords.length - 1] = coords[0]; // close the ring

    return res.status(200).json({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "Polygon", coordinates: [coords] },
          properties: { contour: mins, color: "#2AA198", opacity: 0.3 },
        },
      ],
      _fallback: true,
    });
  }

  try {
    const url = `https://api.mapbox.com/isochrone/v1/mapbox/${profile}/${lon},${lat}?contours_minutes=${minutes}&polygons=true&access_token=${token}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Mapbox API returned ${response.status}`);
    }
    const data = await response.json();
    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate");
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch isochrone data", message: error.message });
  }
}
