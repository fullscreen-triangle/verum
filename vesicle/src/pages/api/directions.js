export default async function handler(req, res) {
  const { origin, destination, profile = "driving" } = req.query;
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

  if (!origin || !destination) {
    return res.status(400).json({ error: "Missing origin or destination. Use format: lon,lat" });
  }

  if (!token) {
    // Fallback: synthetic route between origin and destination
    const [oLon, oLat] = origin.split(",").map(Number);
    const [dLon, dLat] = destination.split(",").map(Number);
    const steps = 20;
    const coords = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      // Add slight curve to make it look like a real route
      const midOffsetLon = (dLon - oLon) * 0.15 * Math.sin(t * Math.PI);
      const midOffsetLat = (dLat - oLat) * 0.1 * Math.sin(t * Math.PI);
      coords.push([
        oLon + (dLon - oLon) * t + midOffsetLon * (1 - Math.abs(2 * t - 1)),
        oLat + (dLat - oLat) * t + midOffsetLat * (1 - Math.abs(2 * t - 1)),
      ]);
    }

    const distance = Math.sqrt((dLon - oLon) ** 2 + (dLat - oLat) ** 2) * 111000; // rough meters
    const duration = distance / 13.9; // ~50 km/h in seconds

    return res.status(200).json({
      routes: [
        {
          geometry: { type: "LineString", coordinates: coords },
          distance: Math.round(distance),
          duration: Math.round(duration),
          legs: [{ summary: "Synthetic route", distance: Math.round(distance), duration: Math.round(duration) }],
        },
      ],
      waypoints: [
        { location: [oLon, oLat], name: "Origin" },
        { location: [dLon, dLat], name: "Destination" },
      ],
      _fallback: true,
    });
  }

  try {
    const url = `https://api.mapbox.com/directions/v5/mapbox/${profile}/${origin};${destination}?geometries=geojson&access_token=${token}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Mapbox API returned ${response.status}`);
    }
    const data = await response.json();
    res.setHeader("Cache-Control", "s-maxage=120, stale-while-revalidate");
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch directions", message: error.message });
  }
}
