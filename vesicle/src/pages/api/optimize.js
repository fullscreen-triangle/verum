export default async function handler(req, res) {
  const { coordinates, profile = "driving" } = req.query;
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

  if (!coordinates) {
    return res.status(400).json({ error: "Missing coordinates. Use format: lon,lat;lon,lat;..." });
  }

  if (!token) {
    // Fallback: synthetic optimized trip
    const points = coordinates.split(";").map((p) => {
      const [lon, lat] = p.split(",").map(Number);
      return [lon, lat];
    });

    // Build a simple route through all points in order (synthetic optimization)
    const allCoords = [];
    let totalDistance = 0;
    let totalDuration = 0;

    for (let i = 0; i < points.length; i++) {
      const from = points[i];
      const to = points[(i + 1) % points.length];
      const steps = 10;
      for (let s = 0; s <= steps; s++) {
        const t = s / steps;
        allCoords.push([
          from[0] + (to[0] - from[0]) * t,
          from[1] + (to[1] - from[1]) * t,
        ]);
      }
      const dist = Math.sqrt((to[0] - from[0]) ** 2 + (to[1] - from[1]) ** 2) * 111000;
      totalDistance += dist;
      totalDuration += dist / 13.9;
    }

    return res.status(200).json({
      code: "Ok",
      trips: [
        {
          geometry: { type: "LineString", coordinates: allCoords },
          distance: Math.round(totalDistance),
          duration: Math.round(totalDuration),
          legs: points.map((_, i) => ({
            summary: `Leg ${i + 1}`,
            distance: Math.round(totalDistance / points.length),
            duration: Math.round(totalDuration / points.length),
          })),
          weight_name: "routability",
          weight: Math.round(totalDuration),
        },
      ],
      waypoints: points.map((p, i) => ({
        location: p,
        name: `Stop ${i + 1}`,
        waypoint_index: i,
        trips_index: 0,
      })),
      _fallback: true,
    });
  }

  try {
    const url = `https://api.mapbox.com/optimized-trips/v1/mapbox/${profile}/${coordinates}?geometries=geojson&access_token=${token}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Mapbox API returned ${response.status}`);
    }
    const data = await response.json();
    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate");
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch optimized trip", message: error.message });
  }
}
