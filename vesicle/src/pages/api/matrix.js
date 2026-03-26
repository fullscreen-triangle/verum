export default async function handler(req, res) {
  const { coordinates, profile = "driving" } = req.query;
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

  if (!coordinates) {
    return res.status(400).json({ error: "Missing coordinates. Use format: lon,lat;lon,lat;..." });
  }

  if (!token) {
    // Fallback: synthetic distance/duration matrix
    const points = coordinates.split(";");
    const n = points.length;
    const durations = [];
    const distances = [];

    for (let i = 0; i < n; i++) {
      const row_d = [];
      const row_dist = [];
      const [iLon, iLat] = points[i].split(",").map(Number);
      for (let j = 0; j < n; j++) {
        if (i === j) {
          row_d.push(0);
          row_dist.push(0);
        } else {
          const [jLon, jLat] = points[j].split(",").map(Number);
          const dist = Math.sqrt((jLon - iLon) ** 2 + (jLat - iLat) ** 2) * 111000;
          row_dist.push(Math.round(dist));
          row_d.push(Math.round(dist / 13.9)); // ~50 km/h
        }
      }
      durations.push(row_d);
      distances.push(row_dist);
    }

    return res.status(200).json({
      code: "Ok",
      durations,
      distances,
      sources: points.map((p) => {
        const [lon, lat] = p.split(",").map(Number);
        return { location: [lon, lat], name: "" };
      }),
      destinations: points.map((p) => {
        const [lon, lat] = p.split(",").map(Number);
        return { location: [lon, lat], name: "" };
      }),
      _fallback: true,
    });
  }

  try {
    const url = `https://api.mapbox.com/directions-matrix/v1/mapbox/${profile}/${coordinates}?access_token=${token}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Mapbox API returned ${response.status}`);
    }
    const data = await response.json();
    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate");
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch matrix data", message: error.message });
  }
}
