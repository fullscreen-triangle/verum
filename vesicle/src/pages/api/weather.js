export default async function handler(req, res) {
  const lat = req.query.lat || 48.1351;
  const lon = req.query.lon || 11.5820;
  const key = process.env.OPENWEATHERMAP_API_KEY;

  if (!key) {
    return res.status(200).json({
      main: { temp: 14.2, humidity: 72, pressure: 1013 },
      wind: { speed: 3.4, deg: 220 },
      weather: [{ main: "Clouds", description: "overcast clouds", icon: "04d" }],
      clouds: { all: 85 },
      rain: null,
      visibility: 10000,
      _fallback: true,
    });
  }

  try {
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${key}&units=metric`;
    const response = await fetch(url);
    const data = await response.json();
    res.setHeader("Cache-Control", "s-maxage=60, stale-while-revalidate");
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch weather data" });
  }
}
