import useSWR from "swr";

const fetcher = (url) => fetch(url).then((res) => res.json());

export function useWeather() {
  return useSWR("/api/weather?lat=48.1351&lon=11.5820", fetcher, {
    refreshInterval: 60000,
    revalidateOnFocus: false,
  });
}

export function useTraffic() {
  return useSWR("/api/traffic?bbox=11.4,48.0,11.7,48.2", fetcher, {
    refreshInterval: 120000,
    revalidateOnFocus: false,
  });
}

export function useCellTowers() {
  return useSWR("/api/celltowers", fetcher, {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
  });
}
