import ReactMapGL from "react-map-gl";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import data from "../public/data.json";
import InfoPanel from "../components/InfoPanel";

function Map({ width, height, viewport, setViewport, token }) {
  const layers = [
    new ScatterplotLayer({
      id: "scatterplot-layer",
      data,
      filled: true,
      radiusScale: 30,
      radiusMinPixels: 1,
      radiusMaxPixels: 2,
      getPosition: (data) => [data.longitude, data.latitude],
      getFillColor: (data) => {
        if (data.deaths > 0) {
          return [2526, 18, 25];
        } else if (data.major_injuries > 0) {
          return [255, 69, 0];
        } else if (data.minor_injuries >= 0) {
          return [255, 204, 51];
        }
      },
      pickable: true,
    }),
    new HeatmapLayer({
      id: "heatmap-layer",
      data: data,
      getPosition: (data) => [data.longitude, data.latitude],
      getWeight: (data) =>
        data.deaths * 2 +
        data.major_injuries * 0.75 +
        data.minor_injuries * 0.5,
      radiusPixels: 20,
    }),
  ];

  function getTooltip({ object }) {
    if (!object) {
      return null;
    }
    const deaths = object.deaths;
    const major = object.major_injuries;
    const minor = object.minor_injuries;

    if (deaths === 0 && major === 0 && minor === 0) {
      return `No Injuries or Deaths`;
    } else {
      return `\
    ${deaths} Deaths,
    ${major} Major Injuries,
    ${minor} Minor Injuries,
    `;
    }
  }

  return (
    <>
      <ReactMapGL
        {...viewport}
        width={width}
        height={height}
        mapStyle="mapbox://styles/przemekp/ckddfi7im0quk1iqf0jui75m0"
        mapboxApiAccessToken={token}
        onViewportChange={(viewport) => {
          setViewport(viewport);
        }}
      >
        <DeckGL
          initialViewState={viewport}
          layers={layers}
          getTooltip={getTooltip}
        >
          )}
          <InfoPanel
            style={{
              width: "100px",
              height: "100px",
              background: "red",
              position: "absolute",
            }}
          />
        </DeckGL>
      </ReactMapGL>
    </>
  );
}

export default Map;
