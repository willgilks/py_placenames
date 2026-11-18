import json

import branca.colormap as cm
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipyleaflet import GeoJSON, LegendControl, Map, Popup
from ipywidgets import HTML, Layout
from shiny import ui

from constants import GB_BOUNDARIES


def _pretty_bins(min_val: float, max_val: float, n: int = 5) -> list[float]:
    """Replicates R's pretty(c(min_val, max_val), n = 5) bin breakpoints."""
    if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
        upper = max(1.0, float(max_val) if not np.isnan(max_val) else 1.0)
        return [0.0, upper]

    raw_step = (max_val - min_val) / max(n, 1)
    mag = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / mag

    if residual <= 1.5:
        step = 1.0 * mag
    elif residual <= 3.0:
        step = 2.0 * mag
    elif residual <= 7.0:
        step = 5.0 * mag
    else:
        step = 10.0 * mag

    start = np.floor(min_val / step) * step
    stop = np.ceil(max_val / step) * step
    bins = list(np.arange(start, stop + step * 0.5, step))

    if min_val >= 0:
        bins = [b for b in bins if b >= 0]

    return [round(float(b), 6) for b in bins] if len(bins) >= 2 else [0.0, max(1.0, float(max_val))]


def make_leaflet_map(
    leaflet_county_data: pd.DataFrame,
    ptitle: str = "",
    gb_boundaries: gpd.GeoDataFrame = GB_BOUNDARIES,
    zoom_level: int = 6,
    is_percentage: bool = True,
    fix_percent_scale: bool = False,
    zoom_restrict: bool = False,
    starting_zoom: int = 6,
    height: str = "550px",
) -> Map:
    """Plots GB counties coloured by percentage using ipyleaflet."""
    # Bind to shapefile
    boundaries_with_data = gb_boundaries.merge(
        leaflet_county_data, left_on="name", right_on="county_name", how="left"
    ).copy()
    boundaries_with_data["pc"] = boundaries_with_data["pc"].fillna(0.0)

    # Transform CRS to WGS84
    map_data_tf = boundaries_with_data.to_crs(epsg=4326)

    # Set colour scale ranges
    domain_vals = map_data_tf["pc"].astype(float)
    max_domain_val = float(domain_vals.max()) if len(domain_vals) > 0 else 0.0
    min_domain_val = float(domain_vals.min()) if len(domain_vals) > 0 else 0.0

    if fix_percent_scale:
        max_upper = max(100.0, max_domain_val)
        map_bins = list(np.linspace(0.0, max_upper, 6))
    else:
        map_bins = _pretty_bins(min_domain_val, max_domain_val, n=5)

    # Build YlOrRd stepped palette matching R's colorBin("YlOrRd", bins=map_bins)
    n_bins = max(len(map_bins) - 1, 1)
    cmap = plt.get_cmap("YlOrRd", n_bins)
    bin_colors = [mcolors.to_hex(cmap(i)) for i in range(n_bins)]
    step_cmap = cm.StepColormap(
        colors=bin_colors,
        index=map_bins,
        vmin=map_bins[0],
        vmax=map_bins[-1],
    )

    def get_fill_color(val: float) -> str:
        if pd.isna(val) or val == 0:
            return "#FFFFFF"
        return step_cmap(val)

    # Build GeoJSON features with pre-computed fill colours and HTML popups
    features = []
    for _, row in map_data_tf.iterrows():
        pc_val = float(row["pc"]) if pd.notna(row["pc"]) else 0.0
        n_places = round(float(row["n_places"])) if pd.notna(row.get("n_places")) else 0
        n_county = (
            round(float(row["n_county_places"]))
            if pd.notna(row.get("n_county_places"))
            else 0
        )

        county_name_text = f"<span>{row['name']}</span><br>"
        origin_and_percent_text = (
            f"<span>{round(pc_val)}% {ptitle} ({n_places}/{n_county})</span>"
        )

        example_text = ""
        raw_matches = row.get("strings_and_matches")
        if pd.notna(raw_matches) and str(raw_matches).strip() != "":
            hover_subtext = str(raw_matches).replace(".", "<br>")
            example_text = f"<br><span style='font-size:12px;'>{hover_subtext}</span>"

        popup_html = f"{county_name_text}{origin_and_percent_text}{example_text}"

        geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())["features"][0][
            "geometry"
        ]
        features.append(
            {
                "type": "Feature",
                "geometry": geom_json,
                "properties": {
                    "name": row["name"],
                    "pc": pc_val,
                    "fillColor": get_fill_color(pc_val),
                    "popup_html": popup_html,
                },
            }
        )

    geojson_dict = {"type": "FeatureCollection", "features": features}

    m = Map(
        center=(55.25, -2.75),
        zoom=starting_zoom,
        min_zoom=starting_zoom if zoom_restrict else 5,
        max_zoom=starting_zoom if zoom_restrict else 10,
        zoom_control=not zoom_restrict,
        dragging=not zoom_restrict,
        scroll_wheel_zoom=not zoom_restrict,
        attribution_control=False,
        layout=Layout(width="100%", height=height),
    )

    # Style callback reads precomputed fillColor per county polygon
    def style_callback(feature: dict) -> dict:
        return {
            "fillColor": feature["properties"]["fillColor"],
            "fillOpacity": 1.0,
            "weight": 0.5,
            "color": "black",
        }

    geo_layer = GeoJSON(
        data=geojson_dict,
        style_callback=style_callback,
    )

    # Click popup handler matching R leaflet addPolygons(popup=...)
    def on_feature_click(event: str, feature: dict, **kwargs) -> None:
        coords = kwargs.get("coordinates", (55.25, -2.75))
        html_widget = HTML(
            value=(
                f"<div style='max-height:150px; max-width:160px; overflow-y:auto;'>"
                f"{feature['properties']['popup_html']}</div>"
            )
        )
        popup = Popup(
            location=coords,
            child=html_widget,
            close_button=True,
            auto_close=True,
            close_on_escape_key=True,
            auto_pan=False,
        )
        m.add_layer(popup)

    geo_layer.on_click(on_feature_click)
    m.add_layer(geo_layer)

    # Add legend in top-right corner
    legend_dict = {}
    for i in range(len(map_bins) - 1):
        label = f"{round(map_bins[i], 1)} – {round(map_bins[i + 1], 1)}"
        legend_dict[label] = bin_colors[i]
    legend_dict["0 (None)"] = "#FFFFFF"

    legend = LegendControl(
        legend_dict,
        title=ptitle or "Percentage (%)",
        position="topright",
    )
    m.add_control(legend)

    return m


def leaflet_multi_wrapper(
    input_df: pd.DataFrame,
    gb_boundaries: gpd.GeoDataFrame = GB_BOUNDARIES,
) -> list[Map]:
    """Builds a list of locked zoom-5 ipyleaflet maps, one per language origin."""
    origin_names = sorted(input_df["origin"].dropna().unique())
    maps: list[Map] = []

    for origin_name in origin_names:
        df = (
            input_df[input_df["origin"] == origin_name][
                [
                    "county_name",
                    "origin",
                    "pc",
                    "strings_and_matches",
                    "n_county_places",
                    "n_places",
                ]
            ]
            .loc[lambda d: np.isfinite(d["pc"])]
            .drop_duplicates()
            .copy()
        )

        m = make_leaflet_map(
            leaflet_county_data=df,
            ptitle=str(origin_name),
            gb_boundaries=gb_boundaries,
            fix_percent_scale=False,
            zoom_restrict=True,
            starting_zoom=5,
            height="380px",
        )
        maps.append(m)

    return maps


def render_multi_map_grid(maps: list[Map]) -> ui.Tag:
    """Renders the list of maps in a 3-column CSS grid matching leafsync::sync(ncol = 3)."""
    from shinywidgets import output_widget

    cards = [
        ui.div(
            output_widget(f"origin_map_{idx}"),
            style="height: 380px; width: 100%; background: #add8e6;",
        )
        for idx in range(len(maps))
    ]

    return ui.TagList(
        ui.tags.style(
            """
            .leaflet-container { background: #add8e6 !important; }
            .multi-map-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                width: 100%;
                margin: 10px 0;
            }
            """
        ),
        ui.div(*cards, class_="multi-map-grid"),
    )