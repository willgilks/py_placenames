from __future__ import annotations

import branca.colormap as cm
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipyleaflet import GeoJSON, LegendControl, Map, Popup, ZoomControl
from ipywidgets import HTML, Layout


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

    return (
        [round(float(b), 6) for b in bins]
        if len(bins) >= 2
        else [0.0, max(1.0, float(max_val))]
    )


def create_base_map(
    zoom_restrict: bool = False,
    starting_zoom: int = 6,
    height: str = "650px",
) -> Map:
    """Creates a single reusable ipyleaflet Map with exactly one zoom control."""
    m = Map(
        center=(55.25, -2.75),
        zoom=starting_zoom,
        min_zoom=starting_zoom if zoom_restrict else 5,
        max_zoom=starting_zoom if zoom_restrict else 10,
        zoom_control=False,
        dragging=not zoom_restrict,
        scroll_wheel_zoom=not zoom_restrict,
        attribution_control=False,
        layout=Layout(width="100%", height=height),
    )
    m.clear_controls()
    if not zoom_restrict:
        m.add_control(ZoomControl(position="topleft"))
    return m


def update_leaflet_map(
    m: Map,
    leaflet_county_data: pd.DataFrame,
    ptitle: str = "",
    gb_boundaries: gpd.GeoDataFrame | None = None,
    fix_percent_scale: bool = False,
) -> None:
    """Updates an existing ipyleaflet Map's GeoJSON layer and Legend in place."""
    if gb_boundaries is None:
        return

    base_geo = gb_boundaries[["name", "geometry"]].copy()
    boundaries_with_data = base_geo.merge(
        leaflet_county_data, left_on="name", right_on="county_name", how="left"
    )
    boundaries_with_data["pc"] = boundaries_with_data["pc"].fillna(0.0)

    domain_vals = boundaries_with_data["pc"].astype(float)
    max_domain_val = float(domain_vals.max()) if len(domain_vals) > 0 else 0.0
    min_domain_val = float(domain_vals.min()) if len(domain_vals) > 0 else 0.0

    if fix_percent_scale:
        max_upper = max(100.0, max_domain_val)
        map_bins = list(np.linspace(0.0, max_upper, 6))
    else:
        map_bins = _pretty_bins(min_domain_val, max_domain_val, n=5)

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

    boundaries_with_data["fillColor"] = boundaries_with_data["pc"].map(get_fill_color)

    popup_htmls = []
    for _, row in boundaries_with_data.iterrows():
        pc_val = float(row["pc"])
        n_places = round(float(row["n_places"])) if pd.notna(row.get("n_places")) else 0
        n_county = (
            round(float(row["n_county_places"]))
            if pd.notna(row.get("n_county_places"))
            else 0
        )

        county_name_text = f"<b>{row['name']}</b><br>"
        origin_text = f"<span>{round(pc_val)}% {ptitle} ({n_places}/{n_county})</span>"

        example_text = ""
        raw_matches = row.get("strings_and_matches")
        if pd.notna(raw_matches) and str(raw_matches).strip() != "":
            hover_subtext = str(raw_matches).replace(".", "<br>")
            example_text = f"<br><span style='font-size:12px;'>{hover_subtext}</span>"

        popup_htmls.append(f"{county_name_text}{origin_text}{example_text}")

    boundaries_with_data["popup_html"] = popup_htmls

    geojson_dict = boundaries_with_data[
        ["name", "pc", "fillColor", "popup_html", "geometry"]
    ].__geo_interface__

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

    def on_feature_click(event: str, feature: dict, **kwargs) -> None:
        coords = kwargs.get("coordinates", (55.25, -2.75))
        html_widget = HTML(
            value=(
                "<div style='max-height:150px; max-width:180px; overflow-y:auto;'>"
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

    # Remove old GeoJSON/Popup layers and old LegendControls before adding updated ones
    m.clear_layers()
    m.add_layer(geo_layer)

    for ctrl in list(m.controls):
        if isinstance(ctrl, LegendControl):
            m.remove_control(ctrl)

    legend_dict = {}
    for i in range(len(map_bins) - 1):
        label = f"{round(map_bins[i], 1)} – {round(map_bins[i + 1], 1)}"
        legend_dict[label] = bin_colors[i]
    legend_dict["0 (None)"] = "#FFFFFF"

    m.add_control(
        LegendControl(
            legend_dict,
            title=ptitle or "Percentage (%)",
            position="topright",
        )
    )
