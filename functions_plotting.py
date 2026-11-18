from __future__ import annotations

import branca.colormap as cm
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


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


def make_county_choropleth(
    county_data: pd.DataFrame,
    ptitle: str,
    gb_boundaries: gpd.GeoDataFrame,
    geojson_obj: dict,
    fix_percent_scale: bool = False,
) -> go.Figure:
    """Plots a clean, tile-free GB county choropleth using Plotly."""
    base_geo = gb_boundaries[["name"]].copy()
    merged = base_geo.merge(
        county_data, left_on="name", right_on="county_name", how="left"
    )
    merged["pc"] = merged["pc"].fillna(0.0)

    domain_vals = merged["pc"].astype(float)
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
        vmin=map_bins,
        vmax=map_bins[-1],
    )

    def get_fill_color(val: float) -> str:
        if pd.isna(val) or val == 0:
            return "#FFFFFF"
        return step_cmap(val)

    merged["fill_color"] = merged["pc"].map(get_fill_color)

    # Build rich hover labels matching your original R Leaflet popups
    hover_texts = []
    for _, row in merged.iterrows():
        pc_val = float(row["pc"])
        n_places = round(float(row["n_places"])) if pd.notna(row.get("n_places")) else 0
        n_county = (
            round(float(row["n_county_places"]))
            if pd.notna(row.get("n_county_places"))
            else 0
        )

        header = (
            f"<b>{row['name']}</b><br>{round(pc_val)}% {ptitle} ({n_places}/{n_county})"
        )
        raw_matches = row.get("strings_and_matches")
        if pd.notna(raw_matches) and str(raw_matches).strip() != "":
            matches_lines = "<br>".join(
                part.strip() for part in str(raw_matches).split(".") if part.strip()
            )
            hover_texts.append(f"{header}<br><br>{matches_lines}")
        else:
            hover_texts.append(header)

    merged["hover_text"] = hover_texts

    # Assign each county a discrete integer bucket for exact stepped legend colours
    unique_colors = ["#FFFFFF"] + bin_colors
    color_to_idx = {c: i for i, c in enumerate(unique_colors)}
    merged["color_idx"] = merged["fill_color"].map(color_to_idx)

    # Build discrete Plotly colorscale
    n_total = len(unique_colors)
    discrete_scale = []
    for i, hex_col in enumerate(unique_colors):
        discrete_scale.append([i / n_total, hex_col])
        discrete_scale.append([(i + 1) / n_total, hex_col])

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson_obj,
            locations=merged["name"],
            featureidkey="properties.name",
            z=merged["color_idx"],
            zmin=0,
            zmax=n_total,
            colorscale=discrete_scale,
            showscale=False,
            text=merged["hover_text"],
            hoverinfo="text",
            marker_line_color="black",
            marker_line_width=0.6,
        )
    )

    # Add a clean, readable categorical legend for your exact bins + 0 (None)
    legend_entries = [("0% (None)", "#FFFFFF")] + [
        (f"{round(map_bins[i], 1)} – {round(map_bins[i + 1], 1)}%", bin_colors[i])
        for i in range(len(map_bins) - 1)
    ]
    for label, hex_col in reversed(legend_entries):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": hex_col,
                    "symbol": "square",
                    "line": {"color": "black", "width": 0.8},
                },
                name=label,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="#add8e6",
    )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="#add8e6",
        plot_bgcolor="#add8e6",
        height=680,
        xaxis={"visible": False},
        yaxis={"visible": False},
        legend={
            "title": {"text": f"<b>{ptitle}</b>"},
            "x": 0.98,
            "y": 0.98,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 0.8,
        },
        hoverlabel={
            "bgcolor": "white",
            "font_size": 12,
            "align": "left",
        },
    )

    return fig
