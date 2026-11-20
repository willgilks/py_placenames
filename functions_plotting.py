from __future__ import annotations

import branca.colormap as cm
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# make colour bins for map plot values
def _pretty_bins(min_val: float, max_val: float, n: int = 5) -> list[float]:
    """make colour bin breakpoints."""
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


def _extract_poly_coords(geom) -> tuple[list[float | None], list[float | None]]:
    """get co-ordinate lists separated by None for MultiPolygons."""
    xs: list[float | None] = []
    ys: list[float | None] = []

    if geom is None or geom.is_empty:
        return xs, ys

    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        x_ext, y_ext = poly.exterior.xy
        xs.extend(x_ext)
        ys.extend(y_ext)
        xs.append(None)
        ys.append(None)

    return xs, ys


# make british county map object
def build_gb_county_map(
    county_data: pd.DataFrame,
    ptitle: str,
    gb_boundaries: gpd.GeoDataFrame,
    hover_examples_max_lines: int | None = 10,
) -> go.Figure:
    """Plots a clean GB county choropleth map with zero external map tiles."""
    merged = gb_boundaries[["name", "geometry"]].merge(
        county_data, left_on="name", right_on="county_name", how="left"
    )
    merged["pc"] = merged["pc"].fillna(0.0)

    domain_vals = merged["pc"].astype(float)
    max_domain_val = float(domain_vals.max()) if len(domain_vals) > 0 else 0.0
    min_domain_val = float(domain_vals.min()) if len(domain_vals) > 0 else 0.0

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
            return "#F8F9FA"
        return step_cmap(val)

    fig = go.Figure()

    for _, row in merged.iterrows():
        xs, ys = _extract_poly_coords(row["geometry"])
        if not xs:
            continue

        pc_val = float(row["pc"])
        fill_col = get_fill_color(pc_val)
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
            all_parts = [
                part.strip() for part in str(raw_matches).split(".") if part.strip()
            ]

            if pd.notna(hover_examples_max_lines):
                max_lines = int(hover_examples_max_lines)

            else:
                max_lines = 10

            if len(all_parts) > max_lines:
                shown_parts = all_parts[:max_lines]
                shown_parts.append(f"<i>... and {len(all_parts) - max_lines} more</i>")
            else:
                shown_parts = all_parts

            matches_lines = "<br>".join(shown_parts)
            hover_text = f"{header}<br><br>{matches_lines}"
        else:
            hover_text = header

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill_col,
                line={"color": "#333333", "width": 0.6},
                text=hover_text,
                hoverinfo="text",
                hoveron="fills",
                showlegend=False,
            )
        )

    # Categorical legend entries
    legend_entries = [("0% (None)", "#F8F9FA")] + [
        (f"{round(map_bins[i], 1)}–{round(map_bins[i + 1], 1)}%", bin_colors[i])
        for i in range(len(map_bins) - 1)
    ]
    for label, hex_col in reversed(legend_entries):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 13,
                    "color": hex_col,
                    "symbol": "square",
                    "line": {"color": "#333333", "width": 0.8},
                },
                name=label,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        autosize=True,
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        dragmode="pan",
        xaxis={
            "visible": False,
            "range": [-8.5, 3.0],
            "autorange": False,
            "fixedrange": False,
        },
        yaxis={
            "visible": False,
            "range": [49.8, 61.0],
            "autorange": False,
            "scaleanchor": "x",
            "scaleratio": 1.72,
            "fixedrange": False,
        },
        legend={
            # "title": {"text": f"<b>{ptitle}</b>"},
            "title": {"text": f"<b>{ptitle}</b>", "font": {"size": 11}},
            "font": {"size": 10},
            # "x": 0.98,
            # "y": 0.98,
            "x": 0.99,
            "y": 0.99,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": "#333333",
            "borderwidth": 0.8,
        },
        hoverlabel={
            "bgcolor": "white",
            # "font_size": 12,
            "font_size": 11,
            "align": "left",
        },
    )

    # return go.FigureWidget(fig)
    return fig
