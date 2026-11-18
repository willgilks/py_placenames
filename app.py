from pathlib import Path

import geopandas as gpd
import pandas as pd
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from functions_plotting import create_base_map, update_leaflet_map

# Load pre-computed datasets from ./data/
DATA_DIR = Path("data")
county_plot_dat = pd.read_csv(DATA_DIR / "county_plot_dat.csv")
gb_boundaries = gpd.read_file(DATA_DIR / "gb_boundaries.geojson")[
    ["name", "geometry"]
].assign(
    geometry=lambda df: df.geometry.simplify(tolerance=0.004, preserve_topology=True)
)

ORIGINS = sorted(county_plot_dat["origin"].dropna().unique().tolist())

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "selected_origin",
            "Select Linguistic Origin:",
            choices=ORIGINS,
            selected=ORIGINS[0] if ORIGINS else None,
        ),
        ui.input_switch(
            "fix_scale",
            "Fix percentage scale (0–100%)",
            value=False,
        ),
        ui.hr(),
        ui.p(
            "Click any county polygon to view matching place name elements and examples.",
            class_="text-muted small",
        ),
    ),
    ui.card(
        ui.card_header(ui.output_text("single_map_title")),
        ui.tags.style(".leaflet-container { background: #add8e6 !important; }"),
        output_widget("single_origin_map"),
        full_screen=True,
    ),
    title="GB Place Name Origins",
)


def server(input, output, session):
    # Create the map widget once per user session
    map_widget = create_base_map(
        zoom_restrict=False,
        starting_zoom=6,
        height="650px",
    )

    @render_widget
    def single_origin_map():
        return map_widget

    @reactive.calc
    def filtered_origin_df() -> pd.DataFrame:
        chosen = input.selected_origin() or ORIGINS[0]
        return county_plot_dat[county_plot_dat["origin"] == chosen].drop_duplicates()

    @render.text
    def single_map_title() -> str:
        chosen = input.selected_origin() or ORIGINS[0]
        return f"Percentage of Place Names with {chosen} Origin"

    @reactive.effect
    def _sync_map_data():
        chosen = input.selected_origin() or ORIGINS[0]
        update_leaflet_map(
            m=map_widget,
            leaflet_county_data=filtered_origin_df(),
            ptitle=chosen,
            gb_boundaries=gb_boundaries,
            fix_percent_scale=input.fix_scale(),
        )


app = App(app_ui, server)
