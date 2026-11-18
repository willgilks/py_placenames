from pathlib import Path

import geopandas as gpd
import pandas as pd
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from functions_plotting import (
    leaflet_multi_wrapper,
    make_leaflet_map,
    render_multi_map_grid,
)

# Load pre-computed datasets from ./data/
DATA_DIR = Path("data")
county_plot_dat = pd.read_csv(DATA_DIR / "county_plot_dat.csv")
gb_boundaries = gpd.read_file(DATA_DIR / "gb_boundaries.geojson")

ORIGINS = sorted(county_plot_dat["origin"].dropna().unique().tolist())

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Single Origin Explorer",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "selected_origin",
                    "Select Linguistic Origin:",
                    choices=ORIGINS,
                    selected=ORIGINS if ORIGINS else None,
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
                output_widget("single_origin_map"),
                full_screen=True,
            ),
        ),
    ),
    ui.nav_panel(
        "All Origins Grid",
        render_multi_map_grid([None] * len(ORIGINS)),
    ),
    title="GB Place Name Origins",
)


def server(input, output, session):
    @reactive.calc
    def filtered_origin_df() -> pd.DataFrame:
        return county_plot_dat[
            county_plot_dat["origin"] == input.selected_origin()
        ].drop_duplicates()

    @render.text
    def single_map_title() -> str:
        return f"Percentage of Place Names with {input.selected_origin()} Origin"

    @render_widget
    def single_origin_map():
        return make_leaflet_map(
            leaflet_county_data=filtered_origin_df(),
            ptitle=input.selected_origin(),
            gb_boundaries=gb_boundaries,
            fix_percent_scale=input.fix_scale(),
            zoom_restrict=False,
            starting_zoom=6,
            height="650px",
        )

    # Register all small-multiple maps for the "All Origins Grid" tab
    all_maps = leaflet_multi_wrapper(
        input_df=county_plot_dat,
        gb_boundaries=gb_boundaries,
    )

    for idx, m in enumerate(all_maps):

        def _register_map(map_obj, widget_id):
            def _map_widget():
                return map_obj

            _map_widget.__name__ = widget_id
            render_widget(_map_widget)

        _register_map(m, f"origin_map_{idx}")


app = App(app_ui, server)