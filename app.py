from pathlib import Path

import geopandas as gpd
import pandas as pd
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from functions_plotting import build_gb_county_map

# Load base app data
DATA_DIR = Path("data")
county_plot_dat = pd.read_csv(DATA_DIR / "county_plot_dat.csv")
gb_boundaries = gpd.read_file(DATA_DIR / "gb_boundaries.geojson")[
    ["name", "geometry"]
].assign(
    geometry=lambda df: df.geometry.simplify(tolerance=0.004, preserve_topology=True)
)

ORIGINS = sorted(county_plot_dat["origin"].dropna().unique().tolist())
DEFAULT_ORIGIN = ORIGINS if ORIGINS else ""

# define ui
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "selected_origin",
            "Origin",
            choices=ORIGINS,
            selected=DEFAULT_ORIGIN,
        ),
        ui.input_switch(
            "fix_scale",
            "Fix scale (0–100%)",
            value=False,
        ),
        ui.hr(),
        ui.p(
            "<explainer text here>",
            class_="text-muted small",
        ),
    ),
    ui.card(
        ui.card_header(ui.output_text("single_map_title")),
        output_widget("single_origin_map"),
        full_screen=True,
    ),
    title="<title here>",
)


# define reactive
def server(input, output, session):
    @reactive.calc

    # filter for selected origin
    def filtered_origin_df() -> pd.DataFrame:
        chosen = input.selected_origin() or DEFAULT_ORIGIN
        return county_plot_dat[county_plot_dat["origin"] == chosen].drop_duplicates()

    @render.text

    # set title with origin
    def single_map_title() -> str:
        chosen = input.selected_origin() or DEFAULT_ORIGIN
        return f"Percentage of Place Names with {chosen} Origin"

    @render_widget
    # make map
    def single_origin_map():
        chosen = input.selected_origin() or DEFAULT_ORIGIN
        return build_gb_county_map(
            county_data=filtered_origin_df(),
            ptitle=chosen,
            gb_boundaries=gb_boundaries,
            fix_percent_scale=input.fix_scale(),
        )


app = App(app_ui, server)
