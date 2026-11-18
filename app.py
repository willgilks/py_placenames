from pathlib import Path

import geopandas as gpd
import pandas as pd
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from functions_plotting import make_leaflet_map, make_small_multiples_figure

# Load pre-computed datasets from ./data/ and keep only needed columns
DATA_DIR = Path("data")
county_plot_dat = pd.read_csv(DATA_DIR / "county_plot_dat.csv")
gb_boundaries = (
    gpd.read_file(DATA_DIR / "gb_boundaries.geojson")[["name", "geometry"]]
    .assign(geometry=lambda df: df.geometry.simplify(tolerance=0.004, preserve_topology=True))
)

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
        ),
    ),
    ui.nav_panel(
        "All Origins Grid",
        ui.uiOutput("all_origins_grid_img"),
    ),
    title="GB Place Name Origins",
)


def server(input, output, session):
    @reactive.calc
    def filtered_origin_df() -> pd.DataFrame:
        chosen = input.selected_origin() or ORIGINS[0]
        return county_plot_dat[county_plot_dat["origin"] == chosen].drop_duplicates()

    @render.text
    def single_map_title() -> str:
        chosen = input.selected_origin() or ORIGINS[0]
        return f"Percentage of Place Names with {chosen} Origin"

    @render_widget
    def single_origin_map():
        chosen = input.selected_origin() or ORIGINS[0]
        return make_leaflet_map(
            leaflet_county_data=filtered_origin_df(),
            ptitle=chosen,
            gb_boundaries=gb_boundaries,
            fix_percent_scale=input.fix_scale(),
            zoom_restrict=False,
            starting_zoom=6,
            height="650px",
        )

    @render.ui
    def all_origins_grid_img():
        b64_png = make_small_multiples_figure(county_plot_dat, gb_boundaries)
        return ui.tags.img(
            src=f"data:image/png;base64,{b64_png}",
            style="width: 100%; height: auto; border-radius: 6px;",
        )


app = App(app_ui, server)
