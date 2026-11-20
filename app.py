import io
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from constants import EXPLAINER_TEXT, SIDEBAR_TEXT
from functions_plotting import build_gb_county_map

# Load base app data
DATA_DIR = Path("data")
county_plot_dat = pd.read_csv(DATA_DIR / "county_plot_dat.csv")

placename_strings_reference = pd.read_csv(DATA_DIR / "placename_strings_reference.csv")


gb_boundaries = gpd.read_file(DATA_DIR / "gb_boundaries.geojson")[
    ["name", "geometry"]
].assign(
    geometry=lambda df: df.geometry.simplify(tolerance=0.004, preserve_topology=True)
)

ORIGINS = sorted(county_plot_dat["origin"].dropna().unique().tolist())
DEFAULT_ORIGIN = ORIGINS if ORIGINS else ""

DOWNLOADABLE_TABLES: dict[str, pd.DataFrame] = {
    "Origins and strings": placename_strings_reference,
    # "Place names data": placenames_dat_raw,
    "County level data": county_plot_dat,
}
# define ui
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "selected_origin",
            "Origin",
            choices=ORIGINS,
            selected=DEFAULT_ORIGIN,
        ),
        ui.hr(),
        ui.p(
            ui.markdown(SIDEBAR_TEXT),
            class_="text-muted small",
        ),
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Map",
            ui.card_header(ui.output_text("single_map_title")),
            output_widget("single_origin_map"),
        ),
        ui.nav_panel(
            "About",
            ui.div(
                ui.markdown(EXPLAINER_TEXT),
                class_="p-3",
            ),
        ),
        ui.nav_panel(
            "Data",
            ui.div(
                ui.input_select(
                    "selected_table",
                    "Dataset",
                    choices=list(DOWNLOADABLE_TABLES.keys()),
                ),
                ui.download_button(
                    "download_table_csv",
                    "Download CSV",
                    class_="btn-outline-primary mb-3",
                ),
                class_="d-flex align-items-end gap-3 mb-2",
            ),
            ui.output_data_frame("active_data_table"),
        ),
        id="main_tabs",
        full_screen=True,
    ),
    title="British Place Name Origins",
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
            hover_examples_max_lines=5,
        )

    @reactive.calc
    # table for downloading
    def filtered_active_df() -> pd.DataFrame:
        df = DOWNLOADABLE_TABLES[input.selected_table()]
        # Case-insensitive check for an 'origin' column
        origin_cols = [c for c in df.columns if c.lower() == "origin"]
        if origin_cols:
            col = origin_cols[0]
            return df[df[col] == input.selected_origin()].copy()
        return df

    @render.data_frame
    def active_data_table():
        return render.DataGrid(filtered_active_df(), filters=True)

    @render.download(
        filename=lambda: (
            f"{input.selected_table().lower().replace(' ', '_')}_"
            f"{input.selected_origin().lower().replace(' ', '_')}.csv"
        )
    )
    def download_table_csv():
        buf = io.StringIO()
        filtered_active_df().to_csv(buf, index=False)
        yield buf.getvalue()


app = App(app_ui, server)
