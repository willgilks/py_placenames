from pathlib import Path

import geopandas as gpd
import pandas as pd
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from constants import EXPLAINER_TEXT, KEY_TO_LANGUAGES, SIDEBAR_TEXT
from functions_plotting import build_gb_county_map

LANGUAGE_TO_KEY = {lang: code for code, lang in KEY_TO_LANGUAGES.items()}

# Load base app data
DATA_DIR = Path("data")
county_plot_dat = pd.read_csv(DATA_DIR / "county_plot_dat.csv")

placename_strings_reference = pd.read_csv(DATA_DIR / "placename_strings_reference.csv")
matched_and_filtered_dat = pd.read_csv(DATA_DIR / "matched_and_filtered_dat.csv")
print("Columns:", matched_and_filtered_dat.columns.tolist())


gb_boundaries = gpd.read_file(DATA_DIR / "gb_boundaries.geojson")[
    ["name", "geometry"]
].assign(
    geometry=lambda df: df.geometry.simplify(tolerance=0.004, preserve_topology=True)
)

ORIGINS = sorted(county_plot_dat["origin"].dropna().unique().tolist())
DEFAULT_ORIGIN = next(iter(ORIGINS), "")

DOWNLOADABLE_TABLES: dict[str, pd.DataFrame] = {
    "Origins and strings": placename_strings_reference,
    "County level data": county_plot_dat,
    "Place level data": matched_and_filtered_dat,
}


# DEFINE UI ####
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


# DEFINE SERVER ####
def server(input, output, session):
    @reactive.calc
    def filtered_origin_df() -> pd.DataFrame:
        chosen = input.selected_origin() or DEFAULT_ORIGIN
        return county_plot_dat[county_plot_dat["origin"] == chosen].drop_duplicates()

    @render.text
    def single_map_title() -> str:
        chosen = input.selected_origin() or DEFAULT_ORIGIN
        return f"Percentage of Place Names with {chosen} Origin"

    @render_widget
    def single_origin_map():
        chosen = input.selected_origin() or DEFAULT_ORIGIN
        return build_gb_county_map(
            county_data=filtered_origin_df(),
            ptitle=chosen,
            gb_boundaries=gb_boundaries,
            hover_examples_max_lines=5,
        )

    @reactive.calc
    def active_df() -> pd.DataFrame:
        df = DOWNLOADABLE_TABLES[input.selected_table()].copy()

        if "origin" in df.columns:
            chosen = input.selected_origin() or DEFAULT_ORIGIN
            abbr = LANGUAGE_TO_KEY.get(chosen)

            # Exact full-name match (for county-level table)
            full_match = df["origin"] == chosen

            # Exact token match on comma-separated codes (for origins & strings table)
            if abbr and not full_match.any():
                abbr_match = (
                    df["origin"]
                    .fillna("")
                    .apply(
                        lambda cell: (
                            abbr in [token.strip() for token in str(cell).split(",")]
                        )
                    )
                )
                df = df[abbr_match]
            else:
                df = df[full_match]

        if "comb_origins" in df.columns:
            chosen = input.selected_origin() or DEFAULT_ORIGIN
            abbr = LANGUAGE_TO_KEY.get(chosen)

            abbr_match = (
                df["comb_origins"]
                .fillna("")
                .apply(
                    lambda cell: (
                        abbr in [token.strip() for token in str(cell).split(",")]
                    )
                )
            )
            df = df[abbr_match]

        return df.reset_index(drop=True)

    @render.data_frame
    def active_data_table():
        return render.DataGrid(active_df(), filters=True)

    @render.download(
        filename=lambda: f"{input.selected_table().lower().replace(' ', '_')}.csv"
    )
    def download_table_csv():
        yield active_df().to_csv(index=False)


app = App(app_ui, server)
