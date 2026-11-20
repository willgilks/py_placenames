import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from constants import EXPLAINER_TEXT, KEY_TO_LANGUAGES, SIDEBAR_TEXT
from functions_plotting import build_gb_county_map

LANGUAGE_TO_KEY = {lang: code for code, lang in KEY_TO_LANGUAGES.items()}

# LOAD BASE DATA ####
DATA_DIR = Path("data")
county_plot_dat = pd.read_csv(DATA_DIR / "county_plot_dat.csv")
placename_strings_reference = pd.read_csv(DATA_DIR / "placename_strings_reference.csv")
matched_and_filtered_dat = pd.read_csv(DATA_DIR / "matched_and_filtered_dat.csv")

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

# Pre-build all origin figures once at startup
PREBUILT_MAPS: dict[str, go.Figure] = {
    origin: build_gb_county_map(
        county_data=county_plot_dat[
            county_plot_dat["origin"] == origin
        ].drop_duplicates(),
        ptitle=origin,
        gb_boundaries=gb_boundaries,
        hover_examples_max_lines=5,
    )
    for origin in ORIGINS
}


def _match_token_column(series: pd.Series, abbr: str) -> pd.Series:
    """Vectorized exact-token match on comma-separated abbreviation strings."""
    pattern = rf"(?:^|,\s*){re.escape(abbr)}(?:\s*,|$)"
    return series.fillna("").str.contains(pattern, regex=True)


def _filter_table_by_origin(df: pd.DataFrame, chosen: str) -> pd.DataFrame:
    """Filters a DataFrame by full origin name or comma-separated origin codes."""
    abbr = LANGUAGE_TO_KEY.get(chosen)

    if "origin" in df.columns:
        full_match = df["origin"] == chosen
        if full_match.any() or not abbr:
            return df[full_match]
        return df[_match_token_column(df["origin"], abbr)]

    if "comb_origins" in df.columns and abbr:
        return df[_match_token_column(df["comb_origins"], abbr)]

    return df


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
        ui.div(
            ui.markdown(SIDEBAR_TEXT),
            class_="text-muted small",
        ),
        width=260,
    ),
    ui.layout_columns(
        # Middle column: Map
        ui.card(
            ui.card_header(ui.output_text("single_map_title")),
            output_widget("single_origin_map"),
            full_screen=True,
        ),
        # Right column: Data selector, Download button & Table
        ui.card(
            ui.card_header("Data"),
            ui.div(
                ui.input_select(
                    "selected_table",
                    "Table",
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
            full_screen=True,
        ),
        col_widths=(6, 6),
    ),
    # Bottom section: Explainer / About text
    ui.card(
        ui.card_header("About"),
        ui.div(
            ui.markdown(EXPLAINER_TEXT),
            class_="small",
        ),
    ),
    title="British Place Name Origins",
)


# DEFINE SERVER ####
def server(input, output, session):
    map_widget = go.FigureWidget(PREBUILT_MAPS[DEFAULT_ORIGIN])

    @reactive.calc
    def chosen_origin() -> str:
        return input.selected_origin() or DEFAULT_ORIGIN

    @render.text
    def single_map_title() -> str:
        return f"Percentage of Place Names with {chosen_origin()} Origin"

    @render_widget
    def single_origin_map():
        return map_widget

    @reactive.effect
    def _update_map():
        target = PREBUILT_MAPS[chosen_origin()]
        with map_widget.batch_update():
            map_widget.data = []
            map_widget.add_traces(target.data)
            map_widget.layout.legend = target.layout.legend

    @reactive.calc
    def active_df() -> pd.DataFrame:
        raw_df = DOWNLOADABLE_TABLES[input.selected_table()]
        filtered = _filter_table_by_origin(raw_df, chosen_origin())
        return filtered.reset_index(drop=True)

    @render.data_frame
    def active_data_table():
        return render.DataGrid(active_df(), filters=True)

    @render.download(
        filename=lambda: f"{input.selected_table().lower().replace(' ', '_')}.csv"
    )
    def download_table_csv():
        yield active_df().to_csv(index=False)


app = App(app_ui, server)
