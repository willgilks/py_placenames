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

PREBUILT_MAPS: dict[str, go.Figure] = {}


def get_origin_map(origin: str) -> go.Figure:
    """Builds and caches the Plotly figure for a given origin on first request."""
    if origin not in PREBUILT_MAPS:
        PREBUILT_MAPS[origin] = build_gb_county_map(
            county_data=county_plot_dat[
                county_plot_dat["origin"] == origin
            ].drop_duplicates(),
            ptitle=origin,
            gb_boundaries=gb_boundaries,
            hover_examples_max_lines=5,
        )
    return PREBUILT_MAPS[origin]


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
        open="open",  # Keeps Origin selector visible at top on mobile screens
    ),
    ui.tags.style(
        """
        @media (max-width: 767.98px) {
            /* Fit map comfortably inside a phone viewport so users can scroll past it */
            .js-plotly-plot, .plotly, .plot-container {
                max-height: 460px !important;
            }
            /* Keep Dataset dropdown and Download button tidy on narrow screens */
            .data-controls {
                flex-wrap: wrap;
                gap: 0.5rem !important;
            }
            /* Prevent DataGrid from stretching excessively tall on mobile */
            .shiny-data-grid {
                max-height: 380px !important;
            }
        }
        """
    ),
    ui.layout_columns(
        # Middle / Top-on-mobile: Map
        ui.card(
            ui.card_header(ui.output_text("single_map_title")),
            output_widget("single_origin_map"),
            full_screen=True,
        ),
        # Right / Second-on-mobile: Data table
        ui.card(
            ui.card_header("Data"),
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
                class_="d-flex align-items-end gap-3 mb-2 data-controls",
            ),
            ui.output_data_frame("active_data_table"),
            full_screen=True,
        ),
        col_widths={"sm": (12, 12), "lg": (6, 6)},
    ),
    # Bottom: Explainer text
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
    map_widget = go.FigureWidget(get_origin_map(DEFAULT_ORIGIN))

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
        target = get_origin_map(chosen_origin())

        with map_widget.batch_update():
            for cur_trace, new_trace in zip(map_widget.data, target.data):
                if cur_trace.fill == "toself":
                    # County polygon. update only colour and hover text
                    cur_trace.fillcolor = new_trace.fillcolor
                    cur_trace.text = new_trace.text
                else:
                    # Legend dummy marker. update label and color
                    cur_trace.name = new_trace.name
                    cur_trace.marker.color = new_trace.marker.color
                    cur_trace.visible = True

            # Hide any extra legend slots if target has fewer bins than initial map
            for extra_trace in map_widget.data[len(target.data) :]:
                extra_trace.visible = False

            map_widget.layout.legend.title.text = target.layout.legend.title.text

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
