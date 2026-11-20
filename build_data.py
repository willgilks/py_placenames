import os
from pathlib import Path

import pandas as pd

from constants import (
    ALLOWED_STRING_POSITIONS,
    DATA_PATH,
    GB_BOUNDARIES,
    KEY_TO_LANGUAGES,
    LEAD_REMOVE_STRINGS,
    PLACE_TYPES,
    TERMINAL_REMOVE_STRINGS,
    WIKI_URL,
)
from functions_data_prep import (
    assign_counties_from_sf,
    clean_placenames_data,
    prepare_wiki_data,
    pull_wiki_origin_strings,
)
from functions_placename_matching import (
    match_origin_to_placename,
    post_prep_matched_names,
)


def run_pipeline() -> None:
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # Pull table of origins and text strings from wiki
    wiki_table_raw = pull_wiki_origin_strings(wiki_url=WIKI_URL)
    wiki_table_raw.to_csv(out_dir / "wiki_table_raw.csv", index=False)

    # Prep wiki reference data for string-origin matching
    wiki_prepped = prepare_wiki_data(
        input_df=wiki_table_raw,
        key_to_languages=KEY_TO_LANGUAGES,
        allowed_string_positions=ALLOWED_STRING_POSITIONS,
    )
    term_ref = wiki_prepped["term_origin_position_dat"]
    term_ref.to_csv(out_dir / "placename_strings_reference.csv", index=False)

    # Load raw place names data
    data_root = os.getenv("DATA_PATH_ROOT", "")
    full_data_path = Path(data_root) / DATA_PATH if data_root else Path(DATA_PATH)

    placenames_dat_raw = pd.read_csv(
        full_data_path, encoding="latin1", low_memory=False
    )

    # Basic filter and standardisation
    placenames_initial_clean = clean_placenames_data(
        input_df=placenames_dat_raw,
        place_types=PLACE_TYPES,
        lead_remove_strings=LEAD_REMOVE_STRINGS,
        terminal_remove_strings=TERMINAL_REMOVE_STRINGS,
    )

    # Assign counties to places based on their lat and long
    placenames_matched_with_counties = assign_counties_from_sf(
        input_df=placenames_initial_clean,
        gb_boundaries=GB_BOUNDARIES,
    )

    # Assign origins to place names based on partial string matching
    origin_placename_result = match_origin_to_placename(
        origin_to_placename_input_df=placenames_matched_with_counties,
        search_terms_df=term_ref,
        min_term_length=2,
    )

    # Calculate proportions of each origin in each county
    post_prep_result_list = post_prep_matched_names(
        assignment_result_df=origin_placename_result,
        placenames_matched_with_counties_df=placenames_matched_with_counties,
    )

    # make plot data
    county_plot_dat = post_prep_result_list["county_plot_dat"]
    matched_dat = post_prep_result_list["matched_and_filtered_dat"]

    # write csvs
    county_plot_dat.to_csv(out_dir / "county_plot_dat.csv", index=False)
    matched_dat.to_csv(out_dir / "matched_and_filtered_dat.csv", index=False)

    # write county boundaries
    GB_BOUNDARIES.to_file(out_dir / "gb_boundaries.geojson", driver="GeoJSON")
    print("Pipeline complete! All processed files saved to ./data/")


# execute
if __name__ == "__main__":
    run_pipeline()
