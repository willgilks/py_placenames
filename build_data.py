from datetime import datetime
from pathlib import Path

import pandas as pd

from constants import (
    ALLOWED_STRING_POSITIONS,
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
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Pull table of origins and text strings from wiki
    print(f"{datetime.now().strftime('%H:%M:%S')} Load wiki data")
    raw_wiki_path = data_dir / "wiki_table_raw.csv"

    try:
        wiki_table_raw = pull_wiki_origin_strings(wiki_url=WIKI_URL)
        wiki_table_raw.to_csv(raw_wiki_path, index=False)
    except Exception as exc:
        if raw_wiki_path.exists():
            print(
                f"{datetime.now():%H:%M:%S} Warning: Wiki pull failed ({exc}). "
                f"Using cached {raw_wiki_path}"
            )
            wiki_table_raw = pd.read_csv(raw_wiki_path)
        else:
            raise RuntimeError(
                f"Failed to pull Wiki data and no backup file found at {raw_wiki_path}"
            ) from exc

    # Prep wiki reference data for string-origin matching
    wiki_prepped = prepare_wiki_data(
        input_df=wiki_table_raw,
        key_to_languages=KEY_TO_LANGUAGES,
        allowed_string_positions=ALLOWED_STRING_POSITIONS,
    )

    term_ref = wiki_prepped["term_origin_position_dat"]
    term_ref.to_csv(data_dir / "placename_strings_reference.csv", index=False)

    print(f"{datetime.now().strftime('%H:%M:%S')} Load ons place names data")
    raw_placenames_path = data_dir / "IPN_GB_2024.csv"
    placenames_dat_raw = pd.read_csv(
        raw_placenames_path, encoding="latin1", low_memory=False
    )

    # Basic filter and standardisation
    print(f"{datetime.now().strftime('%H:%M:%S')} Basic filter and standardisation")
    placenames_initial_clean = clean_placenames_data(
        input_df=placenames_dat_raw,
        place_types=PLACE_TYPES,
        lead_remove_strings=LEAD_REMOVE_STRINGS,
        terminal_remove_strings=TERMINAL_REMOVE_STRINGS,
    )

    # Assign counties to places based on their lat and long
    print(f"{datetime.now().strftime('%H:%M:%S')} Assign counties ")
    placenames_matched_with_counties = assign_counties_from_sf(
        input_df=placenames_initial_clean,
        gb_boundaries=GB_BOUNDARIES,
    )

    # Assign origins to place names based on partial string matching
    print(f"{datetime.now().strftime('%H:%M:%S')} Assign origins")
    origin_placename_result = match_origin_to_placename(
        origin_to_placename_input_df=placenames_matched_with_counties,
        search_terms_df=term_ref,
        min_term_length=3,
    )

    # Calculate proportions of each origin in each county
    print(f"{datetime.now().strftime('%H:%M:%S')} Calculate country proportions")
    post_prep_result_list = post_prep_matched_names(
        assignment_result_df=origin_placename_result,
        placenames_matched_with_counties_df=placenames_matched_with_counties,
    )

    # make plot data
    print(f"{datetime.now().strftime('%H:%M:%S')} Making plot summary data")
    county_plot_dat = post_prep_result_list["county_plot_dat"]
    matched_dat = post_prep_result_list["matched_and_filtered_dat"]

    # write csvs
    print(f"{datetime.now().strftime('%H:%M:%S')} Write csvs")

    wiki_table_raw.to_csv(data_dir / "wiki_table_raw.csv", index=False)
    county_plot_dat.to_csv(data_dir / "county_plot_dat.csv", index=False)
    matched_dat.to_csv(data_dir / "matched_and_filtered_dat.csv", index=False)

    # write county boundaries
    GB_BOUNDARIES.to_file(data_dir / "gb_boundaries.geojson", driver="GeoJSON")
    print(
        f"{datetime.now().strftime('%H:%M:%S')} Pipeline complete! All processed files saved to ./data/"
    )


# execute
if __name__ == "__main__":
    run_pipeline()
