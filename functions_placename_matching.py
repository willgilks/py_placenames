import itertools
import re

import numpy as np
import pandas as pd

from constants import key_to_languages_table


# make initial search patterns, depending on whether string is a prefix or suffix etc.
def make_placename_pattern_lookup(
    lookup_list: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    """Compiles position-aware regex patterns per language origin."""
    rows = []
    for position, origin_dict in lookup_list.items():
        for origin, elements in origin_dict.items():
            escaped = [re.escape(el) for el in elements]
            if position == "prefix":
                processed = [f"^{el}" for el in escaped]
            elif position == "suffix":
                processed = [f"{el}$" for el in escaped]
            else:
                processed = [f" {el} " for el in escaped]

            rows.append({"origin": origin, "pattern": "|".join(processed)})

    if not rows:
        return pd.DataFrame(columns=["origin", "pattern"])

    df = pd.DataFrame(rows)
    compiled = df.groupby("origin", as_index=False)["pattern"].agg(
        lambda s: "|".join(f" {p} " for p in s)
    )
    return compiled


# make list of search patterns, filtered by how long the search pattern is.
# for downstream use.
def make_search_terms_list(
    input_df: pd.DataFrame,
    min_term_length: int = 2,
) -> dict[str, dict[str, list[str]]]:
    """Builds a nested dictionary {position: {origin: [terms]}} filtered by term length."""
    filtered = input_df[input_df["term_len"] >= min_term_length][
        ["position", "origin", "term"]
    ]

    output_dict: dict[str, dict[str, list[str]]] = {}
    for (position, origin), group in filtered.groupby(
        ["position", "origin"], sort=False
    ):
        output_dict.setdefault(position, {})[origin] = group["term"].tolist()

    return output_dict


# main function for matching origin to place name based on strings.
def match_origin_to_placename(
    origin_to_placename_input_df: pd.DataFrame,
    search_terms_df: pd.DataFrame,
    min_term_length: int = 2,
) -> pd.DataFrame:
    """Matches place names against compiled origin regex patterns."""
    search_terms_list = make_search_terms_list(
        input_df=search_terms_df,
        min_term_length=min_term_length,
    )

    unique_terms = sorted(search_terms_df["term"].dropna().unique())
    all_terms_concat = "|".join(re.escape(t) for t in unique_terms)

    # remove any nonsense names
    valid_names = (
        origin_to_placename_input_df[["searchable_placename"]]
        .dropna()
        .drop_duplicates()
    )

    # keep only place names containing any searchable string
    filtered_df = (
        valid_names[
            valid_names["searchable_placename"].str.contains(
                all_terms_concat, case=False, regex=True, na=False
            )
        ]
        .sort_values("searchable_placename")
        .reset_index(drop=True)
    )

    compiled_lookup = make_placename_pattern_lookup(lookup_list=search_terms_list)

    # Compile regexes once for fast matching across all place names
    compiled_patterns = [
        (row["origin"], re.compile(row["pattern"], flags=re.IGNORECASE))
        for _, row in compiled_lookup.iterrows()
    ]

    comb_origins_list = []
    exact_match_list = []

    for name in filtered_df["searchable_placename"]:
        matched_origins = set()
        matched_substrings = set()

        for origin, regex in compiled_patterns:
            m = regex.search(name)
            if m:
                matched_origins.add(origin)
                matched_substrings.add(m.group(0))

        if not matched_origins:
            comb_origins_list.append("Unknown")
            exact_match_list.append(np.nan)
        else:
            comb_origins_list.append(",".join(sorted(matched_origins)))
            exact_match_list.append(",".join(sorted(matched_substrings)))

    filtered_df["comb_origins"] = comb_origins_list
    filtered_df["exact_match_text"] = exact_match_list
    return filtered_df


# calculate percentage of each origin in each county.
def post_prep_matched_names(
    assignment_result_df: pd.DataFrame,
    placenames_matched_with_counties_df: pd.DataFrame,
    lang_lookup_df: pd.DataFrame = key_to_languages_table,
) -> dict[str, pd.DataFrame]:
    """Summarises origin assignments per place name and aggregates county percentages."""

    # Summarise valid matches per searchable_placename
    valid_assignments = assignment_result_df[
        assignment_result_df["comb_origins"].notna()
        & (assignment_result_df["comb_origins"] != "Unknown")
    ]

    initial_sum_df = valid_assignments.groupby(
        "searchable_placename", as_index=False
    ).agg(
        comb_origins=("comb_origins", lambda s: ",".join(sorted(set(s)))),
        text_matches=("exact_match_text", lambda s: ",".join(pd.unique(s.dropna()))),
    )

    # Join place names with origins back to main data
    origin_placename_matched = placenames_matched_with_counties_df.merge(
        initial_sum_df, on="searchable_placename", how="left"
    )
    origin_placename_matched["comb_origins"] = origin_placename_matched[
        "comb_origins"
    ].fillna("Unknown")

    # Filter joined data to reduce duplication
    df = origin_placename_matched.copy()
    df["placename_length"] = df["place23nm_repaired"].str.len()
    df = df[~df["place23nm_repaired"].str.contains(",", regex=False, na=False)].copy()

    # Keep shortest name per placeid, prioritising arbitraily by lat and long, for plotting.
    min_len_placeid = df.groupby("placeid")["placename_length"].transform("min")
    df = (
        df[df["placename_length"] == min_len_placeid]
        .sort_values(["placeid", "lat", "long"])
        .drop_duplicates(subset=["placeid"], keep="first")
    )

    # Keep shortest name per (lat, long)
    min_len_coords = df.groupby(["lat", "long"])["placename_length"].transform("min")
    df = df[df["placename_length"] == min_len_coords].copy()

    # Keep first entry per (county_name, searchable_placename) ordered by lat, long
    df = df.sort_values(
        ["county_name", "searchable_placename", "lat", "long"]
    ).drop_duplicates(subset=["county_name", "searchable_placename"], keep="first")

    # remove any place names where origin is unknown. this is purely for plotting etc..
    # .. and assumes that unknown place names have the same distribution of origins as assigned place names.
    matched_and_filtered_df = df[df["comb_origins"] != "Unknown"][
        ["county_name", "comb_origins", "place23nm_mod", "text_matches"]
    ].reset_index(drop=True)
    matched_and_filtered_df["rn"] = range(1, len(matched_and_filtered_df) + 1)

    # Explode so there's one row per unique origin, place, and text match
    exploded = matched_and_filtered_df.copy()
    exploded["comb_origins"] = exploded["comb_origins"].str.split(",")
    exploded = exploded.explode("comb_origins")
    exploded["comb_origins"] = exploded["comb_origins"].str.strip()
    exploded = (
        exploded[exploded["comb_origins"] != ""][
            ["rn", "county_name", "comb_origins", "place23nm_mod", "text_matches"]
        ]
        .drop_duplicates()
        .rename(columns={"comb_origins": "origin", "text_matches": "text_match"})
    )

    # County summary weighted by number of origins per place
    exploded["n_origins"] = exploded.groupby(
        ["county_name", "place23nm_mod", "text_match"]
    )["origin"].transform("count")
    exploded["place_wt"] = 1.0 / exploded["n_origins"]

    def _top_five_places(series: pd.Series) -> str:
        top5 = sorted(series.dropna().unique())[:5]
        return ", ".join(top5)

    # get examples of matched places.
    disp_places_map = exploded.groupby(["county_name", "origin", "text_match"])[
        "place23nm_mod"
    ].transform(_top_five_places)
    exploded["disp_str"] = exploded["text_match"].fillna("") + ": " + disp_places_map

    # count number of places matching each origin in each county
    county_summary_df = (
        exploded.groupby(["county_name", "origin"], as_index=False)
        .agg(
            n_places=("place_wt", "sum"),
            strings_and_matches=("disp_str", lambda s: ". ".join(sorted(set(s)))),
        )
        .assign(n_places=lambda d: d["n_places"].round(2))
    )

    county_summary_df["strings_and_matches"] = county_summary_df[
        "strings_and_matches"
    ].str.replace(r" [,]", "", regex=True)

    county_summary_df["n_county_places"] = (
        county_summary_df.groupby("county_name")["n_places"]
        .transform("sum")
        .round(0)
        .astype(int)
    )

    county_summary_df["pc"] = (
        100.0 * (county_summary_df["n_places"] / county_summary_df["n_county_places"])
    ).round(2)

    # Expand grid for mapping so missing county-origin pairs show as zero
    counties = sorted(county_summary_df["county_name"].dropna().unique())
    origins = sorted(county_summary_df["origin"].dropna().unique())
    grid = pd.DataFrame(
        list(itertools.product(counties, origins)),
        columns=["county_name", "origin"],
    )
    # combines the counties and origins with the county percentages
    county_plot_df = grid.merge(
        county_summary_df, on=["county_name", "origin"], how="left"
    )
    ## fills county general county-level info to blank rows.
    county_plot_df["n_county_places"] = (
        county_plot_df.groupby("county_name")["n_county_places"].bfill().ffill()
    )

    # fill missing with zero.
    num_cols = county_plot_df.select_dtypes(include="number").columns
    county_plot_df[num_cols] = county_plot_df[num_cols].fillna(0)

    county_plot_df = county_plot_df.merge(lang_lookup_df, on="origin", how="left")
    county_plot_df["origin"] = county_plot_df["full_origin"]
    county_plot_df = county_plot_df.drop(columns=["full_origin"])

    return {
        "matched_and_filtered_dat": matched_and_filtered_df,
        "county_plot_dat": county_plot_df,
    }  # eof
