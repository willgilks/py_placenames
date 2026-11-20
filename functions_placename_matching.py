import itertools
import re

import numpy as np
import pandas as pd

from constants import key_to_languages_table


def make_search_terms_list(
    input_df: pd.DataFrame,
    min_term_length: int = 2,
) -> dict[str, dict[str, list[str]]]:
    """Builds a nested dictionary {position: {origin: [terms]}} filtered by term length."""
    filtered = input_df[input_df["term_len"] >= min_term_length][
        ["position", "origin", "term"]
    ].dropna()

    output_dict: dict[str, dict[str, list[str]]] = {}
    for (position, origin), group in filtered.groupby(
        ["position", "origin"], sort=False
    ):
        output_dict.setdefault(position, {})[origin] = group["term"].tolist()

    return output_dict


def make_placename_pattern_lookup(
    lookup_list: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    """Compiles position-aware regex patterns per language origin."""
    rows = []
    for position, origin_dict in lookup_list.items():
        for origin, elements in origin_dict.items():
            # Sort longest-first so longer elements match before shorter substrings
            escaped = [
                re.escape(el) for el in sorted(set(elements), key=len, reverse=True)
            ]
            if position == "prefix":
                processed = [f"^{el}" for el in escaped]
            elif position == "suffix":
                processed = [f"{el}$" for el in escaped]
            else:
                processed = escaped

            rows.append({"origin": origin, "pattern": "|".join(processed)})

    if not rows:
        return pd.DataFrame(columns=["origin", "pattern"])

    df = pd.DataFrame(rows)
    return df.groupby("origin", as_index=False)["pattern"].agg(
        lambda s: "|".join(f"(?:{p})" for p in s)
    )


# match origin to place name
def match_origin_to_placename(
    origin_to_placename_input_df: pd.DataFrame,
    search_terms_df: pd.DataFrame,
    min_term_length: int = 2,
) -> pd.DataFrame:
    """Matches place names against compiled origin regex patterns."""
    eligible_terms_df = search_terms_df[search_terms_df["term_len"] >= min_term_length]

    search_terms_list = make_search_terms_list(
        input_df=eligible_terms_df,
        min_term_length=min_term_length,
    )

    unique_terms = sorted(
        eligible_terms_df["term"].dropna().unique(), key=len, reverse=True
    )
    if not unique_terms:
        return pd.DataFrame(
            columns=["searchable_placename", "comb_origins", "exact_match_text"]
        )

    all_terms_concat = "|".join(re.escape(t) for t in unique_terms)

    valid_names = (
        origin_to_placename_input_df[["searchable_placename"]]
        .dropna()
        .drop_duplicates()
    )

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
    compiled_patterns = [
        (row["origin"], re.compile(row["pattern"], flags=re.IGNORECASE))
        for _, row in compiled_lookup.iterrows()
    ]

    comb_origins_list: list[str] = []
    exact_match_list: list[str | float] = []

    for name in filtered_df["searchable_placename"]:
        matched_origins: set[str] = set()
        matched_substrings: set[str] = set()

        for origin, regex in compiled_patterns:
            m = regex.search(name)
            if m:
                matched_origins.add(origin)
                matched_substrings.add(m.group(0).lower())

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

    def _unique_sorted_tokens(series: pd.Series) -> str:
        tokens = {
            tok.strip()
            for cell in series.dropna()
            for tok in str(cell).split(",")
            if tok.strip()
        }
        return ",".join(sorted(tokens))

    # Summarise valid matches per searchable_placename
    valid_assignments = assignment_result_df[
        assignment_result_df["comb_origins"].notna()
        & (assignment_result_df["comb_origins"] != "Unknown")
    ]

    initial_sum_df = valid_assignments.groupby(
        "searchable_placename", as_index=False
    ).agg(
        comb_origins=("comb_origins", _unique_sorted_tokens),
        text_matches=("exact_match_text", _unique_sorted_tokens),
    )

    # Join place names with origins back to main data & deduplicate
    df = placenames_matched_with_counties_df.merge(
        initial_sum_df, on="searchable_placename", how="left"
    )
    df["comb_origins"] = df["comb_origins"].fillna("Unknown")

    df = df[~df["place23nm_repaired"].str.contains(",", regex=False, na=False)].copy()
    df["placename_length"] = df["place23nm_repaired"].str.len()

    # Keep shortest name per placeid, prioritising arbitrarily by lat and long
    min_len_placeid = df.groupby("placeid")["placename_length"].transform("min")
    df = (
        df[df["placename_length"] == min_len_placeid]
        .sort_values(["placeid", "lat", "long"])
        .drop_duplicates(subset=["placeid"], keep="first")
    )

    # Keep shortest name per (lat, long)
    min_len_coords = df.groupby(["lat", "long"])["placename_length"].transform("min")
    df = df[df["placename_length"] == min_len_coords]

    # Keep first entry per (county_name, searchable_placename) ordered by lat, long
    df = df.sort_values(
        ["county_name", "searchable_placename", "lat", "long"]
    ).drop_duplicates(subset=["county_name", "searchable_placename"], keep="first")

    # Filter to known origins for plotting & downloadable place-level table
    matched_and_filtered_out = df[df["comb_origins"] != "Unknown"][
        ["place23nm_mod", "county_name", "comb_origins", "text_matches"]
    ].reset_index(drop=True)

    # Explode so there is one row per unique origin, place, and text match
    exploded = (
        matched_and_filtered_out.assign(
            origin=lambda d: d["comb_origins"].str.split(",")
        )
        .explode("origin")
        .assign(origin=lambda d: d["origin"].str.strip())
        .query("origin != ''")[
            ["county_name", "origin", "place23nm_mod", "text_matches"]
        ]
        .drop_duplicates()
        .rename(columns={"text_matches": "text_match"})
    )

    # Weight each place equally across its matched origins
    exploded["n_origins"] = exploded.groupby(
        ["county_name", "place23nm_mod", "text_match"]
    )["origin"].transform("count")
    exploded["place_wt"] = 1.0 / exploded["n_origins"]

    # Build top-n example strings per (county_name, origin, text_match)
    examples_df = (
        exploded.groupby(["county_name", "origin", "text_match"], as_index=False)[
            "place23nm_mod"
        ]
        .agg(lambda s: ", ".join(sorted(s.dropna().unique())[:5]))
        .assign(
            disp_str=lambda d: d["text_match"].fillna("") + ": " + d["place23nm_mod"]
        )
        .groupby(["county_name", "origin"], as_index=False)["disp_str"]
        .agg(lambda s: ". ".join(sorted(set(s))))
        .rename(columns={"disp_str": "strings_and_matches"})
    )

    # Aggregate exact counts & percentages per (county_name, origin)
    county_summary_df = (
        exploded.groupby(["county_name", "origin"], as_index=False)
        .agg(n_places_raw=("place_wt", "sum"))
        .merge(examples_df, on=["county_name", "origin"], how="left")
    )

    county_summary_df["strings_and_matches"] = county_summary_df[
        "strings_and_matches"
    ].str.replace(r" [,]", "", regex=True)

    county_summary_df["n_county_places"] = (
        county_summary_df.groupby("county_name")["n_places_raw"]
        .transform("sum")
        .round(0)
        .astype(int)
    )
    county_summary_df["pc"] = (
        100.0
        * (county_summary_df["n_places_raw"] / county_summary_df["n_county_places"])
    ).round(2)
    county_summary_df["n_places"] = county_summary_df["n_places_raw"].round(2)
    county_summary_df = county_summary_df.drop(columns=["n_places_raw"])

    # Expand grid so missing county-origin pairs show as zero
    counties = sorted(county_summary_df["county_name"].dropna().unique())
    origins = sorted(county_summary_df["origin"].dropna().unique())
    grid = pd.DataFrame(
        list(itertools.product(counties, origins)),
        columns=["county_name", "origin"],
    )

    county_plot_df = grid.merge(
        county_summary_df, on=["county_name", "origin"], how="left"
    )
    county_plot_df["n_county_places"] = (
        county_plot_df.groupby("county_name")["n_county_places"]
        .bfill()
        .ffill()
        .fillna(0)
        .astype(int)
    )

    num_cols = county_plot_df.select_dtypes(include="number").columns
    county_plot_df[num_cols] = county_plot_df[num_cols].fillna(0)

    county_plot_df = (
        county_plot_df.merge(lang_lookup_df, on="origin", how="left")
        .assign(origin=lambda d: d["full_origin"])
        .drop(columns=["full_origin"])
    )

    return {
        "matched_and_filtered_dat": matched_and_filtered_out,
        "county_plot_dat": county_plot_df,
    }
