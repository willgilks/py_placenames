import itertools
import re
import unicodedata
from io import StringIO

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from constants import (
    ALLOWED_STRING_POSITIONS,
    GB_BOUNDARIES,
    KEY_TO_LANGUAGES,
    LEAD_REMOVE_STRINGS,
    PLACE_TYPES,
    TERMINAL_REMOVE_STRINGS,
    WIKI_URL,
)


# pull origin strings from wiki
# def get_wiki_placenames_lookup(wiki_url: str = WIKI_URL) -> pd.DataFrame:
def pull_wiki_origin_strings(wiki_url: str = WIKI_URL) -> pd.DataFrame:
    """Pulls string and origin info tables from the Wikipedia URL."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PlaceNamesApp/1.0)"}
    response = requests.get(wiki_url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    if not tables:
        tables = soup.find_all("table")

    dfs = [pd.read_html(StringIO(str(tbl)))[0] for tbl in tables]
    return pd.concat(dfs, ignore_index=True)


# for source strings where part of it is in brackets, make these into independent strings.
def expand_optional_parens(terms: pd.Series) -> pd.Series:
    """Expands optional parenthetical blocks like 'auch(en)/(in)' into comma-separated terms."""

    def _expand_cell(cell: str) -> str:
        if not isinstance(cell, str) or "(" not in cell:
            return cell

        sub_terms = cell.split(",")
        expanded_all = []

        for t in sub_terms:
            if "(" not in t:
                expanded_all.append(t)
                continue

            parts = re.findall(r"\([^)]+\)|[^()]+", t)
            opts = []
            for p in parts:
                if p.startswith("(") and p.endswith(")"):
                    inner = p[1:-1]
                    if "/" in inner:
                        opts.append(inner.split("/"))
                    else:
                        opts.append(["", inner])
                else:
                    opts.append([p])

            # expand.grid column-major ordering
            for combo in itertools.product(*reversed(opts)):
                expanded_all.append("".join(reversed(combo)))

        seen = set()
        unique_non_empty = []
        for item in expanded_all:
            if item != "" and item not in seen:
                seen.add(item)
                unique_non_empty.append(item)

        return ",".join(unique_non_empty)

    return terms.apply(_expand_cell)


# re-structure the wiki table for origins and search strings to make it usable.
def prepare_wiki_data(
    input_df: pd.DataFrame,
    key_to_languages: dict = KEY_TO_LANGUAGES,
    allowed_string_positions: list = ALLOWED_STRING_POSITIONS,
) -> dict[str, pd.DataFrame]:
    """Cleans Wikipedia reference data and infers missing term positions from examples."""
    valid_langs = set(key_to_languages.keys())
    allowed_pos_set = set(allowed_string_positions)

    # Initial standardisation
    clean_base = input_df.copy()
    clean_base["source_row_number"] = range(1, len(clean_base) + 1)
    clean_base = clean_base[
        ["Term", "Origin", "Position", "Example", "Meaning", "source_row_number"]
    ].rename(
        columns={
            "Term": "term",
            "Origin": "origin",
            "Position": "position",
            "Example": "example",
            "Meaning": "meaning",
        }
    )

    clean_base = clean_base.fillna("")
    for col in ["term", "origin", "position", "example", "meaning"]:
        clean_base[col] = clean_base[col].astype(str)

    clean_base = clean_base.drop_duplicates()

    for col in ["term", "position", "example"]:
        clean_base[col] = clean_base[col].str.lower()

    # Remove characters in square brackets and parentheses
    for col in ["term", "origin", "position", "example"]:
        clean_base[col] = clean_base[col].str.replace(r"\s*\[[^\]]*\]", "", regex=True)
    for col in ["origin", "position", "example"]:
        clean_base[col] = clean_base[col].str.replace(r"\s*\([^)]*\)", "", regex=True)

    # Standardise weird strings in term column
    clean_base["term"] = clean_base["term"].str.replace(
        r";\s*also\s+|\s+and\s+", ",", regex=True
    )
    clean_base["term"] = clean_base["term"].str.replace(r"[-/\s]", "", regex=True)

    # Filter and sort valid origins
    def _clean_origin(val: str) -> str:
        tokens = re.split(r"[,/\s]+", val)
        valid = sorted({tok for tok in tokens if tok in valid_langs})
        return ",".join(valid)

    clean_base["origin"] = clean_base["origin"].apply(_clean_origin)

    # Clean and standardise positions
    clean_base.loc[clean_base["term"] == "upon,on,in", "position"] = "interfix"

    def _clean_position(val: str) -> str:
        tokens = re.split(r"[,\s]+", val)
        valid = sorted({tok for tok in tokens if tok in allowed_pos_set})
        return ",".join(valid)

    clean_base["position"] = clean_base["position"].apply(_clean_position)

    clean_base = clean_base[
        (clean_base["origin"] != "") & (clean_base["term"] != "")
    ].copy()

    # Explode search strings with optional brackets
    clean_base["term"] = expand_optional_parens(clean_base["term"])

    # Explode search string and example pairs, so that there's one row per search string and example.
    term_example_pairs = clean_base.copy()
    term_example_pairs["term"] = term_example_pairs["term"].str.split(",")
    term_example_pairs = term_example_pairs.explode("term")
    term_example_pairs["example"] = term_example_pairs["example"].str.split(",")
    term_example_pairs = term_example_pairs.explode("example")

    term_example_pairs["term"] = term_example_pairs["term"].str.strip()
    term_example_pairs["example"] = term_example_pairs["example"].str.strip()

    term_example_pairs = term_example_pairs[
        (term_example_pairs["term"] != "")
        & (term_example_pairs["example"] != "")
        & (term_example_pairs["example"] != "all surrounding the city of exeter")
        & (~term_example_pairs["example"].str.contains("people of", regex=False))
    ].copy()

    # Clean noise words but keep dash for interfix matching
    term_example_pairs["example_hyphen"] = (
        term_example_pairs["example"]
        .str.replace(
            r"\.|\b(river|country|park|county|high|king's|the)\b", " ", regex=True
        )
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Space-only version for prefix/suffix/standalone matching
    term_example_pairs["example_clean"] = (
        term_example_pairs["example_hyphen"]
        .str.replace("-", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    term_example_pairs = term_example_pairs[
        term_example_pairs["example_clean"] != ""
    ].copy()

    # Placename origins lookup table
    distinct_places = term_example_pairs[["example_clean", "origin"]].drop_duplicates()
    source_placenames_origins = (
        distinct_places.groupby("example_clean", as_index=False)["origin"]
        .agg(lambda s: ",".join(sorted({o for item in s for o in item.split(",")})))
        .rename(columns={"example_clean": "place_name", "origin": "origins"})
    )

    # Position inference when position is missing
    # This compares the source search string to the source example place names to deduce the position.
    missing_pos = term_example_pairs[term_example_pairs["position"] == ""].copy()
    has_term = missing_pos.apply(
        lambda row: row["term"] in row["example_clean"], axis=1
    )
    candidates = missing_pos[has_term][
        ["term", "origin", "example_clean", "example_hyphen"]
    ].drop_duplicates()

    inferred_rows = []
    for _, row in candidates.iterrows():
        t_esc = re.escape(row["term"])
        ex_clean = row["example_clean"]
        ex_hyph = row["example_hyphen"]

        checks = {
            "standalone": "standalone" in allowed_pos_set
            and bool(re.search(rf"(^|\s){t_esc}($|\s)", ex_clean)),
            "prefix": "prefix" in allowed_pos_set
            and bool(re.search(rf"(^|\s){t_esc}", ex_clean)),
            "suffix": "suffix" in allowed_pos_set
            and bool(re.search(rf"{t_esc}($|\s)", ex_clean)),
            "interfix": "interfix" in allowed_pos_set
            and bool(re.search(rf"(-|\s){t_esc}(-|\s)", ex_hyph)),
        }
        for pos_name, matched in checks.items():
            if matched:
                inferred_rows.append(
                    {
                        "term": row["term"],
                        "origin": row["origin"],
                        "inferred_pos": pos_name,
                    }
                )

    if inferred_rows:
        inferred_df = pd.DataFrame(inferred_rows)
        inferred_positions = (
            inferred_df.groupby(["term", "origin"], as_index=False)["inferred_pos"]
            .agg(lambda s: ",".join(sorted(set(s))))
            .rename(columns={"inferred_pos": "inferred_position"})
        )
    else:
        inferred_positions = pd.DataFrame(
            columns=["term", "origin", "inferred_position"]
        )

    # Combine inferred positions with source-defined positions
    expanded_terms = clean_base.copy()
    expanded_terms["term"] = expanded_terms["term"].str.split(",")
    expanded_terms = expanded_terms.explode("term")
    expanded_terms["term"] = expanded_terms["term"].str.strip()
    expanded_terms = expanded_terms[expanded_terms["term"] != ""].copy()

    merged = expanded_terms.merge(inferred_positions, on=["term", "origin"], how="left")
    merged["position"] = merged.apply(
        lambda r: (
            (r["inferred_position"] or "")
            if r["position"] == "" and pd.notna(r["inferred_position"])
            else r["position"]
        ),
        axis=1,
    )
    merged = merged[merged["position"] != ""].copy()

    merged["position"] = merged["position"].str.split(",")
    merged = merged.explode("position")

    term_origin_position_dat = merged.groupby(
        ["term", "position", "meaning"], as_index=False
    )["origin"].agg(
        lambda s: ",".join(sorted({o for item in s for o in item.split(",")}))
    )
    term_origin_position_dat["term_len"] = term_origin_position_dat["term"].str.len()
    term_origin_position_dat = (
        term_origin_position_dat[["term", "origin", "position", "meaning", "term_len"]]
        .sort_values(["term", "origin", "position"])
        .reset_index(drop=True)
    )

    return {
        "source_placenames_origins": source_placenames_origins,
        "term_origin_position_dat": term_origin_position_dat,
    }


# helper function to remove non-standard characters from source place names
def _latin_ascii(text: str) -> str:
    """Replicates stringi::stri_trans_general(x, 'Latin-ASCII')."""
    if pd.isna(text):
        return text
    normalized = unicodedata.normalize("NFKD", str(text))
    return normalized.encode("ascii", "ignore").decode("ascii")


# standardises and prepares list of uk place names.
def clean_placenames_data(
    input_df: pd.DataFrame,
    place_types: list[str] = PLACE_TYPES,
    lead_remove_strings: list[str] = LEAD_REMOVE_STRINGS,
    terminal_remove_strings: list[str] = TERMINAL_REMOVE_STRINGS,
) -> pd.DataFrame:
    """Filters and normalises raw GB place names data."""
    all_lead_and_termin_strings = set(lead_remove_strings) | set(
        terminal_remove_strings
    )

    # Pre-compile compact grouped regex patterns
    lead_pattern = re.compile(rf"^({'|'.join(map(re.escape, lead_remove_strings))})\s+")
    term_pattern = re.compile(
        rf"\s+({'|'.join(map(re.escape, terminal_remove_strings))})$"
    )
    bad_suffix_pattern = r"\s+(?:church|closes?|copses?|lands?|parks?|shop|square)$"

    # remove rows containing nonsense info.
    df = input_df[
        input_df["descnm"].isin(place_types)
        & ~(
            (input_df["cty61nm"] == "Cornwall and Isles of Scilly")
            & (input_df["descnm"] == "LOC")
        )
        & (input_df["lat"] != 0)
        & (input_df["long"] != 0)
        & (input_df["lat"] > 49)
        & (input_df["long"] > -8)
    ][
        [
            "tempcode",
            "placeid",
            "place23nm",
            "descnm",
            "lat",
            "long",
            "ctry23nm",
            "cty61nm",
        ]
    ].copy()

    # remove nonsense characters from place names
    df["place23nm_repaired"] = df["place23nm"].map(_latin_ascii)

    df = df[
        (df["place23nm_repaired"] != "Area not comprised")
        & (
            ~df["place23nm_repaired"].str.contains(
                bad_suffix_pattern, case=False, regex=True, na=False
            )
        )
    ].copy()

    # get length of repaired place name, for downstream prioritisation.
    df["place23nm_repaired_length"] = df["place23nm_repaired"].str.len()

    # remove unrequired parts of place name.
    df["place23nm_mod"] = (
        df["place23nm_repaired"]
        .str.replace(r"\s*\([^)]*\)|\s+\b(?:by|in|and)\b.*", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # remove more unrequired parts of place name.
    searchable = (
        df["place23nm_mod"]
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(r"(?:,|\s+&|\s+with\s+|\s+urban|\s+on\s+).*", "", regex=True)
        .str.strip()
    )

    # remove common leader and follower words like Green or Park, from place name.
    for _ in range(2):
        searchable = (
            searchable.str.replace(lead_pattern, "", n=1, regex=True)
            .str.replace(term_pattern, "", n=1, regex=True)
            .str.strip()
        )

    df["searchable_placename_length"] = searchable.str.len()
    df["searchable_placename"] = np.where(
        (searchable == "") | searchable.isin(all_lead_and_termin_strings),
        np.nan,
        searchable,
    )

    return df.reset_index(drop=True)


# assign counties to place names based on lat and long compared to the uk spatial object.
def assign_counties_from_sf(
    input_df: pd.DataFrame,
    gb_boundaries: gpd.GeoDataFrame = GB_BOUNDARIES,
) -> pd.DataFrame:
    """Assigns counties to place names based on their lat/long and a reference GeoDataFrame."""
    geo_dat_input = (
        input_df[["tempcode", "placeid", "place23nm_mod", "lat", "long"]]
        .drop_duplicates()
        .copy()
    )

    geo_dat_sf = gpd.GeoDataFrame(
        geo_dat_input,
        geometry=gpd.points_from_xy(geo_dat_input["long"], geo_dat_input["lat"]),
        crs="EPSG:4326",
    )

    # Spatial join (matches st_join(..., join = st_intersects))
    # if a point lies on a shared border, keep first match to avoid duplicating rows
    geo_lookup_result = gpd.sjoin(
        geo_dat_sf,
        gb_boundaries[["name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    geo_lookup_result = geo_lookup_result[
        ~geo_lookup_result.index.duplicated(keep="first")
    ]

    place_county_reference = pd.DataFrame(
        geo_lookup_result.drop(columns=["geometry", "index_right"])
    ).rename(columns={"name": "county_name"})[
        ["tempcode", "placeid", "place23nm_mod", "lat", "long", "county_name"]
    ]

    is_missing = place_county_reference["county_name"].isna() | (
        place_county_reference["county_name"] == ""
    )
    known_towns = place_county_reference[~is_missing].copy()
    missing_towns = place_county_reference[is_missing].copy()

    if len(missing_towns) > 0 and len(known_towns) > 0:
        known_sf = gpd.GeoDataFrame(
            known_towns,
            geometry=gpd.points_from_xy(known_towns["long"], known_towns["lat"]),
            crs="EPSG:4326",
        ).to_crs(
            epsg=27700
        )  # British National Grid (metres) for accurate nearest-neighbour distance

        missing_sf = gpd.GeoDataFrame(
            missing_towns,
            geometry=gpd.points_from_xy(missing_towns["long"], missing_towns["lat"]),
            crs="EPSG:4326",
        ).to_crs(epsg=27700)

        nearest_joined = gpd.sjoin_nearest(
            missing_sf[["geometry"]],
            known_sf[["county_name", "geometry"]],
            how="left",
        )
        nearest_joined = nearest_joined[~nearest_joined.index.duplicated(keep="first")]
        missing_towns["county_name"] = nearest_joined["county_name"].values

    combined_ref = pd.concat([known_towns, missing_towns], ignore_index=True)

    return input_df.merge(
        combined_ref,
        on=["tempcode", "placeid", "place23nm_mod", "lat", "long"],
        how="left",
    )
