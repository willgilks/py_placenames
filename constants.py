from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# Local path where place names data are stored
DATA_PATH = "data/IPN_GB_2024.csv"

# url for wiki table on matching place name strings to origins.
WIKI_URL = (
    "https://en.wikipedia.org/wiki/List_of_elements_in_place_names_in_the_British_Isles"
)

ONS_PLACENAMES_URL = "https://open-geography-portalx-ons.hub.arcgis.com/datasets/ons::index-of-place-names-july-2024-in-gb/about"

ALLOWED_STRING_POSITIONS = ["interfix", "prefix", "standalone", "suffix"]


# full origin names and abbreviations
KEY_TO_LANGUAGES = {
    "Bry": "Brythonic",
    "C": "Cumbric",
    "K": "Cornish",
    "I": "Irish",
    "L": "Latin",
    "ME": "Middle English",
    "NF": "Norman French",
    "OE": "Old English",
    "ON": "Old Norse",
    "P": "Pictish",
    "S": "Scots",
    "SG": "Scots Gaelic",
    "W": "Welsh",
}

# make as single line
LANGUAGE_TO_KEY = {v: k for k, v in KEY_TO_LANGUAGES.items()}

# Used when preparing data for the map and general output formatting
key_to_languages_table = pd.DataFrame(
    list(KEY_TO_LANGUAGES.items()), columns=["origin", "full_origin"]
)

# Place types to keep from the place names source data
PLACE_TYPES = ["BUA", "LOC", "LONB", "MD", "PAR"]


# prepare co-ordinates for counties of the uk but excluding northern ireland.
_CACHE_GEOJSON = Path("data/gb_boundaries.geojson")
_NE_ADMIN1_URL = "https://githubusercontent.comne_10m_admin_1_states_provinces.geojson"

if _CACHE_GEOJSON.exists():
    GB_BOUNDARIES = gpd.read_file(_CACHE_GEOJSON)
else:
    _resp = requests.get(_NE_ADMIN1_URL, timeout=60)
    _resp.raise_for_status()
    _uk_states = gpd.read_file(BytesIO(_resp.content), driver="GeoJSON")
    GB_BOUNDARIES = (
        _uk_states[
            (_uk_states["admin"] == "United Kingdom")
            & (_uk_states["region"] != "Northern Ireland")
        ]
        .to_crs(epsg=4326)
        .assign(
            geometry=lambda df: df.geometry.simplify(
                tolerance=0.001, preserve_topology=True
            )
        )
    )

# words to remove from the front of place names before origin pattern matching
LEAD_REMOVE_STRINGS = [
    "bank",
    "castle",
    "chapel",
    "east",
    "easter",
    "ferry",
    "garrison",
    "garden",
    "gardeners",
    "great",
    "gate",
    "high",
    "higher",
    "low",
    "lower",
    "little",
    "marsh",
    "middle",
    "new",
    "north",
    "old",
    "port",
    "south",
    "temple",
    "the",
    "upper",
    "west",
    "windmill",
]

# words to remove from the end of place names before origin pattern matching.
TERMINAL_REMOVE_STRINGS = [
    "bank",
    "camp",
    "castle",
    "city",
    "colliery",
    "common",
    "corner",
    "cottages",
    "creek",
    "crossing",
    "east",
    "easter",
    "end",
    "edge",
    "estate",
    "docks",
    "down",
    "drive",
    "farm",
    "ferry",
    "field",
    "fields",
    "foot",
    "forest",
    "garden",
    "gardens",
    "garrison",
    "gate",
    "granch",
    "grange",
    "green",
    "grove",
    "heath",
    "house",
    "junction",
    "lane",
    "lodge",
    "lower",
    "marsh",
    "mead",
    "meadow",
    "meadows",
    "mill",
    "mills",
    "moor",
    "old",
    "park",
    "parks",
    "passage",
    "plain",
    "place",
    "port",
    "priors",
    "quay",
    "rd",
    "side",
    "south",
    "st",
    "street",
    "vale",
    "valley",
    "view",
    "village",
]


# ui end

SIDEBAR_TEXT = ""

EXPLAINER_TEXT_LEAD = (
    "The aim of this app is to show which parts of the county are associated with which tribal origins, based on the place names. "
    "Place origin is determined by whether the place name contains a certain prefix or suffix from a particular origin.\n\n"
    "Some prefixes can be from multiple origins. "
    "Where this occurs, the count for that word is divided by the number of origins that it can represent.\n\n"
)

WIKI_URL_TEXT = (
    "Origins and place name word parts from Wikipedia. "
    + WIKI_URL
    + "\n"
    + "Where prefix/suffix position is not provided, it is infered using the example place names. "
    + "\n\n"
)

NATURAL_EARTH_URL = "https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/"


ONS_PLACENAMES_URL_TEXT = (
    f"Place names data and locations from Office of National Statistics (ONS)."
    f"{ONS_PLACENAMES_URL}"
    f"\n"
    f"Place levels used are {', '.join(PLACE_TYPES)}. "
    f"LOC-level locations from Cornwall are omitted due to high density of miscellaneous names.\n\n"
)

COUNTY_BOUNDARIES_TXT = (
    "County boundaries data from Natural Earth "
    + NATURAL_EARTH_URL
    + "\n"
    + "To prevent duplication of place names, common words like Park and "
    + "Field are removed, then locations are grouped by the core place name, "
    "and filtered for the one with the shortest original place name. "
)

EXPLAINER_TEXT = (
    EXPLAINER_TEXT_LEAD
    + WIKI_URL_TEXT
    + ONS_PLACENAMES_URL_TEXT
    + COUNTY_BOUNDARIES_TXT
    + "\n\nWritten in the Python language."
)
