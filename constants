import geopandas as gpd
import pandas as pd

# Local path where place names data are stored
DATA_PATH = "IPN_GB_2024/IPN_GB_2024.csv"

# URL where the reference data for origins and strings are stored
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_elements_in_place_names_in_the_British_Isles"

ALLOWED_STRING_POSITIONS = ["interfix", "prefix", "standalone", "suffix"]

# Lookup for matching abbreviated origin name with full origin name
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

# Used when preparing data for the map and general output formatting
key_to_languages_table = pd.DataFrame(
    list(KEY_TO_LANGUAGES.items()), columns=["origin", "full_origin"]
)

# Place types to keep from the place names source data
PLACE_TYPES = ["BUA", "LOC", "LONB", "MD", "PAR"]

# Geometric Base: UK admin-1 boundaries (excluding Northern Ireland) in WGS84 (EPSG:4326)
_NE_ADMIN1_URL = (
    "https://naciscdn.org"
)
_uk_states = gpd.read_file(_NE_ADMIN1_URL)
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

LEAD_REMOVE_STRINGS = [
    "bank", "castle", "chapel", "east", "easter", "ferry",
    "garrison", "garden", "gardeners", "great", "gate",
    "high", "higher", "low", "lower", "little",
    "marsh", "middle", "new", "north", "old", "port",
    "south",
    "temple", "the", "upper", "west", "windmill",
]

TERMINAL_REMOVE_STRINGS = [
    "bank", "camp", "castle", "city", "colliery", "common", "corner", "cottages", "creek", "crossing",
    "east", "easter", "end", "edge", "estate",
    "docks", "down", "drive",
    "farm", "ferry", "field", "fields", "foot", "forest",
    "garden", "gardens", "garrison", "gate", "granch", "grange", "green", "grove",
    "heath", "house", "junction", "lane", "lodge", "lower",
    "marsh", "mead", "meadow", "meadows", "mill", "mills", "moor", "old",
    "park", "parks", "passage", "plain", "place", "port", "priors",
    "quay", "rd", "side", "south", "st", "street",
    "vale", "valley", "view", "village",
]