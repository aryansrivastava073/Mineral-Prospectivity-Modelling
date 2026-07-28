"""Maps internal feature/column names to the (table, column) keys used in
geology_code_dictionary.xlsx, so the app can show plain-English labels
instead of raw codes."""
from data_prep import load_dictionary

DICT_LOOKUP, MINERAL_LOOKUP = load_dictionary()

# internal column name -> dictionary (table, column) key
FIELD_TABLE_MAP = {
    "host_unit": "geology_host_rock",
    "host_age": "geology_host_rock",
    "mineralization_style": "geology_mineralization",
    "mineralization_texture": "geology_mineralization",
    "continuity": "geology_mineralization",
    "control_type": "geology_mineralization",
    "ore_shape": "geology_orebody_geometry",
    "vein_type": "geology_vein",
    "alteration_type": "geology_alteration",
    "alteration_timing": "geology_alteration",
    "alteration_intensity": "geology_alteration",
    "fault_setting": "geology_fault",
    "structural_order": "geology_fault",
    "pathfinder_elements": "geology_deposit",
    "deposit_type": "geology_deposit",
    "tectonic_setting": "geology_deposit",
    "geophysics_type": "geology_geophysics",
}

# fields whose codes are mineral codes -> use the Minerals sheet instead
MINERAL_FIELDS = {"ore_minerals", "gangue_minerals", "alteration_minerals"}


def decode_code(field, code):
    """Return a human-readable label for a single code in a given field."""
    if field in MINERAL_FIELDS:
        return MINERAL_LOOKUP.get(code, code)
    table = FIELD_TABLE_MAP.get(field)
    if table is None:
        return code
    return DICT_LOOKUP.get((table, field, code), code)


def decode_feature_column(feature_col):
    """feature_col looks like 'host_unit::GRA' -> ('host_unit', 'Granite (or whatever)')"""
    field, code = feature_col.split("::", 1)
    return field, decode_code(field, code)


def pretty_field_name(field):
    return field.replace("_", " ").title()
