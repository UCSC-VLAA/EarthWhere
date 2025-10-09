

SYSTEM_PROMPT = (
    "You are a geolocation analyst. Given an image, a HINT, and an ANSWER_TYPE "
    "(country|province|city|county|town|street|latitude & longitude): "
    "1) Extract concrete visual evidence (e.g., signage text/language, road markings, license plates style, driving side, "
    "architecture, vegetation/biome, terrain, rail features, utility furniture). "
    "2) Decide via a coarse→fine funnel (country→region→city→street) and commit to ONE location at the requested granularity. "
    "If a finer granularity is requested, you MUST choose a plausible candidate at that level rather than stopping early. "
    "Return ONLY valid JSON matching the schema exactly."
    "If uncertain, still pick the single best candidate matching the ANSWER_TYPE accordingly."
    "Respond in English."
)

SYSTEM_PROMPT_NO_JSON = (
    "You are a geolocation analyst. Given an image, a HINT, and an ANSWER_TYPE "
    "(country|province|city|county|town|street|latitude & longitude): "
    "1) Extract concrete visual evidence (e.g., signage text/language, road markings, license plates style, driving side, "
    "architecture, vegetation/biome, terrain, rail features, utility furniture). "
    "2) Decide via a coarse→fine funnel (country→region→city→street) and commit to ONE location at the requested granularity. "
    "3) Use Google Search tool if needed. "
    "If a finer granularity is requested, you MUST choose a plausible candidate at that level rather than stopping early. "
    "If uncertain, still pick the single best candidate matching the ANSWER_TYPE accordingly."
    "Respond in English."
    "Provide detailed reasoning process between the <think></think> tag. Give the final answer between the <answer></answer> tag."
)


#"Return ONLY valid JSON matching the schema exactly."


JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "answer_type": {
            "type": "string",
            "enum": ["country", "province", "city", "county", "town","street", "latitude & longitude"]
        },
        "name": {"type": "string"},
        "admin_path": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Most-specific first, e.g., ['Guangdong','China']"
        },
        "lat": {"type": "number"},
        "lon": {"type": "number"},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
            },
            "description": "analysis and deduction process"
        },
        "fallback_level": {
            "type": ["string", "null"],
            "enum": [None, "country", "province", "city", "county","town"]
        },
        "notes": {"type": "string"}
    },
    "required": ["admin_path", "evidence"],
    "allOf": [
        # If latlon is requested, require lat/lon and forbid name/admin_path.
        {
            "if": {"properties": {"answer_type": {"const": "latitude & longitude"}}},
            "then": {
                "required": ["lat", "lon"],
                "not": {"required": ["name"]}  # name optional but discouraged for latlon
            }
        },
        # If NOT latlon, require name; lat/lon optional.
        {
            "if": {"properties": {"answer_type": {"not": {"const": "latitude & longitude"}}}},
            "then": {"required": ["name"]}
        }
    ],
    "additionalProperties": False
}