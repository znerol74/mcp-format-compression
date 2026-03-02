"""
Core serialization/deserialization for JSON, TOON, and TRON formats.

All three benchmarks (MCPToolBenchPP, MCP-Universe, mcp-bench) delegate
to these functions so that identical input data produces identical output.
"""

import json
import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ToolFormat(str, Enum):
    """Supported serialization formats."""
    JSON = "json"
    TOON = "toon"
    TRON = "tron"


def serialize(data: Any, fmt: ToolFormat) -> str:
    """Convert a Python dict/list/value to the target format string.

    Args:
        data: The Python object to serialize.
        fmt: Target format (JSON, TOON, or TRON).

    Returns:
        Serialized string in the target format.
    """
    if fmt == ToolFormat.JSON:
        return json.dumps(data)
    elif fmt == ToolFormat.TOON:
        from toon_format import encode
        return encode(data)
    elif fmt == ToolFormat.TRON:
        from tron import TRON
        return TRON.stringify(data)
    return json.dumps(data)


def deserialize(text: str, fmt: ToolFormat) -> Any:
    """Parse a format string back to a Python object.

    Args:
        text: The string to parse.
        fmt: Format of the input string.

    Returns:
        Parsed Python object (dict, list, etc.).
    """
    if fmt == ToolFormat.JSON:
        return json.loads(text)
    elif fmt == ToolFormat.TOON:
        from toon_format import decode
        return decode(text)
    elif fmt == ToolFormat.TRON:
        from tron import TRON
        return TRON.parse(text)
    return json.loads(text)


def deserialize_lenient(text: str, fmt: ToolFormat) -> Any:
    """Try target format first, fall back to JSON, fall back to {}.

    Args:
        text: The string to parse.
        fmt: Expected format of the input string.

    Returns:
        Parsed Python object, or empty dict on failure.
    """
    try:
        return deserialize(text, fmt)
    except Exception:
        try:
            return json.loads(text)
        except Exception as e:
            logger.error("Failed to parse as %s or JSON: %s", fmt.value, e)
            return {}
