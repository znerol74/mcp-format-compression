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


class FormatViolation(ValueError):
    """Raised when text cannot be parsed as the requested format.

    Distinct from generic ValueError so call sites can count violations
    separately from other parse errors.
    """

    def __init__(self, fmt: ToolFormat, original_exc: Exception, snippet: str):
        self.fmt = fmt
        self.original_exc = original_exc
        self.snippet = snippet
        super().__init__(
            f"format violation: expected {fmt.value}: "
            f"{type(original_exc).__name__}: {original_exc} | "
            f"snippet={snippet[:200]!r}"
        )


def _looks_like_json_or_tron(text: str) -> bool:
    """Lightweight shape check: JSON/TRON tool-call payloads start with '{' or '['."""
    stripped = text.lstrip()
    return stripped.startswith(("{", "["))


def deserialize_strict(text: str, fmt: ToolFormat) -> Any:
    """Parse text as the given format. No JSON fallback, no empty-dict fallback.

    Use this when a parse failure should be surfaced as a real failure (e.g.
    tool-call output that the model emitted in the wrong format). Pair with
    a try/except FormatViolation block to count the violation.

    Includes a lightweight shape check: TOON's parser is permissive and will
    accept JSON-shaped text by interpreting '{...}' as a degenerate string key.
    If the caller asks for TOON and the text starts with '{' or '[' (a JSON
    or TRON tool-call payload), treat that as a format violation.

    Args:
        text: The string to parse.
        fmt: Expected format of the input string.

    Returns:
        Parsed Python object.

    Raises:
        FormatViolation: if text is not valid in the given format.
    """
    if fmt == ToolFormat.TOON and _looks_like_json_or_tron(text):
        raise FormatViolation(
            fmt,
            ValueError("text starts with '{' or '['; looks like JSON/TRON, not TOON"),
            text,
        )
    try:
        return deserialize(text, fmt)
    except Exception as e:
        raise FormatViolation(fmt, e, text) from e
