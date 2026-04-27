"""
Shared format converter for MCP benchmark alignment.

Provides canonical serialization (JSON/TOON/TRON), prompt building blocks,
and token counting used identically by MCPToolBenchPP, MCP-Universe,
and mcp-bench.
"""

from shared_format.converter import (
    ToolFormat,
    serialize,
    deserialize,
    deserialize_lenient,
    deserialize_strict,
    FormatViolation,
)
from shared_format.prompt_snippets import (
    get_format_name,
    get_format_explanation,
    get_format_intro,
    get_format_reminder,
    serialize_tools,
    serialize_example,
)
from shared_format.token_counter import count_tokens

__all__ = [
    "ToolFormat",
    "serialize",
    "deserialize",
    "deserialize_lenient",
    "deserialize_strict",
    "FormatViolation",
    "get_format_name",
    "get_format_explanation",
    "get_format_intro",
    "get_format_reminder",
    "serialize_tools",
    "serialize_example",
    "count_tokens",
]
