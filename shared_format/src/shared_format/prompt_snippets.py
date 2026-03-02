"""
Shared prompt building blocks for MCP benchmark alignment.

These functions produce identical prompt text across all benchmarks,
ensuring that the LLM receives the same format explanation and
instructions regardless of which benchmark is running.
"""

from typing import Any, Dict, List, Union

from shared_format.converter import ToolFormat, serialize


def get_format_name(fmt: ToolFormat) -> str:
    """Return the uppercase format name for prompt inclusion.

    Returns:
        'TOON', 'TRON', or 'JSON'.
    """
    return fmt.value.upper()


def get_format_explanation(fmt: ToolFormat) -> str:
    """Return the canonical format syntax explanation.

    This text is identical across all benchmarks so the LLM gets the
    same understanding of the format regardless of which benchmark runs.
    """
    if fmt == ToolFormat.TOON:
        return (
            "TOON (Token-Oriented Object Notation) is a compact format:\n"
            "- Objects use indentation instead of braces: key: value\n"
            "- Nested objects are indented with 2 spaces\n"
            "- Strings are unquoted unless they contain special characters (colon, newline)\n"
            "- Arrays use indexed keys: items[0]: first, items[1]: second\n"
            "Example:\n"
            "  name: my_tool\n"
            "  arguments:\n"
            "    query: hello world\n"
            "    limit: 10"
        )
    elif fmt == ToolFormat.TRON:
        return (
            "TRON is a compact JSON variant that uses class definitions "
            "to compress repeated structures:\n"
            "- Define classes for repeated shapes: class A: field1,field2\n"
            '- Use instances: A("value1","value2") instead of '
            '{"field1":"value1","field2":"value2"}\n'
            "- Single objects without repetition look like standard JSON\n"
            "Example:\n"
            '  {"name":"my_tool","arguments":{"query":"hello world","limit":10}}'
        )
    return (
        "JSON (JavaScript Object Notation) uses braces and quoted keys:\n"
        "Example:\n"
        '  {"name": "my_tool", "arguments": {"query": "hello world", "limit": 10}}'
    )


def get_format_intro(fmt: ToolFormat) -> str:
    """Return the standard intro line used before presenting tools/data.

    Returns:
        'The data format used in this conversation is {NAME}.
         Here is how it works:\\n{EXPLANATION}'
    """
    name = get_format_name(fmt)
    explanation = get_format_explanation(fmt)
    return (
        f"The data format used in this conversation is {name}. "
        f"Here is how it works:\n{explanation}"
    )


def get_format_reminder(fmt: ToolFormat) -> str:
    """Return a validation reminder to append to prompts.

    Returns:
        'Your response MUST be valid {NAME}.'
    """
    return f"Your response MUST be valid {get_format_name(fmt)}."


def serialize_tools(tools: Union[List[Dict[str, Any]], Dict[str, List]], fmt: ToolFormat) -> str:
    """Serialize tool definitions for inclusion in prompts.

    Accepts either:
      - A list of tool-definition dicts (keys: name, description, parameters, optionally server)
      - A dict mapping server_name -> list of tool objects (with .name, .description, .inputSchema)

    Format-specific strategy:
      - TRON: Serializes all tools as a single list via one TRON.stringify() call.
        This enables class definitions for repeated structures (~24% fewer tokens).
      - JSON/TOON: Serializes each tool individually, joined with '---' separator.
        (TOON list syntax adds overhead, so individual is more compact.)

    Args:
        tools: Tool definitions in one of the two accepted formats.
        fmt: Target serialization format.

    Returns:
        Formatted string of all tool definitions.
    """
    # Normalize to a flat list of dicts
    tool_defs: List[Dict[str, Any]] = []

    if isinstance(tools, dict):
        # Dict[server_name, List[Tool]] — MCP-Universe style
        for server_name, tool_list in tools.items():
            for tool in tool_list:
                tool_defs.append({
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                })
    else:
        # List[Dict] — MCPToolBenchPP / mcp-bench style
        tool_defs = list(tools)

    if fmt == ToolFormat.TRON:
        # Serialize as a single list so TRON generates class definitions
        return serialize(tool_defs, fmt)
    else:
        # JSON/TOON: individual serialization with separator
        return "\n---\n".join(serialize(td, fmt) for td in tool_defs)


def serialize_example(example_dict: Any, fmt: ToolFormat) -> str:
    """Serialize an example response structure.

    Convenience wrapper around serialize() for generating example
    output to show the LLM.

    Args:
        example_dict: The example data to serialize.
        fmt: Target serialization format.

    Returns:
        Serialized example string.
    """
    return serialize(example_dict, fmt)
