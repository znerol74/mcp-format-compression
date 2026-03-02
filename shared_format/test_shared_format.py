#!/usr/bin/env python3
"""Verify shared_format works correctly and produces identical output."""
import sys

from shared_format import (
    ToolFormat, serialize, deserialize, deserialize_lenient,
    get_format_name, get_format_explanation, get_format_intro,
    get_format_reminder, serialize_tools, serialize_example,
    count_tokens,
)

# Test data: a typical MCP tool definition
SAMPLE_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}

SAMPLE_TOOL_LIST = [
    SAMPLE_TOOL,
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"}
            },
            "required": ["query"]
        }
    }
]

def main():
    print(f"Python: {sys.executable}")
    print(f"shared_format imported successfully\n")

    errors = 0

    # Test serialize/deserialize for each format
    for fmt in [ToolFormat.JSON, ToolFormat.TOON, ToolFormat.TRON]:
        print(f"=== {fmt.value.upper()} ===")
        try:
            serialized = serialize(SAMPLE_TOOL, fmt)
            print(f"  serialize: OK ({len(serialized)} chars)")

            # Verify round-trip
            deserialized = deserialize(serialized, fmt)
            assert deserialized["name"] == "get_weather", f"Round-trip failed for {fmt}"
            print(f"  deserialize round-trip: OK")

            # Token count
            tokens = count_tokens(serialized)
            print(f"  token count: {tokens}")

            # Print first 200 chars of output for visual inspection
            preview = serialized[:200].replace('\n', '\\n')
            print(f"  output: {preview}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    # Test prompt snippets
    print(f"\n=== Prompt Snippets ===")
    for fmt in [ToolFormat.JSON, ToolFormat.TOON, ToolFormat.TRON]:
        name = get_format_name(fmt)
        explanation = get_format_explanation(fmt)
        intro = get_format_intro(fmt)
        reminder = get_format_reminder(fmt)
        print(f"  {name}: explanation={len(explanation)} chars, intro={len(intro)} chars")
        print(f"    reminder: {reminder}")

    # Test serialize_tools
    print(f"\n=== serialize_tools ===")
    for fmt in [ToolFormat.JSON, ToolFormat.TOON, ToolFormat.TRON]:
        try:
            tools_str = serialize_tools(SAMPLE_TOOL_LIST, fmt)
            tokens = count_tokens(tools_str)
            print(f"  {fmt.value.upper()}: {len(tools_str)} chars, {tokens} tokens")
        except Exception as e:
            print(f"  {fmt.value.upper()} ERROR: {e}")
            errors += 1

    # Test serialize_example
    print(f"\n=== serialize_example ===")
    example = {"thought": "test reasoning", "action": {"tool": "get_weather", "arguments": {"location": "Berlin"}}}
    for fmt in [ToolFormat.JSON, ToolFormat.TOON, ToolFormat.TRON]:
        try:
            ex_str = serialize_example(example, fmt)
            print(f"  {fmt.value.upper()}: {len(ex_str)} chars")
        except Exception as e:
            print(f"  {fmt.value.upper()} ERROR: {e}")
            errors += 1

    # Test count_tokens uses tiktoken
    print(f"\n=== Token Counter ===")
    test_text = "Hello world, this is a test of the token counter."
    tokens = count_tokens(test_text)
    print(f"  '{test_text}' -> {tokens} tokens")
    assert tokens > 0, "Token count should be positive"
    # tiktoken should give ~11 tokens, len//4 would give ~12
    print(f"  (tiktoken should give ~11, len//4 would give ~{len(test_text)//4})")

    # Verify JSON is compact (no indent)
    print(f"\n=== JSON Compactness Check ===")
    json_out = serialize({"key": "value", "nested": {"a": 1}}, ToolFormat.JSON)
    assert "\n" not in json_out, f"JSON should be compact (no newlines), got: {json_out}"
    assert "  " not in json_out, f"JSON should be compact (no indent), got: {json_out}"
    print(f"  JSON output is compact: {json_out}")

    print(f"\n{'='*40}")
    if errors == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILED: {errors} errors")
    return errors


if __name__ == "__main__":
    sys.exit(main())
