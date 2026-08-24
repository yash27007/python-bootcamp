#!/usr/bin/env python3
"""Minimal MCP-shaped JSON-RPC 2.0 server, communicating over stdio.

This is a real subprocess, talking real JSON-RPC 2.0 over stdin/stdout (one
JSON object per line) -- the actual wire shape and transport the Model
Context Protocol specifies (stdio transport), not an in-process simulation.
No network socket is opened anywhere; this process only reads its own stdin
and writes its own stdout, so it satisfies this section's "no live external
service calls" constraint while still crossing a real process boundary.

Implements two JSON-RPC methods against the "tools" MCP primitive:

  - list_tools  -- discovery: returns every registered tool's name,
                   description, and JSON-Schema input contract.
  - call_tool   -- invocation: validates the caller's arguments against that
                   tool's JSON-Schema (required-field presence + type
                   checking) before ever calling the underlying Python
                   function, and returns a JSON-RPC result or a JSON-RPC
                   error (code -32602, "Invalid params") on a schema
                   violation.

Three toy tools are registered: add(a, b), word_count(text), and
reverse_string(text).
"""
import json
import sys

TOOLS = {
    "add": {
        "description": "Add two numbers together and return their sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First addend"},
                "b": {"type": "number", "description": "Second addend"},
            },
            "required": ["a", "b"],
        },
    },
    "word_count": {
        "description": "Count the number of whitespace-separated words in a string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to count words in"},
            },
            "required": ["text"],
        },
    },
    "reverse_string": {
        "description": "Reverse the characters of a string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to reverse"},
            },
            "required": ["text"],
        },
    },
}

# JSON-Schema "type" -> acceptable Python type(s) for a basic type check.
JSON_SCHEMA_TYPE_MAP = {
    "number": (int, float),
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema, arguments):
    """Basic JSON-Schema validation: required-field presence + type checking.

    Returns None if `arguments` satisfies `schema`, else a human-readable
    error message string. This is intentionally basic (no $ref, no nested
    schema, no format validators) -- it demonstrates the *mechanism* of
    validating before dispatch, not a full JSON-Schema implementation.
    """
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"

    for field in schema.get("required", []):
        if field not in arguments:
            return f"missing required argument: {field!r}"

    properties = schema.get("properties", {})
    for key, value in arguments.items():
        if key not in properties:
            continue  # unknown extra argument: ignored, not rejected
        expected_type = properties[key]["type"]
        py_type = JSON_SCHEMA_TYPE_MAP.get(expected_type)
        if py_type is None:
            continue
        if expected_type == "number" and isinstance(value, bool):
            # bool is a subclass of int in Python; a JSON boolean is not a
            # JSON number, so this must be rejected explicitly.
            return f"argument {key!r} expected type 'number', got 'boolean'"
        if not isinstance(value, py_type):
            return (
                f"argument {key!r} expected type {expected_type!r}, "
                f"got {type(value).__name__!r}"
            )
    return None


def call_add(arguments):
    return arguments["a"] + arguments["b"]


def call_word_count(arguments):
    return len(arguments["text"].split())


def call_reverse_string(arguments):
    return arguments["text"][::-1]


TOOL_FUNCTIONS = {
    "add": call_add,
    "word_count": call_word_count,
    "reverse_string": call_reverse_string,
}


def make_error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def make_result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def handle_request(req):
    id_ = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "list_tools":
        tools_list = [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["inputSchema"],
            }
            for name, spec in TOOLS.items()
        ]
        return make_result(id_, {"tools": tools_list})

    if method == "call_tool":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name not in TOOLS:
            return make_error(id_, -32601, f"unknown tool: {tool_name!r}")

        schema = TOOLS[tool_name]["inputSchema"]
        validation_error = validate_arguments(schema, arguments)
        if validation_error is not None:
            return make_error(
                id_, -32602, f"invalid params for tool {tool_name!r}: {validation_error}"
            )

        try:
            value = TOOL_FUNCTIONS[tool_name](arguments)
        except Exception as exc:  # tool body raised at runtime
            return make_error(id_, -32000, f"tool execution error: {exc}")

        return make_result(
            id_, {"content": [{"type": "text", "text": str(value)}], "value": value}
        )

    return make_error(id_, -32601, f"method not found: {method!r}")


def main():
    """Read one JSON-RPC request per line from stdin, write one JSON-RPC
    response per line to stdout. Newline-delimited JSON over stdio is the
    real MCP stdio transport framing."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = make_error(None, -32700, f"parse error: {exc}")
        else:
            resp = handle_request(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
