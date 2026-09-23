"""Tiny test-only ijson compatibility shim for miniature JSON fixtures.

Never use this shim for the full corpus. It intentionally loads the complete
JSON document in memory and exists only so the miniature validation can run in
environments where the optional streaming dependency is unavailable.
"""
import json

def items(file_obj, prefix):
    if prefix != "item":
        raise NotImplementedError("validation shim only supports prefix='item'")
    data = json.load(file_obj)
    if not isinstance(data, list):
        raise ValueError("validation shim expects a top-level JSON list")
    yield from data
