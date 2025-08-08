#date: 2025-08-08T17:06:25Z
#url: https://api.github.com/gists/2f77c555300579a4dcd74af980fd50be
#owner: https://api.github.com/users/DaveOkpare

import enum
import json
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, ValidationError

# ---------------- Schema ----------------

class Mode(enum.Enum):
    FAST = "FAST"
    SLOW = "SLOW"

class Inner(BaseModel):
    mode: Mode

class ToolArg(BaseModel):
    name: str
    config: Dict[str, Any]          # may include a "mode"
    inners: List[Inner]

class ToolArgs(BaseModel):
    tools: List[ToolArg]

# ---------------- Payloads ----------------

def payload_needs_coercion(n: int) -> Dict[str, Any]:
    # All enum fields are STRINGS (typical model output)
    return {
        "tools": [
            {
                "name": f"t{i}",
                "config": {"threshold": i / 10, "mode": ("FAST" if i % 2 else "SLOW")},
                "inners": [{"mode": "FAST"}, {"mode": "SLOW"}],
            }
            for i in range(n)
        ]
    }

def payload_already_valid(n: int) -> Dict[str, Any]:
    # All enum fields are actual Enum instances (best for python path)
    return {
        "tools": [
            {
                "name": f"t{i}",
                "config": {"threshold": i / 10, "mode": (Mode.FAST if i % 2 else Mode.SLOW)},
                "inners": [{"mode": Mode.FAST}, {"mode": Mode.SLOW}],
            }
            for i in range(n)
        ]
    }

def dumps_enum_aware(o: Any) -> str:
    # Let always_json run even when objects contain Enum instances
    return json.dumps(o, default=lambda x: x.value if isinstance(x, enum.Enum) else x)

# ---------------- Strategies ----------------

def always_json(obj: Dict[str, Any], *, strict: bool = True):
    return ToolArgs.model_validate_json(dumps_enum_aware(obj), strict=strict)

def hybrid(obj: Dict[str, Any], *, strict: bool = True):
    try:
        return ToolArgs.model_validate(obj, strict=strict)
    except ValidationError:
        return ToolArgs.model_validate_json(json.dumps(obj), strict=strict)

# ---------------- Sanity (correctness) ----------------

def test_strict_repro_correctness():
    obj = payload_needs_coercion(5)
    # Python strict should fail on string enums
    with pytest.raises(ValidationError):
        ToolArgs.model_validate(obj, strict=True)
    # JSON strict should pass (valid enum member strings)
    out = ToolArgs.model_validate_json(json.dumps(obj), strict=True)
    assert isinstance(out, ToolArgs)

# ---------------- Benchmarks ----------------
# We parametrize a "variant" to make JSON export & grouping easy to read.

@pytest.mark.parametrize("n", [5, 500])
@pytest.mark.parametrize("variant", ["model_validate_json", "try_except"])
def test_perf_needs_coercion_strict(benchmark, n, variant):
    obj = payload_needs_coercion(n)
    if variant == "model_validate_json":
        benchmark.extra_info = {"variant": "model_validate_json", "case": "needs_coercion", "n": n}
        benchmark(lambda: always_json(obj, strict=True))
    else:
        benchmark.extra_info = {"variant": "try_except", "case": "needs_coercion", "n": n}
        benchmark(lambda: hybrid(obj, strict=True))

@pytest.mark.parametrize("n", [5, 500])
@pytest.mark.parametrize("variant", ["model_validate_json", "try_except"])
def test_perf_already_valid_strict(benchmark, n, variant):
    obj = payload_already_valid(n)
    if variant == "model_validate_json":
        benchmark.extra_info = {"variant": "model_validate_json", "case": "already_valid", "n": n}
        benchmark(lambda: always_json(obj, strict=True))
    else:
        benchmark.extra_info = {"variant": "try_except", "case": "already_valid", "n": n}
        benchmark(lambda: hybrid(obj, strict=True))