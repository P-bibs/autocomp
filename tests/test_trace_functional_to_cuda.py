from __future__ import annotations

from pathlib import Path

import pytest

from trace_functional_to_cuda.cli import (
    _public_parser,
    _summarize_failures,
    build_output_json,
    TraceFunctionalToCudaError,
)


def _base_trace_bundle() -> dict:
    return {
        "tensor_registry": {
            "input_tensor_names": ["x"],
            "return_canonical_name": "out_tensor",
            "canonical_tensors": {
                "x": {"numel": 16},
                "out_tensor": {"numel": 8},
            },
            "pointer_to_canonical": {
                "0x1000": "x",
                "0x2000": "out_tensor",
            },
            "scratch_order": [],
        },
        "kernel_events": [],
    }


def _base_input_payload() -> dict:
    return {
        "source_tensors": {
            "x": {"size": 16},
        },
        "dest_tensors": {
            "out": {"size": 8},
        },
        "expected_result": "equivalent",
        "pipelines": [],
    }


def test_build_output_json_matches_pointer_arguments_without_pointer_hint() -> None:
    trace_bundle = _base_trace_bundle()
    trace_bundle["kernel_events"] = [
        {
            "args_known": True,
            "ptx": ".version 8.5\n.entry fake_kernel() {}\n",
            "ptx_entry": "fake_kernel",
            "kernel_name": "fake_kernel",
            "grid": {"x": 1, "y": 1, "z": 1},
            "block": {"x": 64, "y": 1, "z": 1},
            "args": [
                {
                    "pointer_hint": False,
                    "pointer_value": "0x1000",
                    "ptx_type": ".u64",
                    "size": 8,
                    "u64": 0x1000,
                    "s64": 0x1000,
                },
                {
                    "pointer_hint": False,
                    "pointer_value": "0x2000",
                    "ptx_type": ".u64",
                    "size": 8,
                    "u64": 0x2000,
                    "s64": 0x2000,
                },
                {
                    "pointer_hint": False,
                    "pointer_value": "0x80",
                    "ptx_type": ".u32",
                    "size": 4,
                    "u32": 128,
                    "s32": 128,
                },
            ],
        }
    ]

    output = build_output_json(trace_bundle, _base_input_payload())
    kernel = output["pipelines"][-1]["kernels"][0]

    assert kernel["arguments"] == ["x", "out", "trace_const_000"]
    assert output["constants"] == {"trace_const_000": 128}


def test_build_output_json_keeps_non_tensor_u64_as_scalar_constant() -> None:
    trace_bundle = _base_trace_bundle()
    trace_bundle["kernel_events"] = [
        {
            "args_known": True,
            "ptx": ".version 8.5\n.entry fake_kernel() {}\n",
            "ptx_entry": "fake_kernel",
            "kernel_name": "fake_kernel",
            "grid": {"x": 1, "y": 1, "z": 1},
            "block": {"x": 64, "y": 1, "z": 1},
            "args": [
                {
                    "pointer_hint": False,
                    "pointer_value": "0x9999",
                    "ptx_type": ".u64",
                    "size": 8,
                    "u64": 0x9999,
                    "s64": 0x9999,
                }
            ],
        }
    ]

    output = build_output_json(trace_bundle, _base_input_payload())
    kernel = output["pipelines"][-1]["kernels"][0]

    assert kernel["arguments"] == ["trace_const_000"]
    assert output["constants"] == {"trace_const_000": 0x9999}


def test_build_output_json_reports_decode_reason() -> None:
    trace_bundle = _base_trace_bundle()
    trace_bundle["kernel_events"] = [
        {
            "args_known": False,
            "args_known_reason": "ptx_symbol_lookup_failed",
            "ptx_entry": "fake_kernel",
            "kernel_name": "fake_kernel",
        }
    ]

    with pytest.raises(TraceFunctionalToCudaError) as excinfo:
        build_output_json(trace_bundle, _base_input_payload())

    assert (
        str(excinfo.value)
        == "Kernel launch trace did not include decoded argument metadata for fake_kernel: ptx_symbol_lookup_failed"
    )


def test_summarize_failures_groups_categories_and_unsupported_ops() -> None:
    failures = [
        (Path("level1/1/correct/0.py"), "Child trace process failed\nchild_output:\nDisallowed torch op encountered: convolution"),
        (Path("level1/1/correct/1.py"), "Child trace process failed\nchild_output:\nDisallowed torch op encountered: convolution"),
        (Path("level1/2/correct/0.py"), "Determinism check failed: the traced output JSON changed between runs."),
        (Path("level1/3/incorrect/0.py"), "dest_tensors.out.size mismatch: json=16 traced=32"),
    ]

    summary = _summarize_failures(failures)

    assert summary["failure_categories"] == {
        "dest_size_mismatch": 1,
        "determinism_check_failed": 1,
        "unsupported_torch_op": 2,
    }
    assert summary["unsupported_torch_ops"] == {"convolution": 2}


def test_public_parser_accepts_max_gpu_concurrency() -> None:
    parser = _public_parser()
    args = parser.parse_args(
        [
            "models",
            "specs",
            "kb",
            "out",
            "--max-gpu-concurrency",
            "7",
        ]
    )

    assert args.max_gpu_concurrency == 7
