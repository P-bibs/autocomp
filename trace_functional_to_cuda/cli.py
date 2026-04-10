from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode


ROOT = Path(__file__).resolve().parents[1]
PRELOAD_SOURCE = ROOT / "trace_functional_to_cuda" / "cuda_trace_module_preload.cpp"
CREATOR_OPS = {
    "arange",
    "empty",
    "empty_like",
    "eye",
    "full",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "zeros",
    "zeros_like",
}
NORMALIZED_NOOP_OPS = {
    "_to_copy",
    "alias",
    "as_strided",
    "clone",
    "contiguous",
    "detach",
    "expand",
    "permute",
    "reshape",
    "squeeze",
    "transpose",
    "unsqueeze",
    "view",
}


class TraceFunctionalToCudaError(RuntimeError):
    pass


@dataclass
class TensorRecord:
    name: str
    canonical_name: str
    shape: list[int]
    numel: int
    dtype: str
    device: str
    data_ptr: int
    kind: str
    alias_of: str | None = None
    source_op: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "shape": self.shape,
            "numel": self.numel,
            "dtype": self.dtype,
            "device": self.device,
            "data_ptr": self.data_ptr,
            "kind": self.kind,
        }
        if self.alias_of is not None:
            payload["alias_of"] = self.alias_of
        if self.source_op is not None:
            payload["source_op"] = self.source_op
        return payload


def _load_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"trace_functional_to_cuda_{uuid.uuid4().hex}", path)
    if spec is None or spec.loader is None:
        raise TraceFunctionalToCudaError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_data_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.data_ptr())


def _tensor_meta(tensor: torch.Tensor) -> tuple[list[int], int, str, str, int]:
    return list(tensor.shape), int(tensor.numel()), str(tensor.dtype), str(tensor.device), _tensor_data_ptr(tensor)


class TensorRegistry:
    def __init__(self) -> None:
        self.pointer_to_canonical: dict[int, str] = {}
        self.canonical_records: dict[str, TensorRecord] = {}
        self.observed_records: list[TensorRecord] = []
        self.input_tensor_names: list[str] = []
        self.scratch_order: list[str] = []
        self._scratch_counter = 0
        self._alias_counter = 0
        self.return_canonical_name: str | None = None

    def register_input(self, name: str, tensor: torch.Tensor) -> None:
        shape, numel, dtype, device, ptr = _tensor_meta(tensor)
        existing_name = self.pointer_to_canonical.get(ptr)
        if existing_name is not None and existing_name != name:
            raise TraceFunctionalToCudaError(
                f"Distinct input names resolved to the same canonical tensor pointer: {existing_name} and {name}"
            )
        record = TensorRecord(
            name=name,
            canonical_name=name,
            shape=shape,
            numel=numel,
            dtype=dtype,
            device=device,
            data_ptr=ptr,
            kind="input",
        )
        self.pointer_to_canonical[ptr] = name
        self.canonical_records[name] = record
        self.observed_records.append(record)
        self.input_tensor_names.append(name)

    def register_creator_outputs(self, output: Any, op_name: str) -> None:
        for tensor in _iter_tensors(output):
            canonical_name = f"scratch_{self._scratch_counter:03d}"
            self._scratch_counter += 1
            shape, numel, dtype, device, ptr = _tensor_meta(tensor)
            record = TensorRecord(
                name=canonical_name,
                canonical_name=canonical_name,
                shape=shape,
                numel=numel,
                dtype=dtype,
                device=device,
                data_ptr=ptr,
                kind="scratch",
                source_op=op_name,
            )
            self.pointer_to_canonical[ptr] = canonical_name
            self.canonical_records[canonical_name] = record
            self.observed_records.append(record)
            self.scratch_order.append(canonical_name)

    def register_alias_outputs(self, output: Any, source_tensor: torch.Tensor, op_name: str) -> None:
        source_name = self.canonical_name_for_tensor(source_tensor)
        if source_name is None:
            raise TraceFunctionalToCudaError(f"Unable to resolve alias source tensor for normalized op {op_name}")
        for tensor in _iter_tensors(output):
            alias_name = f"alias_{self._alias_counter:03d}"
            self._alias_counter += 1
            shape, numel, dtype, device, ptr = _tensor_meta(tensor)
            self.pointer_to_canonical[ptr] = source_name
            record = TensorRecord(
                name=alias_name,
                canonical_name=source_name,
                shape=shape,
                numel=numel,
                dtype=dtype,
                device=device,
                data_ptr=ptr,
                kind="alias",
                alias_of=source_name,
                source_op=op_name,
            )
            self.observed_records.append(record)

    def canonical_name_for_tensor(self, tensor: torch.Tensor) -> str | None:
        return self.pointer_to_canonical.get(_tensor_data_ptr(tensor))

    def canonical_name_for_pointer(self, pointer_value: int) -> str | None:
        return self.pointer_to_canonical.get(pointer_value)

    def mark_return(self, tensor: torch.Tensor) -> None:
        canonical_name = self.canonical_name_for_tensor(tensor)
        if canonical_name is None:
            raise TraceFunctionalToCudaError("Returned tensor was not observed through an allowed torch creator or normalized no-op")
        self.return_canonical_name = canonical_name

    def bundle(self) -> dict[str, Any]:
        if self.return_canonical_name is None:
            raise TraceFunctionalToCudaError("Missing return tensor registration")
        return {
            "input_tensor_names": list(self.input_tensor_names),
            "return_canonical_name": self.return_canonical_name,
            "canonical_tensors": {
                name: record.to_json() for name, record in self.canonical_records.items()
            },
            "observed_tensors": [record.to_json() for record in self.observed_records],
            "pointer_to_canonical": {
                hex(pointer): canonical for pointer, canonical in sorted(self.pointer_to_canonical.items())
            },
            "scratch_order": list(self.scratch_order),
        }


class DispatchTracer(TorchDispatchMode):
    def __init__(self, registry: TensorRegistry) -> None:
        super().__init__()
        self.registry = registry
        self.events: list[dict[str, Any]] = []
        self._sequence = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}
        op_name = _aten_name(func)
        category = _classify_op(op_name)
        source_tensor = _first_tensor([args, kwargs])
        if category == "reject":
            raise TraceFunctionalToCudaError(f"Disallowed torch op encountered: {op_name}")
        if category == "noop":
            _validate_noop_operation(op_name, args, kwargs, source_tensor)

        result = func(*args, **kwargs)

        if category == "creator":
            self.registry.register_creator_outputs(result, op_name)
        else:
            if source_tensor is None:
                raise TraceFunctionalToCudaError(f"Normalized no-op {op_name} did not receive a tensor argument")
            self.registry.register_alias_outputs(result, source_tensor, op_name)

        self._sequence += 1
        event = {
            "kind": "torch",
            "sequence": self._sequence,
            "monotonic_ns": time.monotonic_ns(),
            "op": op_name,
            "category": category,
            "args": _summarize_value(args, self.registry),
            "kwargs": _summarize_value(kwargs, self.registry),
            "outputs": _summarize_value(result, self.registry),
        }
        self.events.append(event)
        return result


def _aten_name(func: Any) -> str:
    schema_name = func._schema.name
    return schema_name.split("::", 1)[1] if "::" in schema_name else schema_name


def _classify_op(op_name: str) -> str:
    if op_name in CREATOR_OPS:
        return "creator"
    if op_name in NORMALIZED_NOOP_OPS:
        return "noop"
    return "reject"


def _first_tensor(values: Iterable[Any]) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            result = _first_tensor(value.values())
            if result is not None:
                return result
        elif isinstance(value, (list, tuple)):
            result = _first_tensor(value)
            if result is not None:
                return result
    return None


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _validate_noop_operation(
    op_name: str,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    source_tensor: torch.Tensor | None,
) -> None:
    if source_tensor is None:
        raise TraceFunctionalToCudaError(f"Normalized no-op {op_name} is missing its source tensor")
    if op_name == "_to_copy":
        requested_dtype = kwargs.get("dtype", source_tensor.dtype)
        requested_device = kwargs.get("device", source_tensor.device)
        requested_layout = kwargs.get("layout", source_tensor.layout)
        if requested_dtype != source_tensor.dtype or torch.device(requested_device) != source_tensor.device:
            raise TraceFunctionalToCudaError("tensor.to(...) is only allowed when device and dtype are unchanged")
        if requested_layout != source_tensor.layout:
            raise TraceFunctionalToCudaError("tensor.to(...) is only allowed when layout is unchanged")


def _summarize_value(value: Any, registry: TensorRegistry) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor": registry.canonical_name_for_tensor(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _summarize_value(item, registry) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item, registry) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _ensure_cuda_available(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise TraceFunctionalToCudaError(
            "CUDA is not available to PyTorch in this environment; cannot trace CUDA kernels."
        )
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise TraceFunctionalToCudaError(f"Expected a CUDA device, got {device}")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise TraceFunctionalToCudaError(f"Requested device {device} but only {torch.cuda.device_count()} CUDA devices are visible")
    return resolved


def _function_input_order(module: ModuleType, function_name: str, state_kwargs: dict[str, Any]) -> list[tuple[str, Any]]:
    function = getattr(module, function_name)
    positional_names = list(getattr(module, "FORWARD_ARG_NAMES", []))
    signature = list(arg.name for arg in function.__signature__.parameters.values()) if hasattr(function, "__signature__") else []
    if not positional_names and signature:
        positional_names = [
            name
            for name in signature
            if name not in state_kwargs
        ]
    ordered: list[tuple[str, Any]] = []
    ordered.extend((name, None) for name in positional_names)
    kw_order = [name for name in getattr(module, "REQUIRED_STATE_NAMES", state_kwargs.keys()) if name in state_kwargs]
    ordered.extend((name, state_kwargs[name]) for name in kw_order)
    return ordered


def _load_kernel_events(path: Path, pid: int, start_ns: int, end_ns: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("event") != "kernel_launch":
            continue
        if payload.get("pid") != pid:
            continue
        monotonic_ns = payload.get("monotonic_ns")
        if not isinstance(monotonic_ns, int):
            continue
        if start_ns <= monotonic_ns <= end_ns:
            events.append(payload)
    events.sort(key=lambda event: (event["monotonic_ns"], event.get("sequence", 0)))
    return events


def _merge_event_streams(torch_events: list[dict[str, Any]], kernel_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = list(torch_events)
    for event in kernel_events:
        merged.append(
            {
                "kind": "kernel",
                "sequence": event.get("sequence"),
                "monotonic_ns": event.get("monotonic_ns"),
                "kernel": event.get("kernel_name") or event.get("ptx_entry"),
                "api": event.get("api"),
                "grid": event.get("grid"),
                "block": event.get("block"),
            }
        )
    merged.sort(key=lambda event: (event.get("monotonic_ns", 0), 0 if event["kind"] == "torch" else 1, event.get("sequence", 0) or 0))
    return merged


def _child_trace(model_file: Path, function_name: str, device: str, bundle_output: Path) -> int:
    resolved_device = _ensure_cuda_available(device)
    kernel_trace_path = Path(os.environ["CUDA_TRACE_JSON_PATH"])
    module = _load_module_from_path(model_file)

    if not hasattr(module, "get_functional_inputs"):
        raise TraceFunctionalToCudaError(f"{model_file} does not define get_functional_inputs()")
    if not hasattr(module, function_name):
        raise TraceFunctionalToCudaError(f"{model_file} does not define {function_name}()")

    forward_args, state_kwargs = module.get_functional_inputs()
    if not isinstance(state_kwargs, dict):
        raise TraceFunctionalToCudaError("get_functional_inputs() must return (forward_args, state_kwargs)")
    forward_args = tuple(forward_args)
    state_kwargs = dict(state_kwargs)

    moved_args = tuple(_move_to_device(value, resolved_device) for value in forward_args)
    moved_state = {key: _move_to_device(value, resolved_device) for key, value in state_kwargs.items()}

    registry = TensorRegistry()

    positional_names = list(getattr(module, "FORWARD_ARG_NAMES", []))
    if positional_names and len(positional_names) != len(moved_args):
        raise TraceFunctionalToCudaError(
            f"FORWARD_ARG_NAMES has {len(positional_names)} names but get_functional_inputs() returned {len(moved_args)} forward args"
        )

    for index, value in enumerate(moved_args):
        if isinstance(value, torch.Tensor):
            name = positional_names[index] if positional_names else f"arg_{index}"
            registry.register_input(name, value)

    required_state_names = list(getattr(module, "REQUIRED_STATE_NAMES", moved_state.keys()))
    for name in required_state_names:
        value = moved_state.get(name)
        if isinstance(value, torch.Tensor):
            registry.register_input(name, value)

    torch.cuda.synchronize(resolved_device)
    tracer = DispatchTracer(registry)
    function = getattr(module, function_name)
    start_ns = time.monotonic_ns()
    with tracer:
        result = function(*moved_args, **moved_state)
    torch.cuda.synchronize(resolved_device)
    end_ns = time.monotonic_ns()

    if not isinstance(result, torch.Tensor):
        raise TraceFunctionalToCudaError("functional_model must return exactly one tensor")
    registry.mark_return(result)

    kernel_events = _load_kernel_events(kernel_trace_path, os.getpid(), start_ns, end_ns)
    bundle = {
        "model_file": str(model_file),
        "device": str(resolved_device),
        "trace_window": {"start_ns": start_ns, "end_ns": end_ns},
        "tensor_registry": registry.bundle(),
        "torch_events": tracer.events,
        "kernel_events": kernel_events,
        "events": _merge_event_streams(tracer.events, kernel_events),
    }
    bundle_output.write_text(json.dumps(bundle, indent=2))
    return 0


def _guess_cuda_home() -> Path:
    candidates: list[Path] = []
    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home:
        candidates.append(Path(env_home))
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        candidates.append(Path(nvcc_path).resolve().parents[1])
    try:
        from torch.utils.cpp_extension import CUDA_HOME  # type: ignore

        if CUDA_HOME:
            candidates.append(Path(CUDA_HOME))
    except Exception:
        pass

    for candidate in candidates:
        if (candidate / "include" / "cuda_runtime_api.h").exists():
            return candidate
    raise TraceFunctionalToCudaError("Unable to locate a CUDA toolkit installation with cuda_runtime_api.h")


def _ensure_runtime_prerequisites() -> None:
    if shutil.which("ninja") is None:
        raise TraceFunctionalToCudaError(
            "ninja is required to import and build inline CUDA extensions from these model files. "
            "Install it and retry."
        )


def build_preload_library() -> Path:
    if not PRELOAD_SOURCE.exists():
        raise TraceFunctionalToCudaError(f"Missing preload source file: {PRELOAD_SOURCE}")

    cuda_home = _guess_cuda_home()
    cache_dir = Path(tempfile.gettempdir()) / "trace_functional_to_cuda"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"cuda_trace_module_preload_{PRELOAD_SOURCE.stat().st_mtime_ns}.so"
    if output_path.exists():
        return output_path

    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise TraceFunctionalToCudaError("Unable to find a C++ compiler for building the preload tracer")

    command = [
        compiler,
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        "-I",
        str(cuda_home / "include"),
        str(PRELOAD_SOURCE),
        "-o",
        str(output_path),
        "-ldl",
        "-pthread",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise TraceFunctionalToCudaError(
            "Failed to compile the preload tracer:\n"
            + textwrap.dedent(
                f"""
                command: {' '.join(command)}
                stdout:
                {result.stdout}
                stderr:
                {result.stderr}
                """
            ).strip()
        )
    return output_path


def _run_child_trace(model_file: Path, function_name: str, device: str, preload_library: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="trace_functional_to_cuda_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        kernel_trace_path = tmp_dir / "kernel_launches.jsonl"
        bundle_path = tmp_dir / "trace_bundle.json"
        env = os.environ.copy()
        existing_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = (
            f"{preload_library}:{existing_preload}" if existing_preload else str(preload_library)
        )
        env["CUDA_TRACE_JSON_PATH"] = str(kernel_trace_path)

        command = [
            sys.executable,
            "-m",
            "trace_functional_to_cuda",
            "__child_trace__",
            "--model-file",
            str(model_file),
            "--function-name",
            function_name,
            "--device",
            device,
            "--bundle-output",
            str(bundle_path),
        ]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise TraceFunctionalToCudaError(
                "Child trace process failed:\n"
                + textwrap.dedent(
                    f"""
                    command: {' '.join(command)}
                    stdout:
                    {result.stdout}
                    stderr:
                    {result.stderr}
                    """
                ).strip()
            )
        return json.loads(bundle_path.read_text())


def _typed_constant_key(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    raise TraceFunctionalToCudaError(f"Unsupported scalar constant type: {type(value).__name__}")


class ConstantRegistry:
    def __init__(self, constants: dict[str, Any]) -> None:
        self.constants = constants
        self.value_to_name: dict[tuple[str, Any], str] = {}
        for name, value in constants.items():
            if isinstance(value, (bool, int, float)):
                self.value_to_name.setdefault(_typed_constant_key(value), name)
        self._counter = 0

    def name_for_value(self, value: Any) -> str:
        key = _typed_constant_key(value)
        existing = self.value_to_name.get(key)
        if existing is not None:
            return existing
        while True:
            name = f"trace_const_{self._counter:03d}"
            self._counter += 1
            if name not in self.constants:
                break
        self.constants[name] = value
        self.value_to_name[key] = name
        return name


def _decode_scalar_argument(arg: dict[str, Any]) -> Any:
    source_type = str(arg.get("source_type", ""))
    ptx_type = str(arg.get("ptx_type", ""))
    if "bool" in source_type:
        if "u8" in arg:
            return bool(arg["u8"])
        if "u32" in arg:
            return bool(arg["u32"])
    if ptx_type == ".f32" and "f32" in arg:
        return float(arg["f32"])
    if ptx_type == ".f64" and "f64" in arg:
        return float(arg["f64"])
    if ptx_type.startswith(".s"):
        if "s32" in arg:
            return int(arg["s32"])
        if "s64" in arg:
            return int(arg["s64"])
    if ptx_type.startswith(".u") or ptx_type.startswith(".b"):
        if "u8" in arg:
            return int(arg["u8"])
        if "u16" in arg:
            return int(arg["u16"])
        if "u32" in arg:
            return int(arg["u32"])
        if "u64" in arg:
            return int(arg["u64"])
    if "s32" in arg:
        return int(arg["s32"])
    if "s64" in arg:
        return int(arg["s64"])
    raise TraceFunctionalToCudaError(f"Unable to decode scalar kernel argument payload: {arg}")


def _parse_pointer_string(pointer_value: str) -> int:
    if not pointer_value.startswith("0x"):
        raise TraceFunctionalToCudaError(f"Unexpected pointer encoding: {pointer_value}")
    return int(pointer_value, 16)


def _pipeline_tensor_name(canonical_name: str, return_canonical_name: str, dest_name: str) -> str:
    return dest_name if canonical_name == return_canonical_name else canonical_name


def _validate_tensor_size_field(
    section_name: str,
    tensor_name: str,
    expected_numel: int,
    payload: dict[str, Any],
) -> None:
    if "size" not in payload:
        raise TraceFunctionalToCudaError(f"{section_name}.{tensor_name} is missing required size")
    declared_size = payload["size"]
    if not isinstance(declared_size, int):
        raise TraceFunctionalToCudaError(f"{section_name}.{tensor_name}.size must be an integer")
    if declared_size != expected_numel:
        raise TraceFunctionalToCudaError(
            f"{section_name}.{tensor_name}.size mismatch: json={declared_size} traced={expected_numel}"
        )


def _is_torch_internal_kernel(kernel_event: dict[str, Any]) -> bool:
    module_path = str(kernel_event.get("module_path", ""))
    module_basename = str(kernel_event.get("module_basename", ""))
    if "/site-packages/torch/lib/" in module_path:
        return True
    if module_basename.startswith("libtorch"):
        return True
    return False


def build_output_json(trace_bundle: dict[str, Any], input_payload: dict[str, Any]) -> dict[str, Any]:
    tensor_registry = trace_bundle["tensor_registry"]
    input_tensor_names = set(tensor_registry["input_tensor_names"])
    source_tensors = input_payload.get("source_tensors", {})
    source_tensor_names = set(source_tensors.keys())
    if input_tensor_names != source_tensor_names:
        raise TraceFunctionalToCudaError(
            f"Input tensor names do not match source_tensors. expected={sorted(source_tensor_names)} actual={sorted(input_tensor_names)}"
        )

    dest_tensors = input_payload.get("dest_tensors", {})
    if len(dest_tensors) != 1:
        raise TraceFunctionalToCudaError("Input JSON must define exactly one dest tensor")
    dest_name = next(iter(dest_tensors))

    return_canonical_name = tensor_registry["return_canonical_name"]
    canonical_tensors = tensor_registry["canonical_tensors"]
    if return_canonical_name not in canonical_tensors and return_canonical_name not in input_tensor_names:
        raise TraceFunctionalToCudaError(f"Unknown return tensor canonical name: {return_canonical_name}")

    for input_name in sorted(input_tensor_names):
        _validate_tensor_size_field(
            "source_tensors",
            input_name,
            int(canonical_tensors[input_name]["numel"]),
            source_tensors[input_name],
        )
    _validate_tensor_size_field(
        "dest_tensors",
        dest_name,
        int(canonical_tensors[return_canonical_name]["numel"]),
        dest_tensors[dest_name],
    )

    output_payload = json.loads(json.dumps(input_payload))
    constants = output_payload.setdefault("constants", {})
    constant_registry = ConstantRegistry(constants)

    scratch_tensors: dict[str, dict[str, int]] = {}
    for scratch_name in tensor_registry["scratch_order"]:
        if scratch_name == return_canonical_name:
            continue
        record = canonical_tensors[scratch_name]
        scratch_tensors[scratch_name] = {"size": int(record["numel"])}

    kernels: list[dict[str, Any]] = []
    pointer_to_canonical = {
        _parse_pointer_string(pointer): canonical
        for pointer, canonical in tensor_registry["pointer_to_canonical"].items()
    }

    for kernel_event in trace_bundle["kernel_events"]:
        if _is_torch_internal_kernel(kernel_event):
            continue
        if not kernel_event.get("args_known"):
            raise TraceFunctionalToCudaError("Kernel launch trace did not include decoded argument metadata")
        ptx = kernel_event.get("ptx", "")
        if not ptx:
            raise TraceFunctionalToCudaError("Kernel launch trace is missing PTX")
        argument_names: list[str] = []
        for arg in kernel_event.get("args", []):
            if arg.get("pointer_hint"):
                pointer_value = arg.get("pointer_value")
                if not pointer_value:
                    raise TraceFunctionalToCudaError(f"Pointer kernel argument is missing pointer_value: {arg}")
                pointer_int = _parse_pointer_string(pointer_value)
                canonical_name = pointer_to_canonical.get(pointer_int)
                if canonical_name is None:
                    raise TraceFunctionalToCudaError(
                        f"Kernel argument pointer {pointer_value} did not match any traced tensor"
                    )
                argument_names.append(_pipeline_tensor_name(canonical_name, return_canonical_name, dest_name))
                continue
            scalar_value = _decode_scalar_argument(arg)
            argument_names.append(constant_registry.name_for_value(scalar_value))

        grid = kernel_event["grid"]
        block = kernel_event["block"]
        kernels.append(
            {
                "source": "ptx",
                "ptx": ptx,
                "kernel": kernel_event.get("ptx_entry") or kernel_event.get("kernel_name"),
                "defines": {},
                "arguments": argument_names,
                "grid_dim": [int(grid["x"]), int(grid["y"]), int(grid["z"])],
                "block_dim": [int(block["x"]), int(block["y"]), int(block["z"])],
            }
        )

    pipeline = {
        "scratch_tensors": scratch_tensors,
        "kernels": kernels,
    }
    output_payload.setdefault("pipelines", []).append(pipeline)
    return output_payload


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def trace_functional_model_to_json(
    model_file: str | Path,
    input_json: str | Path,
    output_json: str | Path,
    *,
    device: str = "cuda:0",
    function_name: str = "functional_model",
    check_determinism: bool = False,
) -> dict[str, Any]:
    model_path = Path(model_file).resolve()
    input_path = Path(input_json).resolve()
    output_path = Path(output_json).resolve()

    if not model_path.exists():
        raise TraceFunctionalToCudaError(f"Model file does not exist: {model_path}")
    if not input_path.exists():
        raise TraceFunctionalToCudaError(f"Input JSON does not exist: {input_path}")

    _ensure_runtime_prerequisites()
    preload_library = build_preload_library()
    input_payload = json.loads(input_path.read_text())

    def run_once() -> dict[str, Any]:
        trace_bundle = _run_child_trace(model_path, function_name, device, preload_library)
        return build_output_json(trace_bundle, input_payload)

    output_payload = run_once()
    if check_determinism:
        second_payload = run_once()
        first_canonical = _canonical_json(output_payload)
        second_canonical = _canonical_json(second_payload)
        if first_canonical != second_canonical:
            diff = "\n".join(
                difflib.unified_diff(
                    first_canonical.splitlines(),
                    second_canonical.splitlines(),
                    fromfile="trace_run_1",
                    tofile="trace_run_2",
                    lineterm="",
                )
            )
            raise TraceFunctionalToCudaError(
                "Determinism check failed: the traced output JSON changed between runs.\n" + diff
            )

    output_path.write_text(json.dumps(output_payload, indent=2) + "\n")
    return output_payload


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=argparse.SUPPRESS)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--function-name", default="functional_model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle-output", required=True)
    return parser


def _public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace a corpus-style functional PyTorch model into a CUDA pipeline JSON entry")
    parser.add_argument("model_file", help="Path to the Python module containing functional_model and get_functional_inputs")
    parser.add_argument("input_json", help="Path to the input pipeline JSON")
    parser.add_argument("output_json", help="Path to write the augmented output JSON")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use for tracing")
    parser.add_argument("--function-name", default="functional_model", help="Function to invoke inside the model module")
    parser.add_argument("--check-determinism", action="store_true", help="Run the full trace twice and require identical canonical JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] == "__child_trace__":
        args = _child_parser().parse_args(argv[1:])
        try:
            return _child_trace(
                Path(args.model_file),
                args.function_name,
                args.device,
                Path(args.bundle_output),
            )
        except TraceFunctionalToCudaError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    args = _public_parser().parse_args(argv)
    try:
        trace_functional_model_to_json(
            args.model_file,
            args.input_json,
            args.output_json,
            device=args.device,
            function_name=args.function_name,
            check_determinism=args.check_determinism,
        )
    except TraceFunctionalToCudaError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
