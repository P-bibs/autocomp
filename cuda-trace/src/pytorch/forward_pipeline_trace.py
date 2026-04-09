#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class StatementSpec:
    stmt_id: int
    kind: str
    target_name: str | None
    callee_expr: str
    arg_exprs: list[str]
    kwarg_exprs: dict[str, str]
    lineno: int


@dataclass
class CallRecord:
    stmt_id: int
    kind: str
    target_name: str | None
    category: str
    op_name: str
    callee_expr: str
    callee: Any
    args: list[Any]
    kwargs: dict[str, Any]
    result: Any
    start_ns: int
    end_ns: int


class ForwardTraceError(RuntimeError):
    pass


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    sanitized = sanitized.strip("_")
    return sanitized or "tensor"


def is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def tensor_pointer(tensor: torch.Tensor) -> int:
    return int(tensor.data_ptr())


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "size": int(tensor.numel()),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


class ForwardCallInstrumenter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.statement_specs: list[StatementSpec] = []
        self._in_model = False
        self._in_forward = False
        self._next_stmt_id = 1
        self._current_names: dict[str, str] = {}
        self._name_versions: dict[str, int] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        previous = self._in_model
        if node.name == "ModelNew":
            self._in_model = True
            node = self.generic_visit(node)
            self._in_model = previous
            return node
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        previous = self._in_forward
        if self._in_model and node.name == "forward":
            self._in_forward = True
            self._current_names = {arg.arg: arg.arg for arg in node.args.args[1:]}
            self._name_versions = {arg.arg: 0 for arg in node.args.args[1:]}
            node.body = [self._instrument_stmt(self._rewrite_stmt_for_ssa(stmt)) for stmt in node.body]
            self._in_forward = previous
            return node
        return self.generic_visit(node)

    def _rewrite_stmt_for_ssa(self, stmt: ast.stmt) -> ast.stmt:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ForwardTraceError("Only simple name assignments are supported in ModelNew.forward")
            stmt.value = self._rewrite_load_names(stmt.value)
            target_name = stmt.targets[0].id
            fresh_name = self._fresh_name(target_name)
            stmt.targets[0].id = fresh_name
            self._current_names[target_name] = fresh_name
            return stmt

        return self._rewrite_load_names(stmt)

    def _rewrite_load_names(self, node: ast.AST) -> ast.AST:
        current_names = self._current_names

        class LoadNameRewriter(ast.NodeTransformer):
            def visit_Name(self, name_node: ast.Name) -> Any:
                if isinstance(name_node.ctx, ast.Load) and name_node.id in current_names:
                    name_node.id = current_names[name_node.id]
                return name_node

        return LoadNameRewriter().visit(node)

    def _fresh_name(self, base_name: str) -> str:
        if base_name not in self._current_names:
            self._name_versions[base_name] = 0
            return base_name

        next_version = self._name_versions.get(base_name, 0) + 1
        self._name_versions[base_name] = next_version
        return f"{base_name}_{next_version}"

    def _instrument_stmt(self, stmt: ast.stmt) -> ast.stmt:
        if not self._in_forward:
            return stmt

        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            spec = self._make_spec("assign_call", stmt.targets[0].id, stmt.value, stmt.lineno)
            stmt.value = self._make_trace_call(spec.stmt_id, stmt.value)
            return ast.copy_location(stmt, stmt)

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            spec = self._make_spec("expr_call", None, stmt.value, stmt.lineno)
            stmt.value = self._make_trace_call(spec.stmt_id, stmt.value)
            return ast.copy_location(stmt, stmt)

        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            spec = self._make_spec("return_call", "return_value", stmt.value, stmt.lineno)
            stmt.value = self._make_trace_call(spec.stmt_id, stmt.value)
            return ast.copy_location(stmt, stmt)

        return stmt

    def _make_spec(self, kind: str, target_name: str | None, call: ast.Call, lineno: int) -> StatementSpec:
        stmt_id = self._next_stmt_id
        self._next_stmt_id += 1
        spec = StatementSpec(
            stmt_id=stmt_id,
            kind=kind,
            target_name=target_name,
            callee_expr=ast.unparse(call.func),
            arg_exprs=[ast.unparse(arg) for arg in call.args],
            kwarg_exprs={
                keyword.arg: ast.unparse(keyword.value)
                for keyword in call.keywords
                if keyword.arg is not None
            },
            lineno=lineno,
        )
        self.statement_specs.append(spec)
        return spec

    def _make_trace_call(self, stmt_id: int, call: ast.Call) -> ast.Call:
        args_list = ast.List(elts=call.args, ctx=ast.Load())
        kwargs_keys = [ast.Constant(value=keyword.arg) for keyword in call.keywords if keyword.arg is not None]
        kwargs_values = [keyword.value for keyword in call.keywords if keyword.arg is not None]
        kwargs_dict = ast.Dict(keys=kwargs_keys, values=kwargs_values)
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="__trace_runtime__", ctx=ast.Load()), attr="trace_call", ctx=ast.Load()),
            args=[ast.Constant(value=stmt_id), call.func, args_list, kwargs_dict],
            keywords=[],
        )


class ForwardTraceRuntime:
    def __init__(self, statement_specs: list[StatementSpec]) -> None:
        self.statement_specs = {spec.stmt_id: spec for spec in statement_specs}
        self.call_records: list[CallRecord] = []
        self.alias_by_ptr: dict[int, str] = {}
        self.tensor_meta_by_name: dict[str, dict[str, Any]] = {}
        self.tensor_role_by_name: dict[str, str] = {}
        self.source_names: set[str] = set()
        self.dest_names: set[str] = set()
        self.synthetic_return_count = 0

    def register_source_inputs(self, arg_names: list[str], args: Iterable[Any]) -> None:
        for name, value in zip(arg_names, args):
            if is_tensor(value):
                self._register_tensor_alias(name, value, role="source")

    def register_model_parameters(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self._register_tensor_alias(sanitize_name(name), param, role="source")
        for name, buf in model.named_buffers():
            self._register_tensor_alias(sanitize_name(name), buf, role="source")

    def mark_outputs(self, outputs: Any) -> None:
        tensors = flatten_tensors(outputs)
        for index, tensor in enumerate(tensors):
            ptr = tensor_pointer(tensor)
            alias = self.alias_by_ptr.get(ptr)
            if alias is None:
                alias = "return_value" if index == 0 else f"return_value_{index}"
                self._register_tensor_alias(alias, tensor, role="dest")
            self.dest_names.add(alias)
            self.tensor_role_by_name[alias] = "dest"
            self.tensor_meta_by_name[alias] = tensor_metadata(tensor)

    def trace_call(self, stmt_id: int, callee: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
        spec = self.statement_specs[stmt_id]
        start_ns = time.monotonic_ns()
        result = callee(*args, **kwargs)
        end_ns = time.monotonic_ns()

        category, op_name = self._classify_call(spec, callee)
        if spec.kind == "assign_call" and spec.target_name is not None:
            if is_tensor(result):
                self._register_tensor_alias(spec.target_name, result, role="intermediate")
        elif spec.kind == "return_call" and is_tensor(result):
            self._register_tensor_alias(spec.target_name or self._synthetic_return_name(), result, role="dest_candidate")

        self.call_records.append(
            CallRecord(
                stmt_id=stmt_id,
                kind=spec.kind,
                target_name=spec.target_name,
                category=category,
                op_name=op_name,
                callee_expr=spec.callee_expr,
                callee=callee,
                args=list(args),
                kwargs=dict(kwargs),
                result=result,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )
        return result

    def build_output(self, kernel_events: list[dict[str, Any]]) -> dict[str, Any]:
        used_source_names: set[str] = set()
        pipeline: list[dict[str, Any]] = []
        remaining_kernels = sorted(
            [event for event in kernel_events if event.get("event") == "kernel_launch"],
            key=lambda event: (event.get("monotonic_ns", 0), event.get("sequence", 0)),
        )

        for record in self.call_records:
            if record.category == "native":
                arguments = self._build_native_arguments(record)
                outputs = self._build_native_outputs(record)
                pipeline.append(
                    {
                        "file": "PyTorch",
                        "kernel": record.op_name,
                        "defines": {},
                        "arguments": arguments,
                        "outputs": outputs,
                    }
                )
                used_source_names.update(name for name in arguments if isinstance(name, str) and name in self.source_names)
                continue

            matched: list[dict[str, Any]] = []
            while remaining_kernels and remaining_kernels[0].get("monotonic_ns", 0) < record.start_ns:
                remaining_kernels.pop(0)
            while remaining_kernels and remaining_kernels[0].get("monotonic_ns", 0) <= record.end_ns:
                matched.append(remaining_kernels.pop(0))

            if not matched:
                raise ForwardTraceError(
                    f"No CUDA kernel launches matched custom call {record.callee_expr} on line {self.statement_specs[record.stmt_id].lineno}"
                )

            for kernel in matched:
                arguments = self._build_kernel_arguments(kernel)
                outputs = self._build_kernel_outputs(record, kernel, len(matched))
                pipeline.append(
                    {
                        "file": kernel.get("module_path", ""),
                        "kernel": kernel.get("kernel_name", ""),
                        "defines": {},
                        "arguments": arguments,
                        "outputs": outputs,
                        "grid_dim": [
                            kernel.get("grid", {}).get("x", 0),
                            kernel.get("grid", {}).get("y", 0),
                            kernel.get("grid", {}).get("z", 0),
                        ],
                        "block_dim": [
                            kernel.get("block", {}).get("x", 0),
                            kernel.get("block", {}).get("y", 0),
                            kernel.get("block", {}).get("z", 0),
                        ],
                        "ptx": kernel.get("ptx", ""),
                    }
                )
                used_source_names.update(name for name in arguments if isinstance(name, str) and name in self.source_names)

        source_tensors = {
            name: {"size": self.tensor_meta_by_name[name]["size"]}
            for name in sorted(used_source_names)
        }
        scratch_tensors = {
            name: {"size": self.tensor_meta_by_name[name]["size"]}
            for name in sorted(self._scratch_tensor_names())
        }
        dest_tensors = {
            name: {"size": self.tensor_meta_by_name[name]["size"]}
            for name in sorted(self.dest_names)
        }
        return {
            "source_tensors": source_tensors,
            "dest_tensors": dest_tensors,
            "pipeline": [
                {
                    "scratch_tensors": scratch_tensors,
                    "kernels": pipeline,
                }
            ],
        }

    def _synthetic_return_name(self) -> str:
        self.synthetic_return_count += 1
        if self.synthetic_return_count == 1:
            return "return_value"
        return f"return_value_{self.synthetic_return_count - 1}"

    def _scratch_tensor_names(self) -> list[str]:
        scratch_names: set[str] = set()
        for name, role in self.tensor_role_by_name.items():
            if role != "intermediate":
                continue
            if name in self.source_names or name in self.dest_names:
                continue
            scratch_names.add(name)
        return sorted(scratch_names)

    def _register_tensor_alias(self, name: str, tensor: torch.Tensor, role: str) -> None:
        sanitized = sanitize_name(name)
        ptr = tensor_pointer(tensor)
        self.alias_by_ptr[ptr] = sanitized
        self.tensor_meta_by_name[sanitized] = tensor_metadata(tensor)
        self.tensor_role_by_name[sanitized] = role
        if role == "source":
            self.source_names.add(sanitized)

    def _classify_call(self, spec: StatementSpec, callee: Any) -> tuple[str, str]:
        if isinstance(callee, nn.Linear):
            return "native", "linear"
        if isinstance(callee, nn.Module):
            return "native", sanitize_name(callee.__class__.__name__.lower())

        callee_module = getattr(callee, "__module__", "") or ""
        callee_name = getattr(callee, "__name__", "") or spec.callee_expr.rsplit(".", 1)[-1]
        normalized = sanitize_name(callee_name)

        if spec.callee_expr.endswith(".to") or spec.callee_expr.endswith(".cuda") or callee_name in {"to", "cuda"}:
            return "native", "copy"
        if spec.callee_expr.startswith("torch."):
            return "native", normalized
        if callee_module.startswith("torch"):
            if "relu" in spec.callee_expr or callee_name == "relu":
                return "native", "relu"
            if "zeros_like" in spec.callee_expr or callee_name == "zeros_like":
                return "native", "zeros_like"
            return "native", normalized
        return "custom", normalized

    def _build_native_arguments(self, record: CallRecord) -> list[Any]:
        if isinstance(record.callee, nn.Linear):
            args: list[Any] = []
            if record.args:
                args.append(self._convert_value(record.args[0]))
            args.append(self._convert_value(record.callee.weight))
            if record.callee.bias is not None:
                args.append(self._convert_value(record.callee.bias))
            return args

        if record.op_name == "copy":
            args = [self._copy_source_symbol(record)]
            args.extend(self._convert_value(arg) for arg in record.args)
            for key in sorted(record.kwargs):
                args.append(self._convert_value(record.kwargs[key]))
            return args

        args = [self._convert_value(arg) for arg in record.args]
        for key in sorted(record.kwargs):
            args.append(self._convert_value(record.kwargs[key]))
        return args

    def _build_native_outputs(self, record: CallRecord) -> list[Any]:
        return self._aliases_for_value(record.result)

    def _build_kernel_arguments(self, kernel_event: dict[str, Any]) -> list[Any]:
        arguments: list[Any] = []
        for arg in kernel_event.get("args", []):
            pointer_value = arg.get("pointer_value")
            if pointer_value:
                ptr = int(pointer_value, 16)
                alias = self.alias_by_ptr.get(ptr)
                arguments.append(alias if alias is not None else pointer_value)
                continue
            if "f32" in arg:
                arguments.append(arg["f32"])
                continue
            if "f64" in arg:
                arguments.append(arg["f64"])
                continue
            if "s64" in arg:
                arguments.append(arg["s64"])
                continue
            if "s32" in arg:
                arguments.append(arg["s32"])
                continue
            if "u64" in arg:
                arguments.append(arg["u64"])
                continue
            if "u32" in arg:
                arguments.append(arg["u32"])

        return arguments

    def _build_kernel_outputs(self, record: CallRecord, kernel_event: dict[str, Any], matched_kernel_count: int) -> list[Any]:
        pointer_outputs = self._kernel_pointer_outputs(kernel_event)
        if pointer_outputs:
            return pointer_outputs

        if matched_kernel_count == 1:
            result_aliases = self._aliases_for_value(record.result)
            if result_aliases:
                return result_aliases

        return []

    def _kernel_pointer_outputs(self, kernel_event: dict[str, Any]) -> list[Any]:
        outputs: list[Any] = []
        for arg in kernel_event.get("args", []):
            pointer_value = arg.get("pointer_value")
            source_type = arg.get("source_type", "")
            if not pointer_value or "*" not in source_type or "const" in source_type:
                continue
            ptr = int(pointer_value, 16)
            alias = self.alias_by_ptr.get(ptr)
            outputs.append(alias if alias is not None else pointer_value)
        return outputs

    def _aliases_for_value(self, value: Any) -> list[Any]:
        aliases: list[Any] = []
        for tensor in flatten_tensors(value):
            alias = self.alias_by_ptr.get(tensor_pointer(tensor))
            if alias is None:
                alias = f"tensor_{tensor_pointer(tensor):x}"
                self._register_tensor_alias(alias, tensor, role="intermediate")
            aliases.append(alias)
        return aliases

    def _convert_value(self, value: Any) -> Any:
        if is_tensor(value):
            alias = self.alias_by_ptr.get(tensor_pointer(value))
            if alias is None:
                alias = f"tensor_{tensor_pointer(value):x}"
                self._register_tensor_alias(alias, value, role="intermediate")
            return alias
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return str(value)

    def _copy_source_symbol(self, record: CallRecord) -> Any:
        source_expr = record.callee_expr.rsplit(".", 1)[0]
        if source_expr in self.tensor_meta_by_name:
            return source_expr
        return source_expr


def find_forward_arg_names(tree: ast.AST) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return [arg.arg for arg in item.args.args[1:]]
    raise ForwardTraceError("Could not find ModelNew.forward")


def load_statement_specs(source: str, source_path: Path) -> tuple[ast.AST, list[StatementSpec], list[str]]:
    tree = ast.parse(source, filename=str(source_path))
    arg_names = find_forward_arg_names(tree)
    instrumenter = ForwardCallInstrumenter()
    new_tree = instrumenter.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, instrumenter.statement_specs, arg_names


def execute_instrumented_module(source_path: Path, runtime: ForwardTraceRuntime, tree: ast.AST) -> dict[str, Any]:
    module_globals: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(source_path),
        "__name__": f"_forward_trace_{source_path.stem}",
        "__package__": None,
        "__trace_runtime__": runtime,
    }
    code = compile(tree, filename=str(source_path), mode="exec")
    exec(code, module_globals)
    return module_globals


def execute_plain_module(source_path: Path) -> dict[str, Any]:
    module_globals: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(source_path),
        "__name__": f"_forward_trace_aux_{source_path.stem}",
        "__package__": None,
    }
    code = compile(source_path.read_text(), filename=str(source_path), mode="exec")
    exec(code, module_globals)
    return module_globals


def read_kernel_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def ensure_kernel_tracer(args: argparse.Namespace, kernel_log_path: Path) -> None:
    if os.environ.get("FORWARD_TRACE_TRACER_ACTIVE") == "1":
        return

    root = Path(__file__).resolve().parents[2]
    tracer_path = Path(os.environ.get("FORWARD_TRACE_TRACER_PATH", root / "build" / "pytorch_ext" / "libcuda_trace_module_preload.so"))
    if not tracer_path.exists():
        raise ForwardTraceError(f"Missing CUDA tracer at {tracer_path}; build it first")

    env = os.environ.copy()
    env["FORWARD_TRACE_TRACER_ACTIVE"] = "1"
    env["FORWARD_TRACE_TRACER_PATH"] = str(tracer_path)
    env["CUDA_TRACE_JSON_PATH"] = str(kernel_log_path)
    env.setdefault("TORCH_EXTENSIONS_DIR", str(root / "build" / "pytorch_ext" / "jit"))
    env.setdefault("TORCH_CUDA_ARCH_LIST", "7.5+PTX")
    env.setdefault("CUDA_TRACE_FILTER_PATH_SUBSTR", env["TORCH_EXTENSIONS_DIR"])
    ld_preload = env.get("LD_PRELOAD", "")
    if ld_preload:
        env["LD_PRELOAD"] = f"{tracer_path}:{ld_preload}"
    else:
        env["LD_PRELOAD"] = str(tracer_path)

    argv = [sys.executable, __file__, args.input_file, args.output_json]
    if args.io_file is not None:
        argv.extend(["--io-file", args.io_file])
    os.execve(sys.executable, argv, env)


def run_trace(input_file: Path, output_json: Path, kernel_log_path: Path, io_file: Path | None) -> None:
    source = input_file.read_text()
    tree, statement_specs, arg_names = load_statement_specs(source, input_file)
    runtime = ForwardTraceRuntime(statement_specs)
    module_globals = execute_instrumented_module(input_file, runtime, tree)
    io_globals = module_globals if io_file is None else execute_plain_module(io_file)

    try:
        get_init_inputs = io_globals["get_init_inputs"]
        get_inputs = io_globals["get_inputs"]
        model_cls = module_globals["ModelNew"]
    except KeyError as exc:
        raise ForwardTraceError(f"Missing required symbol: {exc.args[0]}") from exc

    model = model_cls(*get_init_inputs())
    if torch.cuda.is_available():
        model = model.cuda()

    inputs = get_inputs()
    if torch.cuda.is_available():
        inputs = [value.cuda() if is_tensor(value) and not value.is_cuda else value for value in inputs]
    runtime.register_source_inputs(arg_names, inputs)
    runtime.register_model_parameters(model)

    outputs = model(*inputs)
    runtime.mark_outputs(outputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    final_json = runtime.build_output(read_kernel_events(kernel_log_path))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(final_json, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace a pattern-based PyTorch forward pass into pipeline JSON")
    parser.add_argument("input_file", help="Python file containing ModelNew")
    parser.add_argument("output_json", help="Output JSON path")
    parser.add_argument("--io-file", help="Optional Python file containing get_init_inputs and get_inputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_file = Path(args.input_file).resolve()
    output_json = Path(args.output_json).resolve()
    io_file = Path(args.io_file).resolve() if args.io_file is not None else None
    output_json.parent.mkdir(parents=True, exist_ok=True)

    existing_kernel_log = os.environ.get("CUDA_TRACE_JSON_PATH") if os.environ.get("FORWARD_TRACE_TRACER_ACTIVE") == "1" else None
    if existing_kernel_log:
        run_trace(input_file, output_json, Path(existing_kernel_log), io_file)
        return 0

    with tempfile.TemporaryDirectory(prefix="forward_trace_") as temp_dir:
        kernel_log_path = Path(temp_dir) / "kernel_events.jsonl"
        ensure_kernel_tracer(args, kernel_log_path)
        run_trace(input_file, output_json, kernel_log_path, io_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
