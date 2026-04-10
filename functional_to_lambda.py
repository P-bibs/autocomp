#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
import threading
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence
import uuid

import torch
from tqdm import tqdm

_TORCH_EXTENSION_BUILD_RUN_ID = uuid.uuid4().hex


def _install_isolated_torch_extension_builds() -> None:
    try:
        import torch.utils.cpp_extension as cpp_extension
    except Exception:
        return
    if getattr(cpp_extension.load_inline, "_functional_to_lambda_isolated", False):
        return

    original_load_inline = cpp_extension.load_inline

    def isolated_load_inline(name: str, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("build_directory") is None:
            # Keep each worker on its own JIT extension path so unrelated imports do
            # not block each other on a shared lock file in ~/.cache/torch_extensions.
            build_directory = (
                Path.home()
                / ".cache"
                / "functional_to_lambda_torch_extensions"
                / _TORCH_EXTENSION_BUILD_RUN_ID
                / f"pid-{os.getpid()}"
                / f"thread-{threading.get_ident()}"
                / name
            )
            build_directory.mkdir(parents=True, exist_ok=True)
            kwargs = dict(kwargs)
            kwargs["build_directory"] = str(build_directory)
        return original_load_inline(name, *args, **kwargs)

    isolated_load_inline._functional_to_lambda_isolated = True  # type: ignore[attr-defined]
    cpp_extension.load_inline = isolated_load_inline


_install_isolated_torch_extension_builds()


class FunctionalToLambdaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModuleSpec:
    path: Path
    module: ModuleType
    tree: ast.Module
    functional_model: ast.FunctionDef
    forward_arg_names: list[str]
    state_names: list[str]
    input_shapes: dict[str, tuple[int, ...]]
    state_meta: dict[str, Any]
    output_shape: tuple[int, ...]
    first_tensor_arg: str


@dataclass(frozen=True)
class ValueRef:
    shape: tuple[int, ...]
    renderer: Callable[[Sequence[str]], str]

    def render(self, coord: Sequence[str]) -> str:
        if len(coord) != len(self.shape):
            raise FunctionalToLambdaError(f"Coordinate rank mismatch: expected {len(self.shape)}, got {len(coord)}")
        return self.renderer(coord)


def tuple_expr(items: Sequence[str]) -> str:
    if not items:
        return "()"
    if len(items) == 1:
        return f"({items[0]},)"
    return "(" + ", ".join(items) + ")"


def index_expr(name: str, coord: Sequence[str]) -> str:
    if not coord:
        return name
    if len(coord) == 1:
        return f"{name}[{coord[0]}]"
    return f"{name}[{', '.join(coord)}]"


def scalar_value(expr: str) -> ValueRef:
    return ValueRef(shape=(), renderer=lambda coord: expr)


def tensor_arg_value(name: str, shape: tuple[int, ...], backend: str) -> ValueRef:
    if backend == "smt":
        return ValueRef(shape=shape, renderer=lambda coord, name=name: f"_smt_select('{name}', {tuple_expr(coord)})")
    if backend == "cpp":
        if not shape:
            return ValueRef(shape=shape, renderer=lambda coord, name=name: f"_smt_symbol({cpp_string_literal(name)})")
        return ValueRef(
            shape=shape,
            renderer=lambda coord, name=name: f"_smt_select({cpp_string_literal(name)}{''.join(f', {item}' for item in coord)})",
        )
    return ValueRef(shape=shape, renderer=lambda coord: f"_scalar({index_expr(name, coord)})")


def lambda_value(name: str, shape: tuple[int, ...]) -> ValueRef:
    return ValueRef(shape=shape, renderer=lambda coord: f"{name}({tuple_expr(coord)})")


def full_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = full_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def normalize_tuple(value: Any, ndim: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, tuple):
        if len(value) == ndim:
            return tuple(int(v) for v in value)
        if len(value) == 1:
            return tuple(int(value[0]) for _ in range(ndim))
    raise FunctionalToLambdaError(f"Expected int/tuple with rank {ndim}, got {value!r}")


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def py_literal(value: Any) -> str:
    if isinstance(value, tuple):
        return repr(tuple(value))
    return repr(value)


def cpp_string_literal(value: str) -> str:
    return json.dumps(value)


def tensor_size(shape: Sequence[int]) -> int:
    return product(shape)


def build_cpp_json_spec(spec: ModuleSpec, cpp_output_path: str | Path) -> dict[str, Any]:
    cpp_path = Path(cpp_output_path)
    source_tensors: dict[str, dict[str, int]] = {}

    for name in spec.forward_arg_names:
        shape = spec.input_shapes.get(name)
        if shape is not None:
            source_tensors[name] = {"size": tensor_size(shape)}

    for name in spec.state_names:
        value = spec.state_meta[name]
        if isinstance(value, torch.Tensor):
            source_tensors[name] = {"size": tensor_size(tuple(value.shape))}

    return {
        "source_tensors": source_tensors,
        "dest_tensors": {
            "out": {"size": tensor_size(spec.output_shape)},
        },
        "constants": {},
        "pipelines": [
            {
                "scratch_tensors": {},
                "kernels": [
                    {
                        "source": "spec",
                        "file": cpp_path.name,
                    }
                ],
            }
        ],
        "expected_result": "equivalent",
    }


def write_cpp_outputs(spec: ModuleSpec, cpp_source: str, cpp_output_path: str | Path) -> None:
    cpp_path = Path(cpp_output_path)
    cpp_path.write_text(cpp_source)
    json_path = cpp_path.with_suffix(".json")
    json_spec = build_cpp_json_spec(spec, cpp_path)
    json_path.write_text(json.dumps(json_spec, indent=2) + "\n")


def broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    out: list[int] = []
    for a, b in zip(reversed(lhs), reversed(rhs)):
        if a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        elif a == b:
            out.append(a)
        else:
            raise FunctionalToLambdaError(f"Incompatible broadcast shapes: {lhs} and {rhs}")
    longer = lhs if len(lhs) > len(rhs) else rhs
    out.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
    return tuple(reversed(out))


def broadcast_coord(out_coord: Sequence[str], out_shape: tuple[int, ...], in_shape: tuple[int, ...]) -> list[str]:
    if not in_shape:
        return []
    offset = len(out_shape) - len(in_shape)
    mapped: list[str] = []
    for idx, dim in enumerate(in_shape):
        out_idx = idx + offset
        if out_idx < 0:
            raise FunctionalToLambdaError(f"Cannot broadcast {in_shape} into {out_shape}")
        mapped.append("0" if dim == 1 else out_coord[out_idx])
    return mapped


def reduction_shape(shape: tuple[int, ...], dims: Sequence[int], keepdim: bool) -> tuple[int, ...]:
    normalized = {dim if dim >= 0 else dim + len(shape) for dim in dims}
    if keepdim:
        return tuple(1 if idx in normalized else size for idx, size in enumerate(shape))
    return tuple(size for idx, size in enumerate(shape) if idx not in normalized)


def conv_output_size(in_size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return ((in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def conv_transpose_output_size(in_size: int, kernel: int, stride: int, padding: int, dilation: int, output_padding: int) -> int:
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def pool_output_size(in_size: int, kernel: int, stride: int, padding: int, dilation: int, ceil_mode: bool) -> int:
    numerator = in_size + 2 * padding - dilation * (kernel - 1) - 1
    if ceil_mode:
        return (numerator + stride - 1) // stride + 1
    return numerator // stride + 1


def affine_expr(terms: Sequence[tuple[str, int]], constant: int = 0) -> str:
    pieces: list[tuple[str, str]] = []
    for term, coeff in terms:
        if coeff == 0:
            continue
        mag = abs(int(coeff))
        rendered = term if mag == 1 else f"({term} * {mag})"
        pieces.append(("+" if coeff > 0 else "-", rendered))
    if constant:
        pieces.append(("+" if constant > 0 else "-", str(abs(int(constant)))))
    if not pieces:
        return "0"
    first_sign, first_term = pieces[0]
    expr = first_term if first_sign == "+" else f"(-{first_term})"
    for sign, term in pieces[1:]:
        expr = f"({expr} {sign} {term})"
    return expr


def render_nested_sum(bounds: Sequence[int], body_builder: Callable[[list[str]], str], prefix: str) -> str:
    vars_ = [f"{prefix}{idx}" for idx in range(len(bounds))]
    expr = body_builder(vars_)
    for var, bound in reversed(list(zip(vars_, bounds))):
        expr = f"sum(map(lambda {var}: {expr}, range({int(bound)})), zero)"
    return expr


def render_nested_stack_reduce(kind: str, bounds: Sequence[int], body_builder: Callable[[list[str]], str], prefix: str) -> str:
    vars_ = [f"{prefix}{idx}" for idx in range(len(bounds))]
    expr = body_builder(vars_)
    for var, bound in reversed(list(zip(vars_, bounds))):
        expr = f"{kind}(tuple(map(lambda {var}: {expr}, range({int(bound)}))))"
    return expr


def render_nested_smt_sum(bounds: Sequence[int], body_builder: Callable[[list[str]], str], prefix: str) -> str:
    vars_ = [f"{prefix}{idx}" for idx in range(len(bounds))]
    expr = body_builder(vars_)
    for var, bound in reversed(list(zip(vars_, bounds))):
        expr = f"_smt_add(tuple(map(lambda {var}: {expr}, range({int(bound)}))))"
    return expr


def render_nested_smt_extreme(kind: str, bounds: Sequence[int], body_builder: Callable[[list[str]], str], prefix: str) -> str:
    vars_ = [f"{prefix}{idx}" for idx in range(len(bounds))]
    expr = body_builder(vars_)
    reducer = "_smt_max" if kind == "max" else "_smt_min"
    for var, bound in reversed(list(zip(vars_, bounds))):
        expr = f"{reducer}(tuple(map(lambda {var}: {expr}, range({int(bound)}))))"
    return expr


def render_nested_cpp_sum(bounds: Sequence[int], body_builder: Callable[[list[str]], str], prefix: str) -> str:
    if not bounds:
        return body_builder([])
    if len(bounds) == 1:
        var = f"{prefix}0"
        return f"_smt_add_range({int(bounds[0])}, [&](long long {var}) -> SmtString {{ return {body_builder([var])}; }})"
    vars_ = [f"idx[{idx}]" for idx in range(len(bounds))]
    return (
        f"_smt_add_nd<{len(bounds)}>({{{', '.join(str(int(bound)) for bound in bounds)}}}, "
        f"[&](const auto& idx) -> SmtString {{ return {body_builder(vars_)}; }})"
    )


def render_nested_cpp_extreme(kind: str, bounds: Sequence[int], body_builder: Callable[[list[str]], str], prefix: str) -> str:
    if not bounds:
        return body_builder([])
    if len(bounds) == 1:
        var = f"{prefix}0"
        reducer = "_smt_max_range" if kind == "max" else "_smt_min_range"
        return f"{reducer}({int(bounds[0])}, [&](long long {var}) -> SmtString {{ return {body_builder([var])}; }})"
    vars_ = [f"idx[{idx}]" for idx in range(len(bounds))]
    reducer = "_smt_max_nd" if kind == "max" else "_smt_min_nd"
    return (
        f"{reducer}<{len(bounds)}>({{{', '.join(str(int(bound)) for bound in bounds)}}}, "
        f"[&](const auto& idx) -> SmtString {{ return {body_builder(vars_)}; }})"
    )


def render_reduction_input_coord(
    out_coord: Sequence[str],
    input_rank: int,
    dims: Sequence[int],
    reduce_vars: Sequence[str],
    keepdim: bool,
) -> list[str]:
    normalized = [dim if dim >= 0 else dim + input_rank for dim in dims]
    reduced = set(normalized)
    reduced_iter = iter(reduce_vars)
    out_iter = iter(out_coord)
    coord: list[str] = []
    for axis in range(input_rank):
        if axis in reduced:
            coord.append(next(reduced_iter))
            if keepdim:
                next(out_iter)
        else:
            coord.append(next(out_iter))
    return coord


def load_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"functional_to_lambda_{uuid.uuid4().hex}", path)
    if spec is None or spec.loader is None:
        raise FunctionalToLambdaError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise FunctionalToLambdaError(f"Missing function {name}")


def validate_function_shape(function_node: ast.FunctionDef) -> None:
    for stmt in function_node.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise FunctionalToLambdaError("Unsupported statement: only simple name assignments are supported")
            continue
        if isinstance(stmt, ast.Return):
            continue
        raise FunctionalToLambdaError(f"Unsupported statement: {ast.unparse(stmt)}")


def load_module_spec(path: str | Path) -> ModuleSpec:
    target = Path(path)
    source = target.read_text()
    tree = ast.parse(source)
    module = load_module_from_path(target)
    functional_model = find_function(tree, "functional_model")
    validate_function_shape(functional_model)
    if not hasattr(module, "functional_model"):
        raise FunctionalToLambdaError("Module does not define functional_model")
    if not hasattr(module, "FORWARD_ARG_NAMES"):
        raise FunctionalToLambdaError("Module does not define FORWARD_ARG_NAMES")
    if not hasattr(module, "REQUIRED_STATE_NAMES"):
        raise FunctionalToLambdaError("Module does not define REQUIRED_STATE_NAMES")
    with torch.device("meta"):
        model = module.build_reference_model()
        state_kwargs = module.extract_state_kwargs(model)
        forward_args = tuple(module.get_inputs())
        output = module.functional_model(*forward_args, **state_kwargs)
    input_shapes = {
        name: tuple(arg.shape)
        for name, arg in zip(module.FORWARD_ARG_NAMES, forward_args)
        if isinstance(arg, torch.Tensor)
    }
    tensor_arg_names = [name for name, arg in zip(module.FORWARD_ARG_NAMES, forward_args) if isinstance(arg, torch.Tensor)]
    if not tensor_arg_names:
        raise FunctionalToLambdaError("Expected at least one tensor input")
    return ModuleSpec(
        path=target,
        module=module,
        tree=tree,
        functional_model=functional_model,
        forward_arg_names=list(module.FORWARD_ARG_NAMES),
        state_names=list(module.REQUIRED_STATE_NAMES),
        input_shapes=input_shapes,
        state_meta=state_kwargs,
        output_shape=tuple(output.shape),
        first_tensor_arg=tensor_arg_names[0],
    )


class LambdaLowerer:
    def __init__(self, spec: ModuleSpec, function_name: str, backend: str = "python") -> None:
        self.spec = spec
        self.function_name = function_name
        self.backend = backend
        self.env: dict[str, ValueRef] = {}
        self.lines: list[str] = []
        self.tmp_index = 0
        self.static_env = {
            name: getattr(spec.module, name)
            for name in dir(spec.module)
            if not name.startswith("__")
        }
        self.static_env.update(spec.state_meta)
        self._init_env()

    def _init_env(self) -> None:
        args = self.spec.module.get_inputs()
        for name, value in zip(self.spec.forward_arg_names, args):
            if isinstance(value, torch.Tensor):
                self.env[name] = tensor_arg_value(name, tuple(value.shape), self.backend)
            else:
                if self.backend == "cpp":
                    self.env[name] = scalar_value(self.atom(name))
                else:
                    self.env[name] = scalar_value(self.atom(name))
        for name in self.spec.state_names:
            value = self.spec.state_meta[name]
            if isinstance(value, torch.Tensor):
                self.env[name] = tensor_arg_value(name, tuple(value.shape), self.backend)
            elif value is None:
                self.env[name] = scalar_value(self.atom("0.0") if self.backend == "cpp" else self.atom("None"))
            else:
                if self.backend == "cpp":
                    self.env[name] = scalar_value(self.atom(name))
                else:
                    self.env[name] = scalar_value(self.atom(name))

    def new_name(self) -> str:
        self.tmp_index += 1
        return f"v{self.tmp_index}"

    def static_value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.static_env:
                return self.static_env[node.id]
            raise FunctionalToLambdaError(f"Unknown static name: {node.id}")
        if isinstance(node, ast.Tuple):
            return tuple(self.static_value(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [self.static_value(elt) for elt in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.static_value(node.operand)
        raise FunctionalToLambdaError(f"Unsupported static expression: {ast.unparse(node)}")

    def keyword_value(self, call: ast.Call, name: str, default: Any = None) -> Any:
        for keyword in call.keywords:
            if keyword.arg == name:
                return self.static_value(keyword.value)
        return default

    def coordinate(self) -> list[str]:
        return self.coord_vars_for_shape(self.spec.output_shape)

    def coord_vars_for_shape(self, shape: Sequence[int], source: str = "coord") -> list[str]:
        vars_: list[str] = []
        for idx, size in enumerate(shape):
            if self.backend == "cpp" and int(size) == 1:
                vars_.append("0")
            else:
                if self.backend == "cpp" and source != "coord":
                    vars_.append(f"{source}{idx}")
                else:
                    vars_.append(f"{source}[{idx}]")
        return vars_

    def cpp_coord_param_names(self, shape: Sequence[int]) -> list[str]:
        return [f"c{idx}" for idx in range(len(shape))]

    def cpp_coord_param_decl(self, shape: Sequence[int]) -> str:
        return ", ".join(f"long long {name}" for name in self.cpp_coord_param_names(shape))

    def emits_cpp(self) -> bool:
        return self.backend == "cpp"

    def emits_smt_strings(self) -> bool:
        return self.backend in {"smt", "cpp"}

    def atom(self, text: str) -> str:
        if self.backend == "smt":
            return repr(text)
        if self.backend == "cpp":
            return cpp_string_literal(text)
        return text

    def fmt_add(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"_smt_add2({lhs}, {rhs})"
        if self.backend == "cpp":
            return f"_smt_add2({lhs}, {rhs})"
        return f"({lhs} + {rhs})"

    def fmt_sub(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"_smt_sub2({lhs}, {rhs})"
        if self.backend == "cpp":
            return f"_smt_sub2({lhs}, {rhs})"
        return f"({lhs} - {rhs})"

    def fmt_mul(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"_smt_mul2({lhs}, {rhs})"
        if self.backend == "cpp":
            return f"_smt_mul2({lhs}, {rhs})"
        return f"({lhs} * {rhs})"

    def fmt_div(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"_smt_div2({lhs}, {rhs})"
        if self.backend == "cpp":
            return f"_smt_div2({lhs}, {rhs})"
        return f"({lhs} / {rhs})"

    def fmt_int_div(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"(_ div {lhs} {rhs})"
        if self.backend == "cpp":
            return f"(({lhs}) / ({rhs}))"
        return f"({lhs} // {rhs})"

    def fmt_mod(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"(mod {lhs} {rhs})"
        if self.backend == "cpp":
            return f"(({lhs}) % ({rhs}))"
        return f"({lhs} % {rhs})"

    def fmt_eq(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"(= {lhs} {rhs})"
        if self.backend == "cpp":
            return f"(({lhs}) == ({rhs}))"
        return f"({lhs} == {rhs})"

    def fmt_le(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"(<= {lhs} {rhs})"
        if self.backend == "cpp":
            return f"(({lhs}) <= ({rhs}))"
        return f"({lhs} <= {rhs})"

    def fmt_lt(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"(< {lhs} {rhs})"
        if self.backend == "cpp":
            return f"(({lhs}) < ({rhs}))"
        return f"({lhs} < {rhs})"

    def fmt_in_bounds(self, expr: str, upper: int) -> str:
        return self.fmt_and([self.fmt_le("0", expr), self.fmt_lt(expr, str(upper))])

    def fmt_neg(self, expr: str) -> str:
        if self.emits_smt_strings():
            return self.fmt_sub("0.0", expr)
        return f"(-{expr})"

    def fmt_sqrt(self, expr: str) -> str:
        if self.backend == "smt":
            return f"_smt_fun('sqrt', {expr})"
        if self.backend == "cpp":
            return f"_smt_fun(\"sqrt\", {expr})"
        return f"math.sqrt({expr})"

    def fmt_exp(self, expr: str) -> str:
        if self.backend == "smt":
            return f"_smt_fun('exp', {expr})"
        if self.backend == "cpp":
            return f"_smt_fun(\"exp\", {expr})"
        return f"math.exp({expr})"

    def fmt_tanh(self, expr: str) -> str:
        if self.backend == "smt":
            return f"_smt_fun('tanh', {expr})"
        if self.backend == "cpp":
            return f"_smt_fun(\"tanh\", {expr})"
        return f"math.tanh({expr})"

    def fmt_erf(self, expr: str) -> str:
        if self.backend == "smt":
            return f"_smt_fun('erf', {expr})"
        if self.backend == "cpp":
            return f"_smt_fun(\"erf\", {expr})"
        return f"math.erf({expr})"

    def fmt_abs(self, expr: str) -> str:
        if self.backend == "smt":
            return f"_smt_fun('abs', {expr})"
        if self.backend == "cpp":
            return f"_smt_fun(\"abs\", {expr})"
        return f"abs({expr})"

    def fmt_min(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"_smt_min2({lhs}, {rhs})"
        if self.backend == "cpp":
            return f"_smt_min2({lhs}, {rhs})"
        return f"min({lhs}, {rhs})"

    def fmt_max(self, lhs: str, rhs: str) -> str:
        if self.backend == "smt":
            return f"_smt_max2({lhs}, {rhs})"
        if self.backend == "cpp":
            return f"_smt_max2({lhs}, {rhs})"
        return f"max({lhs}, {rhs})"

    def fmt_if(self, cond: str, true_expr: str, false_expr: str) -> str:
        if self.backend == "smt":
            return f"(ite {cond} {true_expr} {false_expr})"
        if self.backend == "cpp":
            return f"(({cond}) ? ({true_expr}) : ({false_expr}))"
        return f"({true_expr} if {cond} else {false_expr})"

    def fmt_and(self, conds: Sequence[str]) -> str:
        if self.backend == "smt":
            return "true" if not conds else (conds[0] if len(conds) == 1 else f"(and {' '.join(conds)})")
        if self.backend == "cpp":
            if not conds:
                return "true"
            if len(conds) == 1:
                return conds[0]
            return f"({' && '.join(conds)})"
        return " and ".join(conds) if conds else "True"

    def coord_expr(self, items: Sequence[str]) -> str:
        if self.backend == "cpp":
            return ", ".join(items)
        return tuple_expr(items)

    def fmt_clamp(self, expr: str, min_expr: str | None, max_expr: str | None) -> str:
        if min_expr is not None:
            expr = self.fmt_max(expr, min_expr)
        if max_expr is not None:
            expr = self.fmt_min(expr, max_expr)
        return expr

    def build(self) -> str:
        if self.backend == "cpp":
            self.lines.append("#include <algorithm>")
            self.lines.append("#include <charconv>")
            self.lines.append("#include <array>")
            self.lines.append("#include <cstdlib>")
            self.lines.append("#include <initializer_list>")
            self.lines.append("#include <iostream>")
            self.lines.append("#include <memory_resource>")
            self.lines.append("#include <string>")
            self.lines.append("#include <vector>")
            self.lines.append("")
            self.lines.append("using Coord = std::array<long long, %d>;" % len(self.spec.output_shape))
            self.lines.append("using SmtString = std::pmr::string;")
            self.lines.append("")
            self.lines.append("static inline SmtString _smt_symbol(const char* name) {")
            self.lines.append("    return SmtString(name);")
            self.lines.append("}")
            self.lines.append("static inline void _smt_append_atom(SmtString& out, const SmtString& value) {")
            self.lines.append("    out += value;")
            self.lines.append("}")
            self.lines.append("static inline void _smt_append_atom(SmtString& out, const char* value) {")
            self.lines.append("    out += value;")
            self.lines.append("}")
            self.lines.append("template <typename T>")
            self.lines.append("static inline void _smt_append_integral(SmtString& out, T value) {")
            self.lines.append("    char buffer[32];")
            self.lines.append("    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);")
            self.lines.append("    if (ec == std::errc()) out.append(buffer, ptr);")
            self.lines.append("}")
            self.lines.append("static inline void _smt_append_atom(SmtString& out, int value) {")
            self.lines.append("    _smt_append_integral(out, value);")
            self.lines.append("}")
            self.lines.append("static inline void _smt_append_atom(SmtString& out, long long value) {")
            self.lines.append("    _smt_append_integral(out, value);")
            self.lines.append("}")
            self.lines.append("static inline void _smt_append_atom(SmtString& out, unsigned long long value) {")
            self.lines.append("    _smt_append_integral(out, value);")
            self.lines.append("}")
            self.lines.append("static inline void _smt_append_atom(SmtString& out, bool value) {")
            self.lines.append("    out += value ? \"true\" : \"false\";")
            self.lines.append("}")
            self.lines.append("template <typename T>")
            self.lines.append("static inline void _smt_append_atom(SmtString& out, T value) {")
            self.lines.append("    out += std::to_string(value);")
            self.lines.append("}")
            self.lines.append("template <typename... Args>")
            self.lines.append("static inline SmtString _smt_select(const char* name, const Args&... idxs) {")
            self.lines.append("    constexpr std::size_t count = sizeof...(Args);")
            self.lines.append("    if constexpr (count == 0) return SmtString(name);")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(std::char_traits<char>::length(name) + count * 16);")
            self.lines.append("    for (std::size_t i = 0; i < count; ++i) result += \"(select \";")
            self.lines.append("    result += name;")
            self.lines.append("    auto append_idx = [&](const auto& idx) {")
            self.lines.append("        result += ' ';")
            self.lines.append("        _smt_append_atom(result, idx);")
            self.lines.append("        result += ')';")
            self.lines.append("    };")
            self.lines.append("    (append_idx(idxs), ...);")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("static inline SmtString _smt_fun(const char* name, const SmtString& arg) {")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(arg.size() + std::char_traits<char>::length(name) + 3);")
            self.lines.append("    result += '(';")
            self.lines.append("    result += name;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += arg;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_add2(const A& a, const B& b) {")
            self.lines.append("    SmtString sa;")
            self.lines.append("    SmtString sb;")
            self.lines.append("    _smt_append_atom(sa, a);")
            self.lines.append("    _smt_append_atom(sb, b);")
            self.lines.append("    if (sa == \"0.0\") return sb;")
            self.lines.append("    if (sb == \"0.0\") return sa;")
            self.lines.append("    SmtString result = \"(+ \";")
            self.lines.append("    result += sa;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += sb;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename B>")
            self.lines.append("static inline SmtString _smt_add2(SmtString a, const B& b) {")
            self.lines.append("    if (a == \"0.0\") {")
            self.lines.append("        SmtString sb;")
            self.lines.append("        _smt_append_atom(sb, b);")
            self.lines.append("        return sb;")
            self.lines.append("    }")
            self.lines.append("    SmtString sb;")
            self.lines.append("    _smt_append_atom(sb, b);")
            self.lines.append("    if (sb == \"0.0\") return a;")
            self.lines.append("    if (a.size() >= 2 && a[0] == '(' && a[1] == '+') {")
            self.lines.append("        a.pop_back();")
            self.lines.append("        a += ' ';")
            self.lines.append("        a += sb;")
            self.lines.append("        a += ')';")
            self.lines.append("        return a;")
            self.lines.append("    }")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(a.size() + sb.size() + 5);")
            self.lines.append("    result += \"(+ \";")
            self.lines.append("    result += a;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += sb;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_sub2(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(- \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_mul2(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(* \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_div2(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(/ \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_max2(const A& a, const B& b) {")
            self.lines.append("    SmtString sa;")
            self.lines.append("    SmtString sb;")
            self.lines.append("    _smt_append_atom(sa, a);")
            self.lines.append("    _smt_append_atom(sb, b);")
            self.lines.append("    if (sa == \"(- 1.0e309)\") return sb;")
            self.lines.append("    if (sb == \"(- 1.0e309)\") return sa;")
            self.lines.append("    SmtString result = \"(max \";")
            self.lines.append("    result += sa;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += sb;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename B>")
            self.lines.append("static inline SmtString _smt_max2(SmtString a, const B& b) {")
            self.lines.append("    if (a == \"(- 1.0e309)\") {")
            self.lines.append("        SmtString sb;")
            self.lines.append("        _smt_append_atom(sb, b);")
            self.lines.append("        return sb;")
            self.lines.append("    }")
            self.lines.append("    SmtString sb;")
            self.lines.append("    _smt_append_atom(sb, b);")
            self.lines.append("    if (sb == \"(- 1.0e309)\") return a;")
            self.lines.append("    if (a.size() >= 4 && a[0] == '(' && a[1] == 'm' && a[2] == 'a' && a[3] == 'x') {")
            self.lines.append("        a.pop_back();")
            self.lines.append("        a += ' ';")
            self.lines.append("        a += sb;")
            self.lines.append("        a += ')';")
            self.lines.append("        return a;")
            self.lines.append("    }")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(a.size() + sb.size() + 8);")
            self.lines.append("    result += \"(max \";")
            self.lines.append("    result += a;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += sb;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_min2(const A& a, const B& b) {")
            self.lines.append("    SmtString sa;")
            self.lines.append("    SmtString sb;")
            self.lines.append("    _smt_append_atom(sa, a);")
            self.lines.append("    _smt_append_atom(sb, b);")
            self.lines.append("    if (sa == \"0.0\") return sb;")
            self.lines.append("    if (sb == \"0.0\") return sa;")
            self.lines.append("    SmtString result = \"(min \";")
            self.lines.append("    result += sa;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += sb;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename B>")
            self.lines.append("static inline SmtString _smt_min2(SmtString a, const B& b) {")
            self.lines.append("    if (a == \"0.0\") {")
            self.lines.append("        SmtString sb;")
            self.lines.append("        _smt_append_atom(sb, b);")
            self.lines.append("        return sb;")
            self.lines.append("    }")
            self.lines.append("    SmtString sb;")
            self.lines.append("    _smt_append_atom(sb, b);")
            self.lines.append("    if (sb == \"0.0\") return a;")
            self.lines.append("    if (a.size() >= 4 && a[0] == '(' && a[1] == 'm' && a[2] == 'i' && a[3] == 'n') {")
            self.lines.append("        a.pop_back();")
            self.lines.append("        a += ' ';")
            self.lines.append("        a += sb;")
            self.lines.append("        a += ')';")
            self.lines.append("        return a;")
            self.lines.append("    }")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(a.size() + sb.size() + 8);")
            self.lines.append("    result += \"(min \";")
            self.lines.append("    result += a;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += sb;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_eq(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(= \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_le(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(<= \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_lt(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(< \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_mod(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(mod \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename A, typename B>")
            self.lines.append("static inline SmtString _smt_int_div(const A& a, const B& b) {")
            self.lines.append("    SmtString result = \"(_ div \";")
            self.lines.append("    _smt_append_atom(result, a);")
            self.lines.append("    result += ' ';")
            self.lines.append("    _smt_append_atom(result, b);")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("static inline SmtString _smt_ite(const SmtString& cond, const SmtString& true_expr, const SmtString& false_expr) {")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(cond.size() + true_expr.size() + false_expr.size() + 8);")
            self.lines.append("    result += \"(ite \";")
            self.lines.append("    result += cond;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += true_expr;")
            self.lines.append("    result += ' ';")
            self.lines.append("    result += false_expr;")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename... Args>")
            self.lines.append("static inline SmtString _smt_and(const Args&... conds) {")
            self.lines.append("    if constexpr (sizeof...(Args) == 0) return \"true\";")
            self.lines.append("    if constexpr (sizeof...(Args) == 1) {")
            self.lines.append("        SmtString result;")
            self.lines.append("        (_smt_append_atom(result, conds), ...);")
            self.lines.append("        return result;")
            self.lines.append("    }")
            self.lines.append("    SmtString result = \"(and\";")
            self.lines.append("    ((result += ' ', _smt_append_atom(result, conds)), ...);")
            self.lines.append("    result += \")\";")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("static inline SmtString _smt_add(const std::vector<SmtString>& terms) {")
            self.lines.append("    if (terms.empty()) return \"0.0\";")
            self.lines.append("    if (terms.size() == 1) return terms.front();")
            self.lines.append("    SmtString result = \"(+\";")
            self.lines.append("    for (const auto& term : terms) result += \" \" + term;")
            self.lines.append("    result += \")\";")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("static inline SmtString _smt_max(const std::vector<SmtString>& terms) {")
            self.lines.append("    if (terms.empty()) return \"(- 1.0e309)\";")
            self.lines.append("    SmtString acc = terms.back();")
            self.lines.append("    for (std::size_t i = terms.size() - 1; i-- > 0;) {")
            self.lines.append("        acc = \"(ite (>= \" + terms[i] + \" \" + acc + \") \" + terms[i] + \" \" + acc + \")\";")
            self.lines.append("    }")
            self.lines.append("    return acc;")
            self.lines.append("}")
            self.lines.append("static inline SmtString _smt_min(const std::vector<SmtString>& terms) {")
            self.lines.append("    if (terms.empty()) return \"0.0\";")
            self.lines.append("    SmtString acc = terms.back();")
            self.lines.append("    for (std::size_t i = terms.size() - 1; i-- > 0;) {")
            self.lines.append("        acc = \"(ite (<= \" + terms[i] + \" \" + acc + \") \" + terms[i] + \" \" + acc + \")\";")
            self.lines.append("    }")
            self.lines.append("    return acc;")
            self.lines.append("}")
            self.lines.append("template <typename F>")
            self.lines.append("static inline SmtString _smt_add_range(long long n, F f) {")
            self.lines.append("    if (n <= 0) return \"0.0\";")
            self.lines.append("    std::pmr::vector<SmtString> terms;")
            self.lines.append("    terms.reserve(static_cast<std::size_t>(n));")
            self.lines.append("    for (long long i = 0; i < n; ++i) {")
            self.lines.append("        SmtString term = f(i);")
            self.lines.append("        if (term == \"0.0\") continue;")
            self.lines.append("        terms.push_back(std::move(term));")
            self.lines.append("    }")
            self.lines.append("    if (terms.empty()) return SmtString(\"0.0\");")
            self.lines.append("    if (terms.size() == 1) return std::move(terms.front());")
            self.lines.append("    std::size_t total = 3;")
            self.lines.append("    for (const auto& term : terms) total += term.size() + 1;")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(total);")
            self.lines.append("    result += \"(+\";")
            self.lines.append("    for (const auto& term : terms) {")
            self.lines.append("        result += ' ';")
            self.lines.append("        result += term;")
            self.lines.append("    }")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <std::size_t Rank, typename F>")
            self.lines.append("static inline SmtString _smt_add_nd(const std::array<long long, Rank>& bounds, F f) {")
            self.lines.append("    if constexpr (Rank == 0) return f(std::array<long long, 0>{});")
            self.lines.append("    for (long long bound : bounds) if (bound <= 0) return \"0.0\";")
            self.lines.append("    std::array<long long, Rank> idx{};")
            self.lines.append("    std::size_t capacity = 1;")
            self.lines.append("    for (long long bound : bounds) capacity *= static_cast<std::size_t>(bound);")
            self.lines.append("    std::pmr::vector<SmtString> terms;")
            self.lines.append("    terms.reserve(capacity);")
            self.lines.append("    while (true) {")
            self.lines.append("        SmtString term = f(idx);")
            self.lines.append("        if (term != \"0.0\") terms.push_back(std::move(term));")
            self.lines.append("        std::size_t axis = Rank;")
            self.lines.append("        while (axis > 0) {")
            self.lines.append("            --axis;")
            self.lines.append("            if (++idx[axis] < bounds[axis]) goto next_index;")
            self.lines.append("            idx[axis] = 0;")
            self.lines.append("        }")
            self.lines.append("        break;")
            self.lines.append("next_index:;")
            self.lines.append("    }")
            self.lines.append("    if (terms.empty()) return SmtString(\"0.0\");")
            self.lines.append("    if (terms.size() == 1) return std::move(terms.front());")
            self.lines.append("    std::size_t total = 3;")
            self.lines.append("    for (const auto& term : terms) total += term.size() + 1;")
            self.lines.append("    SmtString result;")
            self.lines.append("    result.reserve(total);")
            self.lines.append("    result += \"(+\";")
            self.lines.append("    for (const auto& term : terms) {")
            self.lines.append("        result += ' ';")
            self.lines.append("        result += term;")
            self.lines.append("    }")
            self.lines.append("    result += ')';")
            self.lines.append("    return result;")
            self.lines.append("}")
            self.lines.append("template <typename F>")
            self.lines.append("static inline SmtString _smt_max_range(long long n, F f) {")
            self.lines.append("    if (n <= 0) return \"(- 1.0e309)\";")
            self.lines.append("    SmtString result;")
            self.lines.append("    bool has_term = false;")
            self.lines.append("    for (long long i = 0; i < n; ++i) {")
            self.lines.append("        SmtString term = f(i);")
            self.lines.append("        if (term == \"(- 1.0e309)\") continue;")
            self.lines.append("        if (!has_term) {")
            self.lines.append("            result = std::move(term);")
            self.lines.append("            has_term = true;")
            self.lines.append("            continue;")
            self.lines.append("        }")
            self.lines.append("        if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'a' && result[3] == 'x') {")
            self.lines.append("            result.pop_back();")
            self.lines.append("            result += ' ';")
            self.lines.append("            result += term;")
            self.lines.append("            result += ')';")
            self.lines.append("        } else {")
            self.lines.append("            SmtString next = \"(max \";")
            self.lines.append("            next += result;")
            self.lines.append("            next += ' ';")
            self.lines.append("            next += term;")
            self.lines.append("            next += ')';")
            self.lines.append("            result.swap(next);")
            self.lines.append("        }")
            self.lines.append("    }")
            self.lines.append("    return has_term ? result : SmtString(\"(- 1.0e309)\");")
            self.lines.append("}")
            self.lines.append("template <std::size_t Rank, typename F>")
            self.lines.append("static inline SmtString _smt_max_nd(const std::array<long long, Rank>& bounds, F f) {")
            self.lines.append("    if constexpr (Rank == 0) return f(std::array<long long, 0>{});")
            self.lines.append("    for (long long bound : bounds) if (bound <= 0) return \"(- 1.0e309)\";")
            self.lines.append("    std::array<long long, Rank> idx{};")
            self.lines.append("    SmtString result;")
            self.lines.append("    bool has_term = false;")
            self.lines.append("    while (true) {")
            self.lines.append("        SmtString term = f(idx);")
            self.lines.append("        if (term != \"(- 1.0e309)\") {")
            self.lines.append("            if (!has_term) {")
            self.lines.append("                result = std::move(term);")
            self.lines.append("                has_term = true;")
            self.lines.append("            } else if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'a' && result[3] == 'x') {")
            self.lines.append("                result.pop_back();")
            self.lines.append("                result += ' ';")
            self.lines.append("                result += term;")
            self.lines.append("                result += ')';")
            self.lines.append("            } else {")
            self.lines.append("                SmtString next = \"(max \";")
            self.lines.append("                next += result;")
            self.lines.append("                next += ' ';")
            self.lines.append("                next += term;")
            self.lines.append("                next += ')';")
            self.lines.append("                result.swap(next);")
            self.lines.append("            }")
            self.lines.append("        }")
            self.lines.append("        std::size_t axis = Rank;")
            self.lines.append("        while (axis > 0) {")
            self.lines.append("            --axis;")
            self.lines.append("            if (++idx[axis] < bounds[axis]) goto next_index;")
            self.lines.append("            idx[axis] = 0;")
            self.lines.append("        }")
            self.lines.append("        break;")
            self.lines.append("next_index:;")
            self.lines.append("    }")
            self.lines.append("    return has_term ? result : SmtString(\"(- 1.0e309)\");")
            self.lines.append("}")
            self.lines.append("template <typename F>")
            self.lines.append("static inline SmtString _smt_min_range(long long n, F f) {")
            self.lines.append("    if (n <= 0) return \"0.0\";")
            self.lines.append("    SmtString result;")
            self.lines.append("    bool has_term = false;")
            self.lines.append("    for (long long i = 0; i < n; ++i) {")
            self.lines.append("        SmtString term = f(i);")
            self.lines.append("        if (term == \"0.0\") continue;")
            self.lines.append("        if (!has_term) {")
            self.lines.append("            result = std::move(term);")
            self.lines.append("            has_term = true;")
            self.lines.append("            continue;")
            self.lines.append("        }")
            self.lines.append("        if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'i' && result[3] == 'n') {")
            self.lines.append("            result.pop_back();")
            self.lines.append("            result += ' ';")
            self.lines.append("            result += term;")
            self.lines.append("            result += ')';")
            self.lines.append("        } else {")
            self.lines.append("            SmtString next = \"(min \";")
            self.lines.append("            next += result;")
            self.lines.append("            next += ' ';")
            self.lines.append("            next += term;")
            self.lines.append("            next += ')';")
            self.lines.append("            result.swap(next);")
            self.lines.append("        }")
            self.lines.append("    }")
            self.lines.append("    return has_term ? result : SmtString(\"0.0\");")
            self.lines.append("}")
            self.lines.append("template <std::size_t Rank, typename F>")
            self.lines.append("static inline SmtString _smt_min_nd(const std::array<long long, Rank>& bounds, F f) {")
            self.lines.append("    if constexpr (Rank == 0) return f(std::array<long long, 0>{});")
            self.lines.append("    for (long long bound : bounds) if (bound <= 0) return \"0.0\";")
            self.lines.append("    std::array<long long, Rank> idx{};")
            self.lines.append("    SmtString result;")
            self.lines.append("    bool has_term = false;")
            self.lines.append("    while (true) {")
            self.lines.append("        SmtString term = f(idx);")
            self.lines.append("        if (term != \"0.0\") {")
            self.lines.append("            if (!has_term) {")
            self.lines.append("                result = std::move(term);")
            self.lines.append("                has_term = true;")
            self.lines.append("            } else if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'i' && result[3] == 'n') {")
            self.lines.append("                result.pop_back();")
            self.lines.append("                result += ' ';")
            self.lines.append("                result += term;")
            self.lines.append("                result += ')';")
            self.lines.append("            } else {")
            self.lines.append("                SmtString next = \"(min \";")
            self.lines.append("                next += result;")
            self.lines.append("                next += ' ';")
            self.lines.append("                next += term;")
            self.lines.append("                next += ')';")
            self.lines.append("                result.swap(next);")
            self.lines.append("            }")
            self.lines.append("        }")
            self.lines.append("        std::size_t axis = Rank;")
            self.lines.append("        while (axis > 0) {")
            self.lines.append("            --axis;")
            self.lines.append("            if (++idx[axis] < bounds[axis]) goto next_index;")
            self.lines.append("            idx[axis] = 0;")
            self.lines.append("        }")
            self.lines.append("        break;")
            self.lines.append("next_index:;")
            self.lines.append("    }")
            self.lines.append("    return has_term ? result : SmtString(\"0.0\");")
            self.lines.append("}")
            self.lines.append("")
            self.lines.append(f"static constexpr long long kOutputSize = {product(self.spec.output_shape)}LL;")
            self.lines.append("static inline bool _flat_coord_in_bounds(long long flat_coord) {")
            self.lines.append("    return flat_coord >= 0 && flat_coord < kOutputSize;")
            self.lines.append("}")
            self.lines.append("")
            self.lines.append("SmtString %s(long long flat_coord) {" % self.function_name)
            self.lines.append("    const SmtString zero = \"0.0\";")
            self.lines.append("    const SmtString neg_inf = \"(- 1.0e309)\";")
            self.lines.append("    if (!_flat_coord_in_bounds(flat_coord)) return _smt_symbol(\"invalid_flat_coord\");")
            self.lines.append("    Coord coord{};")
            if self.spec.output_shape:
                self.lines.append("    long long remaining = flat_coord;")
                for idx in range(len(self.spec.output_shape) - 1, -1, -1):
                    dim = int(self.spec.output_shape[idx])
                    self.lines.append(f"    coord[{idx}] = remaining % {dim}LL;")
                    if idx > 0:
                        self.lines.append(f"    remaining /= {dim}LL;")
        elif self.backend == "python":
            self.lines.append("import math")
        self.lines.append("")
        if self.backend != "cpp":
            signature_parts = ["coord", *self.spec.forward_arg_names]
            if self.spec.state_names:
                signature_parts.extend(["*", *self.spec.state_names])
            signature = ", ".join(signature_parts)
            self.lines.append(f"def {self.function_name}({signature}):")
        if self.backend == "python":
            self.lines.append("    _scalar = lambda v: v.item() if hasattr(v, 'item') else v")
            self.lines.append("    zero = 0.0")
            self.lines.append("    neg_inf = float('-inf')")
        elif self.backend == "smt":
            self.lines.append("    zero = '0.0'")
            self.lines.append("    neg_inf = '(- 1.0e309)'")
            self.lines.append("    _smt_select = lambda name, idxs: name if not idxs else f\"(select {_smt_select(name, idxs[:-1])} {idxs[-1]})\"")
            self.lines.append("    _smt_fun = lambda name, arg: f\"({name} {arg})\"")
            self.lines.append("    _smt_add2 = lambda a, b: f\"(+ {a} {b})\"")
            self.lines.append("    _smt_sub2 = lambda a, b: f\"(- {a} {b})\"")
            self.lines.append("    _smt_mul2 = lambda a, b: f\"(* {a} {b})\"")
            self.lines.append("    _smt_div2 = lambda a, b: f\"(/ {a} {b})\"")
            self.lines.append("    _smt_max2 = lambda a, b: f\"(ite (>= {a} {b}) {a} {b})\"")
            self.lines.append("    _smt_min2 = lambda a, b: f\"(ite (<= {a} {b}) {a} {b})\"")
            self.lines.append("    _smt_add = lambda terms: zero if not terms else (terms[0] if len(terms) == 1 else f\"(+ {' '.join(terms)})\")")
            self.lines.append("    _smt_max = lambda terms: neg_inf if not terms else (terms[0] if len(terms) == 1 else f\"(ite (>= {terms[0]} {_smt_max(terms[1:])}) {terms[0]} {_smt_max(terms[1:])})\")")
            self.lines.append("    _smt_min = lambda terms: zero if not terms else (terms[0] if len(terms) == 1 else f\"(ite (<= {terms[0]} {_smt_min(terms[1:])}) {terms[0]} {_smt_min(terms[1:])})\")")
        current: ValueRef | None = None
        for stmt in self.spec.functional_model.body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise FunctionalToLambdaError("Only simple name assignments are supported")
                target = stmt.targets[0].id
                value = self.lower_expr(stmt.value)
                if isinstance(stmt.value, ast.Call) and self.is_alias_call(stmt.value):
                    self.env[target] = value
                    continue
                name = self.new_name()
                coord_vars = self.coord_vars_for_shape(value.shape)
                if self.backend == "cpp":
                    param_decl = self.cpp_coord_param_decl(value.shape)
                    param_vars = self.coord_vars_for_shape(value.shape, source="c")
                    self.lines.append(f"    const auto {name} = [&](%s) -> SmtString {{ return %s; }};" % (param_decl, value.render(param_vars)))
                    ref = ValueRef(shape=value.shape, renderer=lambda coord, name=name: f"{name}({self.coord_expr(coord)})")
                else:
                    self.lines.append(f"    {name} = lambda coord: {value.render(coord_vars)}")
                    ref = lambda_value(name, value.shape)
                self.env[target] = ref
                current = ref
            elif isinstance(stmt, ast.Return):
                value = self.lower_expr(stmt.value)
                current = value
            else:
                raise FunctionalToLambdaError(f"Unsupported statement: {ast.unparse(stmt)}")
        if current is None:
            raise FunctionalToLambdaError("No return value generated")
        if current.shape != self.spec.output_shape:
            raise FunctionalToLambdaError(
                f"Lowered output shape {current.shape} does not match meta output shape {self.spec.output_shape}"
            )
        return_coord = self.coord_vars_for_shape(current.shape)
        self.lines.append(f"    return {current.render(return_coord)};" if self.backend == "cpp" else f"    return {current.render(return_coord)}")
        if self.backend == "cpp":
            self.lines.append("}")
            self.lines.append("")
            self.lines.append("int main(int argc, char** argv) {")
            self.lines.append("    if (argc != 2) {")
            self.lines.append(
                '        std::cerr << "expected exactly 1 coordinate argument, got " << (argc - 1) << \'\\n\';'
            )
            self.lines.append("        return 1;")
            self.lines.append("    }")
            self.lines.append("    std::pmr::unsynchronized_pool_resource smt_resource;")
            self.lines.append("    std::pmr::set_default_resource(&smt_resource);")
            self.lines.append("    long long flat_coord = std::strtoll(argv[1], nullptr, 10);")
            self.lines.append("    if (!_flat_coord_in_bounds(flat_coord)) {")
            self.lines.append('        std::cerr << "flat coordinate out of range: expected in [0, " << kOutputSize << "), got " << flat_coord << \'\\n\';')
            self.lines.append("        return 1;")
            self.lines.append("    }")
            self.lines.append(f"    std::cout << {self.function_name}(flat_coord) << '\\n';")
            self.lines.append("    return 0;")
            self.lines.append("}")
        self.lines.append("")
        return "\n".join(self.lines)

    def is_alias_call(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "detach":
            inner = node.func.value
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "clone":
                return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "clone":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "detach":
            return True
        return False

    def alias_source(self, node: ast.Call) -> ast.AST:
        current: ast.AST = node
        while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute) and current.func.attr in {"clone", "detach"}:
            current = current.func.value
        return current

    def lower_expr(self, node: ast.AST) -> ValueRef:
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise FunctionalToLambdaError(f"Unknown value name: {node.id}")
            return self.env[node.id]

        if isinstance(node, ast.Constant):
            return scalar_value(self.atom(py_literal(node.value)))

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.lower_expr(node.operand)
            if self.backend == "smt":
                return scalar_value(f"_smt_sub2('0.0', {value.render(())})")
            return scalar_value(self.fmt_neg(value.render(())))

        if isinstance(node, ast.BinOp):
            lhs = self.lower_expr(node.left)
            if isinstance(node.op, ast.Pow):
                exponent = self.static_value(node.right)
                if exponent != 2:
                    raise FunctionalToLambdaError(f"Unsupported binary operator: {ast.unparse(node)}")
                rhs = lhs
            else:
                rhs = self.lower_expr(node.right)
            op = {
                ast.Add: self.fmt_add,
                ast.Sub: self.fmt_sub,
                ast.Mult: self.fmt_mul,
                ast.Div: self.fmt_div,
                ast.Pow: self.fmt_mul,
            }.get(type(node.op))
            if op is None:
                raise FunctionalToLambdaError(f"Unsupported binary operator: {ast.unparse(node)}")
            out_shape = broadcast_shape(lhs.shape, rhs.shape)
            return ValueRef(
                shape=out_shape,
                renderer=lambda coord, lhs=lhs, rhs=rhs, out_shape=out_shape, op=op: (
                    op(
                        lhs.render(broadcast_coord(coord, out_shape, lhs.shape)),
                        rhs.render(broadcast_coord(coord, out_shape, rhs.shape)),
                    )
                ),
            )

        if isinstance(node, ast.Subscript):
            return self.lower_subscript(node)

        if isinstance(node, ast.Call):
            return self.lower_call(node)

        raise FunctionalToLambdaError(f"Unsupported expression: {ast.unparse(node)}")

    def lower_subscript(self, node: ast.Subscript) -> ValueRef:
        if isinstance(node.value, ast.Call):
            call = node.value
            callee = full_name(call.func)
            if callee in {"torch.min", "torch.max"} and isinstance(node.slice, ast.Constant) and node.slice.value == 0:
                input_value = self.lower_expr(call.args[0])
                dim = self.keyword_value(call, "dim")
                if dim is None and len(call.args) >= 2:
                    dim = self.static_value(call.args[1])
                keepdim = bool(self.keyword_value(call, "keepdim", False))
                return self.reduce_extreme(input_value, dim, keepdim, "min" if callee == "torch.min" else "max")
        raise FunctionalToLambdaError(f"Unsupported indexing pattern: {ast.unparse(node)}")

    def lower_call(self, node: ast.Call) -> ValueRef:
        if self.is_alias_call(node):
            return self.lower_expr(self.alias_source(node))
        callee = full_name(node.func)

        if callee in {"F.conv1d", "F.conv2d", "F.conv3d"}:
            return self.lower_conv(node, transpose=False)
        if callee in {"F.conv_transpose2d", "F.conv_transpose3d"}:
            return self.lower_conv(node, transpose=True)
        if callee == "F.linear":
            return self.lower_linear(node)
        if callee == "F.group_norm":
            return self.lower_group_norm(node)
        if callee == "F.instance_norm":
            return self.lower_instance_norm(node)
        if callee in {"F.max_pool1d", "F.max_pool2d", "F.max_pool3d"}:
            return self.lower_max_pool(node)
        if callee == "F.avg_pool1d":
            return self.lower_avg_pool1d(node)
        if callee == "F.adaptive_avg_pool3d":
            return self.lower_adaptive_avg_pool3d(node)
        if callee in {"torch.sum"}:
            return self.lower_sum(node)
        if callee in {"torch.mean"}:
            return self.lower_mean(node, self.lower_expr(node.args[0]), dim_arg_index=1)
        if callee in {"torch.min"}:
            return self.lower_torch_min(node)
        if callee in {"torch.max"}:
            return self.lower_torch_max(node)
        if callee in {"torch.clamp"}:
            return self.lower_clamp(node)
        if callee in {"torch.matmul"}:
            return self.lower_matmul(node)
        if callee in {"torch.bmm"}:
            return self.lower_bmm(node)
        if callee in {"torch.cumsum"}:
            return self.lower_cumsum(node)
        if callee in {"torch.softmax", "F.softmax"}:
            return self.lower_softmax(node)
        if callee in {"torch.relu"}:
            return self.lower_unary(node, lambda e: self.fmt_max("0.0", e))
        if callee in {"torch.abs"}:
            return self.lower_unary(node, lambda e: self.fmt_abs(e))
        if callee in {"torch.tanh"}:
            return self.lower_unary(node, lambda e: self.fmt_tanh(e))
        if callee in {"torch.sigmoid"}:
            return self.lower_unary(node, lambda e: self.fmt_div("1.0", self.fmt_add("1.0", self.fmt_exp(self.fmt_neg(e)))))
        if callee in {"torch.nn.functional.leaky_relu", "F.leaky_relu"}:
            return self.lower_leaky_relu(node)
        if callee in {"torch.nn.functional.hardswish"}:
            return self.lower_unary(node, lambda e: self.fmt_div(self.fmt_mul(e, self.fmt_min(self.fmt_max(self.fmt_add(e, "3.0"), "0.0"), "6.0")), "6.0"))
        if callee in {"torch.nn.functional.mish"}:
            return self.lower_unary(
                node,
                lambda e: self.fmt_mul(
                    e,
                    self.fmt_tanh(
                        self.fmt_add(
                            f"(log1p {self.fmt_exp(self.fmt_neg(self.fmt_abs(e)))})"
                            if self.backend == "smt"
                            else (f"_smt_fun(\"log1p\", {self.fmt_exp(self.fmt_neg(self.fmt_abs(e)))})" if self.backend == "cpp" else f"math.log1p({self.fmt_exp(f'-{self.fmt_abs(e)}')})"),
                            self.fmt_max(e, "0.0"),
                        )
                    ),
                ),
            )
        if callee in {"torch.nn.functional.gelu"}:
            return self.lower_unary(node, lambda e: self.fmt_mul(self.fmt_mul("0.5", e), self.fmt_add("1.0", self.fmt_erf(self.fmt_mul(e, "0.7071067811865476")))))
        if callee == "torch.tensor":
            value = self.static_value(node.args[0])
            return scalar_value(self.atom(py_literal(value)))

        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method == "mean":
                return self.lower_mean(node, self.lower_expr(node.func.value), dim_arg_index=0)
            if method == "unsqueeze":
                input_value = self.lower_expr(node.func.value)
                dim = self.static_value(node.args[0])
                return self.unsqueeze(input_value, dim)
            if method == "squeeze":
                input_value = self.lower_expr(node.func.value)
                dim = self.static_value(node.args[0]) if node.args else None
                return self.squeeze(input_value, dim)

        raise FunctionalToLambdaError(f"Unsupported call: {ast.unparse(node)}")

    def lower_unary(self, node: ast.Call, render_fn: Callable[[str], str]) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        return ValueRef(shape=input_value.shape, renderer=lambda coord, input_value=input_value: render_fn(input_value.render(coord)))

    def lower_optional_value(self, node: ast.AST | None) -> ValueRef | None:
        if node is None:
            return None
        if isinstance(node, ast.Constant) and node.value is None:
            return None
        if isinstance(node, ast.Name) and self.static_env.get(node.id) is None:
            return None
        return self.lower_expr(node)

    def render_optional_bias(self, bias: ValueRef | None, index: str) -> str | None:
        if bias is None:
            return None
        if bias.shape == ():
            return bias.render(())
        if bias.shape == (1,):
            return bias.render(["0"])
        if len(bias.shape) == 1:
            return bias.render([index])
        raise FunctionalToLambdaError(f"Unsupported bias shape: {bias.shape}")

    def normalize_reduction_dims(self, input_value: ValueRef, dim: Any) -> list[int]:
        if dim is None:
            return list(range(len(input_value.shape)))
        if isinstance(dim, (list, tuple)):
            return [int(item) for item in dim]
        return [int(dim)]

    def unsqueeze(self, value: ValueRef, dim: int) -> ValueRef:
        rank = len(value.shape)
        dim = dim if dim >= 0 else dim + rank + 1
        out_shape = value.shape[:dim] + (1,) + value.shape[dim:]
        return ValueRef(
            shape=out_shape,
            renderer=lambda coord, value=value, dim=dim: value.render(list(coord[:dim]) + list(coord[dim + 1 :])),
        )

    def squeeze(self, value: ValueRef, dim: int | None) -> ValueRef:
        if dim is None:
            dims = [idx for idx, size in enumerate(value.shape) if size == 1]
            out_shape = tuple(size for size in value.shape if size != 1)

            def render(coord: Sequence[str], value: ValueRef = value, dims: list[int] = dims) -> str:
                coord_iter = iter(coord)
                source: list[str] = []
                for axis, size in enumerate(value.shape):
                    source.append("0" if axis in dims and size == 1 else next(coord_iter))
                return value.render(source)

            return ValueRef(shape=out_shape, renderer=render)

        dim = dim if dim >= 0 else dim + len(value.shape)
        if value.shape[dim] != 1:
            return value
        out_shape = value.shape[:dim] + value.shape[dim + 1 :]
        return ValueRef(
            shape=out_shape,
            renderer=lambda coord, value=value, dim=dim: value.render(list(coord[:dim]) + ["0"] + list(coord[dim:])),
        )

    def lower_linear(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        weight = self.lower_expr(node.args[1])
        bias = self.lower_optional_value(node.args[2] if len(node.args) > 2 else None)
        if len(input_value.shape) != 2 or len(weight.shape) != 2:
            raise FunctionalToLambdaError("Only rank-2 linear is supported")
        batch, in_features = input_value.shape
        out_features = weight.shape[0]
        out_shape = (batch, out_features)

        def render(coord: Sequence[str], input_value=input_value, weight=weight, bias=bias, in_features=in_features) -> str:
            b, o = coord
            if self.backend == "smt":
                summation = render_nested_smt_sum([in_features], lambda vars_: self.fmt_mul(weight.render([o, vars_[0]]), input_value.render([b, vars_[0]])), "li_")
            elif self.backend == "cpp":
                summation = render_nested_cpp_sum([in_features], lambda vars_: self.fmt_mul(weight.render([o, vars_[0]]), input_value.render([b, vars_[0]])), "li_")
            else:
                summation = render_nested_sum([in_features], lambda vars_: self.fmt_mul(weight.render([o, vars_[0]]), input_value.render([b, vars_[0]])), "li_")
            bias_expr = self.render_optional_bias(bias, o)
            return self.fmt_add(summation, bias_expr) if bias_expr is not None else summation

        return ValueRef(shape=out_shape, renderer=render)

    def lower_conv(self, node: ast.Call, transpose: bool) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        weight = self.lower_expr(node.args[1])
        bias = self.lower_optional_value(node.args[2] if len(node.args) > 2 else None)
        spatial_ndim = len(input_value.shape) - 2
        stride = normalize_tuple(self.keyword_value(node, "stride", 1), spatial_ndim)
        padding = normalize_tuple(self.keyword_value(node, "padding", 0), spatial_ndim)
        dilation = normalize_tuple(self.keyword_value(node, "dilation", 1), spatial_ndim)
        groups = int(self.keyword_value(node, "groups", 1))
        if transpose:
            output_padding = normalize_tuple(self.keyword_value(node, "output_padding", 0), spatial_ndim)
            in_channels = input_value.shape[1]
            out_per_group = weight.shape[1]
            out_channels = out_per_group * groups
            in_per_group = in_channels // groups
            out_spatial = tuple(
                conv_transpose_output_size(input_value.shape[2 + idx], weight.shape[2 + idx], stride[idx], padding[idx], dilation[idx], output_padding[idx])
                for idx in range(spatial_ndim)
            )
            out_shape = (input_value.shape[0], out_channels, *out_spatial)

            def render(coord: Sequence[str]) -> str:
                b = coord[0]
                oc = coord[1]
                spatial = list(coord[2:])
                oc_local = oc if groups == 1 else self.fmt_mod(oc, str(out_per_group))
                group_base = "0" if groups == 1 else f"({self.fmt_int_div(oc, str(out_per_group))} * {in_per_group})"
                reduce_bounds = [in_per_group, *weight.shape[2:]]

                def body(vars_: list[str]) -> str:
                    ic_local = vars_[0]
                    kernels = vars_[1:]
                    input_coords: list[str] = []
                    conds: list[str] = []
                    for axis, (out_axis, k, in_size) in enumerate(zip(spatial, kernels, input_value.shape[2:])):
                        numer = affine_expr([(out_axis, 1), (k, -dilation[axis])], padding[axis])
                        coord_expr = self.fmt_int_div(numer, str(stride[axis]))
                        input_coords.append(coord_expr)
                        conds.append(self.fmt_eq(self.fmt_mod(numer, str(stride[axis])), "0"))
                        conds.append(self.fmt_in_bounds(coord_expr, in_size))
                    ic_global = ic_local if groups == 1 else f"({group_base} + {ic_local})"
                    prod_expr = self.fmt_mul(weight.render([ic_global, oc_local, *kernels]), input_value.render([b, ic_global, *input_coords]))
                    return self.fmt_if(self.fmt_and(conds), prod_expr, "zero")

                if self.backend == "smt":
                    conv_expr = render_nested_smt_sum(reduce_bounds, body, "tconv_")
                elif self.backend == "cpp":
                    conv_expr = render_nested_cpp_sum(reduce_bounds, body, "tconv_")
                else:
                    conv_expr = render_nested_sum(reduce_bounds, body, "tconv_")
                bias_expr = self.render_optional_bias(bias, oc)
                return self.fmt_add(conv_expr, bias_expr) if bias_expr is not None else conv_expr

            return ValueRef(shape=out_shape, renderer=render)

        in_channels = input_value.shape[1]
        out_channels = weight.shape[0]
        in_per_group = weight.shape[1]
        out_per_group = out_channels // groups
        out_spatial = tuple(
            conv_output_size(input_value.shape[2 + idx], weight.shape[2 + idx], stride[idx], padding[idx], dilation[idx])
            for idx in range(spatial_ndim)
        )
        out_shape = (input_value.shape[0], out_channels, *out_spatial)

        def render(coord: Sequence[str]) -> str:
            b = coord[0]
            oc = coord[1]
            spatial = list(coord[2:])
            reduce_bounds = [in_per_group, *weight.shape[2:]]
            group_base = "0" if groups == 1 else f"({self.fmt_int_div(oc, str(out_per_group))} * {in_per_group})"

            def body(vars_: list[str]) -> str:
                ic_local = vars_[0]
                kernels = vars_[1:]
                ic_global = ic_local if groups == 1 else f"({group_base} + {ic_local})"
                input_coords = [
                    affine_expr([(out_axis, stride[axis]), (k, dilation[axis])], -padding[axis])
                    for axis, (out_axis, k) in enumerate(zip(spatial, kernels))
                ]
                conds = [
                    self.fmt_in_bounds(input_coords[idx], input_value.shape[2 + idx])
                    for idx in range(spatial_ndim)
                ]
                prod_expr = self.fmt_mul(weight.render([oc, ic_local, *kernels]), input_value.render([b, ic_global, *input_coords]))
                return self.fmt_if(self.fmt_and(conds), prod_expr, "zero")

            if self.backend == "smt":
                conv_expr = render_nested_smt_sum(reduce_bounds, body, "conv_")
            elif self.backend == "cpp":
                conv_expr = render_nested_cpp_sum(reduce_bounds, body, "conv_")
            else:
                conv_expr = render_nested_sum(reduce_bounds, body, "conv_")
            bias_expr = self.render_optional_bias(bias, oc)
            return self.fmt_add(conv_expr, bias_expr) if bias_expr is not None else conv_expr

        return ValueRef(shape=out_shape, renderer=render)

    def lower_mean(self, node: ast.Call, input_value: ValueRef, dim_arg_index: int) -> ValueRef:
        dim = self.keyword_value(node, "dim")
        if dim is None and len(node.args) > dim_arg_index:
            dim = self.static_value(node.args[dim_arg_index])
        keepdim = bool(self.keyword_value(node, "keepdim", False))
        dims = self.normalize_reduction_dims(input_value, dim)
        if not dims:
            return input_value
        return self.reduce_mean(input_value, dims, keepdim)

    def reduce_mean(self, input_value: ValueRef, dims: Sequence[int], keepdim: bool) -> ValueRef:
        out_shape = reduction_shape(input_value.shape, dims, keepdim)
        bounds = [input_value.shape[dim if dim >= 0 else dim + len(input_value.shape)] for dim in dims]
        count = product(bounds)

        def render(coord: Sequence[str], input_value=input_value, dims=dims, keepdim=keepdim, count=count, bounds=bounds) -> str:
            if self.backend == "smt":
                numer = render_nested_smt_sum(bounds, lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)), "mean_")
            elif self.backend == "cpp":
                numer = render_nested_cpp_sum(bounds, lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)), "mean_")
            else:
                numer = render_nested_sum(bounds, lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)), "mean_")
            return self.fmt_div(numer, str(count))

        return ValueRef(shape=out_shape, renderer=render)

    def lower_sum(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        dim = self.keyword_value(node, "dim")
        if dim is None and len(node.args) > 1:
            dim = self.static_value(node.args[1])
        keepdim = bool(self.keyword_value(node, "keepdim", False))
        dims = self.normalize_reduction_dims(input_value, dim)
        out_shape = reduction_shape(input_value.shape, dims, keepdim)
        bounds = [input_value.shape[d if d >= 0 else d + len(input_value.shape)] for d in dims]

        def render(coord: Sequence[str], input_value=input_value, dims=dims, keepdim=keepdim, bounds=bounds) -> str:
            if self.backend == "smt":
                return render_nested_smt_sum(
                    bounds,
                    lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)),
                    "sum_",
                )
            if self.backend == "cpp":
                return render_nested_cpp_sum(
                    bounds,
                    lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)),
                    "sum_",
                )
            return render_nested_sum(
                bounds,
                lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)),
                "sum_",
            )

        return ValueRef(shape=out_shape, renderer=render)

    def lower_matmul(self, node: ast.Call) -> ValueRef:
        lhs = self.lower_expr(node.args[0])
        rhs = self.lower_expr(node.args[1])
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise FunctionalToLambdaError("Only rank-2 torch.matmul is supported")
        if lhs.shape[1] != rhs.shape[0]:
            raise FunctionalToLambdaError(f"torch.matmul dimension mismatch: {lhs.shape} x {rhs.shape}")
        m, k = lhs.shape
        _, n = rhs.shape
        out_shape = (m, n)

        def render(coord: Sequence[str], lhs=lhs, rhs=rhs, k=k) -> str:
            i, j = coord
            if self.backend == "smt":
                return render_nested_smt_sum([k], lambda vars_: self.fmt_mul(lhs.render([i, vars_[0]]), rhs.render([vars_[0], j])), "matmul_")
            if self.backend == "cpp":
                return render_nested_cpp_sum([k], lambda vars_: self.fmt_mul(lhs.render([i, vars_[0]]), rhs.render([vars_[0], j])), "matmul_")
            return render_nested_sum([k], lambda vars_: self.fmt_mul(lhs.render([i, vars_[0]]), rhs.render([vars_[0], j])), "matmul_")

        return ValueRef(shape=out_shape, renderer=render)

    def lower_bmm(self, node: ast.Call) -> ValueRef:
        lhs = self.lower_expr(node.args[0])
        rhs = self.lower_expr(node.args[1])
        if len(lhs.shape) != 3 or len(rhs.shape) != 3:
            raise FunctionalToLambdaError("Only rank-3 torch.bmm is supported")
        if lhs.shape[0] != rhs.shape[0] or lhs.shape[2] != rhs.shape[1]:
            raise FunctionalToLambdaError(f"torch.bmm dimension mismatch: {lhs.shape} x {rhs.shape}")
        batch, m, k = lhs.shape
        _, _, n = rhs.shape
        out_shape = (batch, m, n)

        def render(coord: Sequence[str], lhs=lhs, rhs=rhs, k=k) -> str:
            b, i, j = coord
            if self.backend == "smt":
                return render_nested_smt_sum([k], lambda vars_: self.fmt_mul(lhs.render([b, i, vars_[0]]), rhs.render([b, vars_[0], j])), "bmm_")
            if self.backend == "cpp":
                return render_nested_cpp_sum([k], lambda vars_: self.fmt_mul(lhs.render([b, i, vars_[0]]), rhs.render([b, vars_[0], j])), "bmm_")
            return render_nested_sum([k], lambda vars_: self.fmt_mul(lhs.render([b, i, vars_[0]]), rhs.render([b, vars_[0], j])), "bmm_")

        return ValueRef(shape=out_shape, renderer=render)

    def lower_cumsum(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        dim = self.keyword_value(node, "dim")
        if dim is None and len(node.args) > 1:
            dim = self.static_value(node.args[1])
        axis = dim if dim >= 0 else dim + len(input_value.shape)

        def render(coord: Sequence[str], input_value=input_value, axis=axis) -> str:
            limit_expr = f"({coord[axis]} + 1)"
            body = lambda idx: input_value.render(list(coord[:axis]) + [idx] + list(coord[axis + 1 :]))
            if self.backend == "smt":
                return f"_smt_add(tuple(map(lambda cumsum_0: {body('cumsum_0')}, range({limit_expr}))))"
            if self.backend == "cpp":
                return f"_smt_add_range({limit_expr}, [&](long long cumsum_0) -> SmtString {{ return {body('cumsum_0')}; }})"
            return f"sum(map(lambda cumsum_0: {body('cumsum_0')}, range({limit_expr})), zero)"

        return ValueRef(shape=input_value.shape, renderer=render)

    def reduce_extreme(self, input_value: ValueRef, dim: int, keepdim: bool, kind: str) -> ValueRef:
        dims = [dim]
        out_shape = reduction_shape(input_value.shape, dims, keepdim)
        axis = dim if dim >= 0 else dim + len(input_value.shape)
        bound = input_value.shape[axis]

        def render(coord: Sequence[str], input_value=input_value, dim=dim, keepdim=keepdim, bound=bound, kind=kind) -> str:
            if self.backend == "smt":
                return render_nested_smt_extreme(
                    kind,
                    [bound],
                    lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), [dim], vars_, keepdim)),
                    f"{kind}_",
                )
            if self.backend == "cpp":
                return render_nested_cpp_extreme(
                    kind,
                    [bound],
                    lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), [dim], vars_, keepdim)),
                    f"{kind}_",
                )
            return render_nested_stack_reduce(
                kind,
                [bound],
                lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), [dim], vars_, keepdim)),
                f"{kind}_",
            )

        return ValueRef(shape=out_shape, renderer=render)

    def lower_torch_min(self, node: ast.Call) -> ValueRef:
        if any(keyword.arg == "dim" for keyword in node.keywords) or len(node.args) == 2 and isinstance(node.args[1], ast.Name) and node.args[1].id == "dim":
            raise FunctionalToLambdaError("torch.min(dim=...) must be indexed with [0]")
        lhs = self.lower_expr(node.args[0])
        rhs = self.lower_expr(node.args[1])
        out_shape = broadcast_shape(lhs.shape, rhs.shape)
        return ValueRef(
            shape=out_shape,
            renderer=lambda coord, lhs=lhs, rhs=rhs, out_shape=out_shape: self.fmt_min(
                lhs.render(broadcast_coord(coord, out_shape, lhs.shape)),
                rhs.render(broadcast_coord(coord, out_shape, rhs.shape)),
            ),
        )

    def lower_torch_max(self, node: ast.Call) -> ValueRef:
        if any(keyword.arg == "dim" for keyword in node.keywords) or len(node.args) == 2 and isinstance(node.args[1], ast.Name) and node.args[1].id == "dim":
            raise FunctionalToLambdaError("torch.max(dim=...) must be indexed with [0]")
        lhs = self.lower_expr(node.args[0])
        rhs = self.lower_expr(node.args[1])
        out_shape = broadcast_shape(lhs.shape, rhs.shape)
        return ValueRef(
            shape=out_shape,
            renderer=lambda coord, lhs=lhs, rhs=rhs, out_shape=out_shape: self.fmt_max(
                lhs.render(broadcast_coord(coord, out_shape, lhs.shape)),
                rhs.render(broadcast_coord(coord, out_shape, rhs.shape)),
            ),
        )

    def lower_clamp(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        min_node = None
        max_node = None
        if len(node.args) > 1:
            min_node = node.args[1]
        if len(node.args) > 2:
            max_node = node.args[2]
        for keyword in node.keywords:
            if keyword.arg == "min":
                min_node = keyword.value
            if keyword.arg == "max":
                max_node = keyword.value
        min_expr = self.lower_expr(min_node).render(()) if min_node is not None else None
        max_expr = self.lower_expr(max_node).render(()) if max_node is not None else None

        def render(coord: Sequence[str], input_value=input_value, min_expr=min_expr, max_expr=max_expr) -> str:
            return self.fmt_clamp(input_value.render(coord), min_expr, max_expr)

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_softmax(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        dim = self.keyword_value(node, "dim")
        if dim is None and len(node.args) > 1:
            dim = self.static_value(node.args[1])
        axis = dim if dim >= 0 else dim + len(input_value.shape)
        bound = input_value.shape[axis]

        def render(coord: Sequence[str], input_value=input_value, axis=axis, bound=bound) -> str:
            numer = self.fmt_exp(input_value.render(coord))
            if self.backend == "smt":
                denom = render_nested_smt_sum(
                    [bound],
                    lambda vars_: self.fmt_exp(input_value.render(list(coord[:axis]) + [vars_[0]] + list(coord[axis + 1 :]))),
                    "softmax_",
                )
            elif self.backend == "cpp":
                denom = render_nested_cpp_sum(
                    [bound],
                    lambda vars_: self.fmt_exp(input_value.render(list(coord[:axis]) + [vars_[0]] + list(coord[axis + 1 :]))),
                    "softmax_",
                )
            else:
                denom = render_nested_sum(
                    [bound],
                    lambda vars_: self.fmt_exp(input_value.render(list(coord[:axis]) + [vars_[0]] + list(coord[axis + 1 :]))),
                    "softmax_",
                )
            return self.fmt_div(numer, denom)

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_leaky_relu(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        negative_slope = self.keyword_value(node, "negative_slope", 0.01)
        if len(node.args) > 1:
            negative_slope = self.static_value(node.args[1])
        slope_expr = self.atom(py_literal(negative_slope))

        def render(coord: Sequence[str], input_value=input_value, slope_expr=slope_expr) -> str:
            expr = input_value.render(coord)
            return self.fmt_if(self.fmt_le("0.0", expr), expr, self.fmt_mul(slope_expr, expr))

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_group_norm(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        weight = self.lower_expr(node.args[2])
        bias = self.lower_expr(node.args[3])
        eps_name = self.atom(node.keywords[0].value.id if node.keywords else "group_norm_eps")
        num_groups = int(self.static_value(node.args[1]))
        channels = input_value.shape[1]
        spatial = input_value.shape[2:]
        channels_per_group = channels // num_groups
        count = channels_per_group * product(spatial)

        def render(coord: Sequence[str]) -> str:
            b = coord[0]
            c = coord[1]
            spatial_coord = list(coord[2:])

            def mean_for(g_expr: str) -> str:
                def body(vars_: list[str]) -> str:
                    channel_expr = f"(({g_expr}) * {channels_per_group} + {vars_[0]})"
                    return input_value.render([b, channel_expr, *vars_[1:]])

                if self.backend == "smt":
                    total = render_nested_smt_sum([channels_per_group, *spatial], body, "gnm_")
                elif self.backend == "cpp":
                    total = render_nested_cpp_sum([channels_per_group, *spatial], body, "gnm_")
                else:
                    total = render_nested_sum([channels_per_group, *spatial], body, "gnm_")
                return self.fmt_div(total, str(count))

            def var_for(g_expr: str, mean_expr: str) -> str:
                def body(vars_: list[str]) -> str:
                    channel_expr = f"(({g_expr}) * {channels_per_group} + {vars_[0]})"
                    sample_expr = input_value.render([b, channel_expr, *vars_[1:]])
                    diff_expr = self.fmt_sub(sample_expr, mean_expr)
                    return self.fmt_mul(diff_expr, diff_expr)

                if self.backend == "smt":
                    total = render_nested_smt_sum([channels_per_group, *spatial], body, "gnv_")
                elif self.backend == "cpp":
                    total = render_nested_cpp_sum([channels_per_group, *spatial], body, "gnv_")
                else:
                    total = render_nested_sum([channels_per_group, *spatial], body, "gnv_")
                return self.fmt_div(total, str(count))

            if self.backend == "cpp":
                return (
                    "([&]() -> SmtString { "
                    f"long long g = {c} / {channels_per_group}; "
                    f"SmtString m = {mean_for('g')}; "
                    f"SmtString v = {var_for('g', 'm')}; "
                    f"return {self.fmt_add(self.fmt_div(self.fmt_mul(weight.render([c]), self.fmt_sub(input_value.render([b, c, *spatial_coord]), 'm')), self.fmt_sqrt(self.fmt_add('v', eps_name))), bias.render([c]))}; "
                    "}())"
                )
            return (
                f"((lambda g: (lambda m: (lambda v: {self.fmt_add(self.fmt_div(self.fmt_mul(weight.render([c]), self.fmt_sub(input_value.render([b, c, *spatial_coord]), 'm')), self.fmt_sqrt(self.fmt_add('v', eps_name))), bias.render([c]))})({var_for('g', 'm')}))({mean_for('g')}))({c} // {channels_per_group}))"
            )

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_instance_norm(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        if len(input_value.shape) != 4:
            raise FunctionalToLambdaError("Only rank-4 instance_norm is supported")
        use_input_stats = self.keyword_value(node, "use_input_stats", True)
        if not use_input_stats:
            raise FunctionalToLambdaError("Only use_input_stats=True instance_norm is supported")
        eps_name = self.atom("instance_norm_eps")
        for keyword in node.keywords:
            if keyword.arg == "eps":
                eps_name = self.atom(ast.unparse(keyword.value))
        weight_name = "instance_norm_weight"
        bias_name = "instance_norm_bias"
        spatial = input_value.shape[2:]
        count = product(spatial)

        def render(coord: Sequence[str]) -> str:
            b, c = coord[0], coord[1]
            spatial_coord = list(coord[2:])
            if self.backend == "smt":
                mean_total = render_nested_smt_sum(list(spatial), lambda vars_: input_value.render([b, c, *vars_]), "inm_")
            elif self.backend == "cpp":
                mean_total = render_nested_cpp_sum(list(spatial), lambda vars_: input_value.render([b, c, *vars_]), "inm_")
            else:
                mean_total = render_nested_sum(list(spatial), lambda vars_: input_value.render([b, c, *vars_]), "inm_")
            mean_expr = self.fmt_div(mean_total, str(count))
            def var_body(vars_: list[str], mean_name: str = "m") -> str:
                diff_expr = self.fmt_sub(input_value.render([b, c, *vars_]), mean_name)
                return self.fmt_mul(diff_expr, diff_expr)
            if self.backend == "smt":
                var_total = render_nested_smt_sum(list(spatial), var_body, "inv_")
            elif self.backend == "cpp":
                var_total = render_nested_cpp_sum(list(spatial), lambda vars_: var_body(vars_, "m"), "inv_")
            else:
                var_total = render_nested_sum(list(spatial), var_body, "inv_")
            var_expr = f"((lambda m: {self.fmt_div(var_total, str(count))})({mean_expr}))"
            if self.backend == "smt":
                weight_expr = f"_smt_select('{weight_name}', ({c},))"
                bias_expr = f"_smt_select('{bias_name}', ({c},))"
            elif self.backend == "cpp":
                weight_expr = f"_smt_select({cpp_string_literal(weight_name)}, {{{c}}})" if self.spec.state_meta.get(weight_name) is not None else cpp_string_literal("1")
                bias_expr = f"_smt_select({cpp_string_literal(bias_name)}, {{{c}}})" if self.spec.state_meta.get(bias_name) is not None else cpp_string_literal("0")
            else:
                weight_expr = f"({weight_name}[{c}] if {weight_name} is not None else 1)"
                bias_expr = f"({bias_name}[{c}] if {bias_name} is not None else 0)"
            if self.backend == "cpp":
                return (
                    "([&]() -> SmtString { "
                    f"SmtString m = {mean_expr}; "
                    f"SmtString v = {self.fmt_div(render_nested_cpp_sum(list(spatial), lambda vars_: var_body(vars_, 'm'), 'inv_'), str(count))}; "
                    f"return {self.fmt_add(self.fmt_div(self.fmt_mul(weight_expr, self.fmt_sub(input_value.render([b, c, *spatial_coord]), 'm')), self.fmt_sqrt(self.fmt_add('v', eps_name))), bias_expr)}; "
                    "}())"
                )
            return (
                f"((lambda m: (lambda v: {self.fmt_add(self.fmt_div(self.fmt_mul(weight_expr, self.fmt_sub(input_value.render([b, c, *spatial_coord]), 'm')), self.fmt_sqrt(self.fmt_add('v', eps_name))), bias_expr)})({var_expr}))({mean_expr}))"
            )

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_max_pool(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        spatial_ndim = len(input_value.shape) - 2
        kernel = normalize_tuple(self.keyword_value(node, "kernel_size", self.static_value(node.args[1]) if len(node.args) > 1 else None), spatial_ndim)
        stride = normalize_tuple(self.keyword_value(node, "stride", kernel), spatial_ndim)
        padding = normalize_tuple(self.keyword_value(node, "padding", 0), spatial_ndim)
        dilation = normalize_tuple(self.keyword_value(node, "dilation", 1), spatial_ndim)
        ceil_mode = bool(self.keyword_value(node, "ceil_mode", False))
        out_spatial = tuple(
            pool_output_size(input_value.shape[2 + idx], kernel[idx], stride[idx], padding[idx], dilation[idx], ceil_mode)
            for idx in range(spatial_ndim)
        )
        out_shape = (input_value.shape[0], input_value.shape[1], *out_spatial)

        def render(coord: Sequence[str]) -> str:
            base = list(coord[:2])
            spatial_coord = list(coord[2:])

            def body(vars_: list[str]) -> str:
                input_coords = [
                    affine_expr([(out_axis, stride[idx]), (vars_[idx], dilation[idx])], -padding[idx])
                    for idx, out_axis in enumerate(spatial_coord)
                ]
                conds = [
                    self.fmt_in_bounds(input_coords[idx], input_value.shape[2 + idx])
                    for idx in range(spatial_ndim)
                ]
                return (
                    self.fmt_if(self.fmt_and(conds), input_value.render([*base, *input_coords]), "neg_inf")
                )

            if self.backend == "smt":
                return render_nested_smt_extreme("max", list(kernel), body, "pool_")
            if self.backend == "cpp":
                return render_nested_cpp_extreme("max", list(kernel), body, "pool_")
            return render_nested_stack_reduce("max", list(kernel), body, "pool_")

        return ValueRef(shape=out_shape, renderer=render)

    def lower_avg_pool1d(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        if len(input_value.shape) != 3:
            raise FunctionalToLambdaError("Only rank-3 avg_pool1d is supported")
        kernel = normalize_tuple(self.keyword_value(node, "kernel_size", self.static_value(node.args[1]) if len(node.args) > 1 else None), 1)[0]
        stride = normalize_tuple(self.keyword_value(node, "stride", kernel), 1)[0]
        padding = normalize_tuple(self.keyword_value(node, "padding", 0), 1)[0]
        ceil_mode = bool(self.keyword_value(node, "ceil_mode", False))
        count_include_pad = bool(self.keyword_value(node, "count_include_pad", True))
        out_length = pool_output_size(input_value.shape[2], kernel, stride, padding, 1, ceil_mode)
        out_shape = (input_value.shape[0], input_value.shape[1], out_length)

        def render(coord: Sequence[str], input_value=input_value, kernel=kernel, stride=stride, padding=padding, count_include_pad=count_include_pad) -> str:
            b, c, out_x = coord

            def input_coord_expr(k_var: str) -> str:
                return affine_expr([(out_x, stride), (k_var, 1)], -padding)

            def value_body(vars_: list[str]) -> str:
                in_x = input_coord_expr(vars_[0])
                return self.fmt_if(self.fmt_in_bounds(in_x, input_value.shape[2]), input_value.render([b, c, in_x]), "zero")

            if self.backend == "smt":
                total = render_nested_smt_sum([kernel], value_body, "avgp1_")
            elif self.backend == "cpp":
                total = render_nested_cpp_sum([kernel], value_body, "avgp1_")
            else:
                total = render_nested_sum([kernel], value_body, "avgp1_")

            if count_include_pad:
                denom = str(kernel)
            else:
                count_body = lambda vars_: self.fmt_if(self.fmt_in_bounds(input_coord_expr(vars_[0]), input_value.shape[2]), "1.0", "0.0")
                if self.backend == "smt":
                    denom = render_nested_smt_sum([kernel], count_body, "avgpc1_")
                elif self.backend == "cpp":
                    denom = render_nested_cpp_sum([kernel], count_body, "avgpc1_")
                else:
                    denom = render_nested_sum([kernel], count_body, "avgpc1_")
            return self.fmt_div(total, denom)

        return ValueRef(shape=out_shape, renderer=render)

    def lower_adaptive_avg_pool3d(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        output_size = self.static_value(node.args[1] if len(node.args) > 1 else node.keywords[0].value)
        if tuple(output_size) != (1, 1, 1):
            raise FunctionalToLambdaError("Only AdaptiveAvgPool3d output_size=(1,1,1) is supported")
        out_shape = (input_value.shape[0], input_value.shape[1], 1, 1, 1)
        count = product(input_value.shape[2:])

        def render(coord: Sequence[str], input_value=input_value, count=count) -> str:
            b, c = coord[0], coord[1]
            if self.backend == "smt":
                total = render_nested_smt_sum(list(input_value.shape[2:]), lambda vars_: input_value.render([b, c, *vars_]), "aap_")
            elif self.backend == "cpp":
                total = render_nested_cpp_sum(list(input_value.shape[2:]), lambda vars_: input_value.render([b, c, *vars_]), "aap_")
            else:
                total = render_nested_sum(list(input_value.shape[2:]), lambda vars_: input_value.render([b, c, *vars_]), "aap_")
            return self.fmt_div(total, str(count))

        return ValueRef(shape=out_shape, renderer=render)


def generate_source(path: str | Path, function_name: str = "value_at", backend: str = "python") -> str:
    spec = load_module_spec(path)
    lowerer = LambdaLowerer(spec, function_name=function_name, backend=backend)
    return lowerer.build()


def batch_output_path(input_root: Path, input_path: Path, output_root: Path, backend: str) -> Path:
    relative_path = input_path.relative_to(input_root)
    suffix = ".cpp" if backend == "cpp" else ".py"
    return output_root / relative_path.with_suffix(suffix)


def iter_input_files(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def write_generated_output(
    input_path: Path,
    output_path: Path,
    *,
    function_name: str,
    backend: str,
) -> None:
    source = generate_source(input_path, function_name=function_name, backend=backend)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "cpp":
        spec = load_module_spec(input_path)
        write_cpp_outputs(spec, source, output_path)
    else:
        output_path.write_text(source)


def generated_outputs_exist(output_path: Path, backend: str) -> bool:
    if backend == "cpp":
        return output_path.exists() and output_path.with_suffix(".json").exists()
    return output_path.exists()


def process_input_file(
    input_root: Path,
    output_root: Path,
    input_path: Path,
    *,
    function_name: str,
    backend: str,
    force: bool,
) -> tuple[Path, str, str | None]:
    relative_path = input_path.relative_to(input_root)
    output_path = batch_output_path(input_root, input_path, output_root, backend)
    if not force and generated_outputs_exist(output_path, backend):
        return relative_path, "skipped", None
    try:
        write_generated_output(
            input_path,
            output_path,
            function_name=function_name,
            backend=backend,
        )
        return relative_path, "succeeded", None
    except Exception as exc:
        return relative_path, "failed", str(exc)


def start_thread_to_terminate_when_parent_process_dies(ppid):
    pid = os.getpid()
    import time
    import threading
    import signal
    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate coordinate-lambda artifacts for a tree of KernelBench functional models")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="KernelBench/KernelBenchFunctional",
        help="Path to the input directory containing functional model files",
    )
    parser.add_argument("--output", default="kcheck_specs", help="Output directory root")
    parser.add_argument("--function-name", default="value_at", help="Generated function name")
    parser.add_argument("--backend", choices=["python", "smt", "cpp"], default="python", help="Generated backend")
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1), help="Number of worker processes for batch processing")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even when the expected output files already exist")
    args = parser.parse_args(argv)

    input_root = Path(args.input_dir)
    output_root = Path(args.output)
    if not input_root.is_dir():
        raise FunctionalToLambdaError(f"Input directory does not exist or is not a directory: {input_root}")
    if args.jobs < 1:
        raise FunctionalToLambdaError(f"--jobs must be at least 1, got {args.jobs}")

    output_root.mkdir(parents=True, exist_ok=True)
    input_files = iter_input_files(input_root)

    success_count = 0
    skipped_count = 0
    failures: list[tuple[Path, str]] = []

    with tqdm(total=len(input_files), desc="Processing", unit="file", smoothing=0) as progress:
        if args.jobs == 1:
            for input_path in input_files:
                relative_path, status, error = process_input_file(
                    input_root,
                    output_root,
                    input_path,
                    function_name=args.function_name,
                    backend=args.backend,
                    force=args.force,
                )
                if status == "succeeded":
                    success_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    failures.append((relative_path, error))
                progress.update(1)
                progress.set_postfix_str(f"Succeeded: {success_count}, Skipped: {skipped_count}, Failed: {len(failures)}")
        else:
            start_method = "fork" if os.name != "nt" else "spawn"
            with ProcessPoolExecutor(
                max_workers=args.jobs,
                mp_context=multiprocessing.get_context(start_method),
                initializer=start_thread_to_terminate_when_parent_process_dies,
                initargs=(os.getpid(),),
            ) as executor:
                try:
                    futures = {
                        executor.submit(
                            process_input_file,
                            input_root,
                            output_root,
                            input_path,
                            function_name=args.function_name,
                            backend=args.backend,
                            force=args.force,
                        ): input_path
                        for input_path in input_files
                    }
                    for future in as_completed(futures):
                        relative_path, status, error = future.result()
                        if status == "succeeded":
                            success_count += 1
                        elif status == "skipped":
                            skipped_count += 1
                        else:
                            failures.append((relative_path, error))
                        progress.update(1)
                        progress.set_postfix_str(f"Succeeded: {success_count}, Skipped: {skipped_count}, Failed: {len(failures)}")
                except KeyboardInterrupt:
                    print("Interrupted, cancelling remaining tasks...")
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=False)

    print(
        f"Processed {len(input_files)} Python files from {input_root} to {output_root}: "
        f"{success_count} succeeded, {skipped_count} skipped, {len(failures)} failed."
    )
    for relative_path, message in failures:
        print(f"FAILED {relative_path}: {message}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
