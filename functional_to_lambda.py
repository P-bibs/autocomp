#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence
import uuid

import torch


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


def tensor_arg_value(name: str, shape: tuple[int, ...]) -> ValueRef:
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
    def __init__(self, spec: ModuleSpec, function_name: str) -> None:
        self.spec = spec
        self.function_name = function_name
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
                self.env[name] = tensor_arg_value(name, tuple(value.shape))
            else:
                self.env[name] = scalar_value(name)
        for name in self.spec.state_names:
            value = self.spec.state_meta[name]
            if isinstance(value, torch.Tensor):
                self.env[name] = tensor_arg_value(name, tuple(value.shape))
            elif value is None:
                self.env[name] = scalar_value("None")
            else:
                self.env[name] = scalar_value(name)

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
        return [f"coord[{idx}]" for idx in range(len(self.spec.output_shape))]

    def build(self) -> str:
        self.lines.append("import math")
        self.lines.append("")
        signature = ", ".join(
            ["coord", *self.spec.forward_arg_names, "*", *self.spec.state_names]
        )
        self.lines.append(f"def {self.function_name}({signature}):")
        self.lines.append("    _scalar = lambda v: v.item() if hasattr(v, 'item') else v")
        self.lines.append("    zero = 0.0")
        self.lines.append("    neg_inf = float('-inf')")
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
                coord_vars = [f"coord[{idx}]" for idx in range(len(value.shape))]
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
        self.lines.append(f"    return {current.render([f'coord[{idx}]' for idx in range(len(current.shape))])}")
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
            return scalar_value(py_literal(node.value))

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.lower_expr(node.operand)
            return scalar_value(f"(-{value.render(())})")

        if isinstance(node, ast.BinOp):
            lhs = self.lower_expr(node.left)
            rhs = self.lower_expr(node.right)
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }.get(type(node.op))
            if op is None:
                raise FunctionalToLambdaError(f"Unsupported binary operator: {ast.unparse(node)}")
            out_shape = broadcast_shape(lhs.shape, rhs.shape)
            return ValueRef(
                shape=out_shape,
                renderer=lambda coord, lhs=lhs, rhs=rhs, out_shape=out_shape, op=op: (
                    f"({lhs.render(broadcast_coord(coord, out_shape, lhs.shape))} {op} "
                    f"{rhs.render(broadcast_coord(coord, out_shape, rhs.shape))})"
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
            if full_name(call.func) == "torch.min" and isinstance(node.slice, ast.Constant) and node.slice.value == 0:
                input_value = self.lower_expr(call.args[0])
                dim = self.keyword_value(call, "dim")
                if dim is None and len(call.args) >= 2:
                    dim = self.static_value(call.args[1])
                keepdim = bool(self.keyword_value(call, "keepdim", False))
                return self.reduce_extreme(input_value, dim, keepdim, "min")
        raise FunctionalToLambdaError(f"Unsupported indexing pattern: {ast.unparse(node)}")

    def lower_call(self, node: ast.Call) -> ValueRef:
        if self.is_alias_call(node):
            return self.lower_expr(self.alias_source(node))
        callee = full_name(node.func)

        if callee in {"F.conv2d", "F.conv3d"}:
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
        if callee == "F.adaptive_avg_pool3d":
            return self.lower_adaptive_avg_pool3d(node)
        if callee in {"torch.sum"}:
            return self.lower_sum(node)
        if callee in {"torch.min"}:
            return self.lower_torch_min(node)
        if callee in {"torch.clamp"}:
            return self.lower_clamp(node)
        if callee in {"torch.softmax", "F.softmax"}:
            return self.lower_softmax(node)
        if callee in {"torch.relu"}:
            return self.lower_unary(node, lambda e: f"max(0.0, {e})")
        if callee in {"torch.tanh"}:
            return self.lower_unary(node, lambda e: f"math.tanh({e})")
        if callee in {"torch.sigmoid"}:
            return self.lower_unary(node, lambda e: f"(1.0 / (1.0 + math.exp(-({e}))))")
        if callee in {"torch.nn.functional.hardswish"}:
            return self.lower_unary(node, lambda e: f"(({e}) * min(max(({e}) + 3.0, 0.0), 6.0) / 6.0)")
        if callee in {"torch.nn.functional.mish"}:
            return self.lower_unary(
                node,
                lambda e: (
                    f"(({e}) * math.tanh(math.log1p(math.exp(-abs({e}))) + max(({e}), 0.0)))"
                ),
            )
        if callee in {"torch.nn.functional.gelu"}:
            return self.lower_unary(node, lambda e: f"(0.5 * ({e}) * (1.0 + math.erf(({e}) * 0.7071067811865476)))")
        if callee == "torch.tensor":
            value = self.static_value(node.args[0])
            return scalar_value(py_literal(value))

        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method == "mean":
                input_value = self.lower_expr(node.func.value)
                dims = self.keyword_value(node, "dim")
                if dims is None and node.args:
                    dims = self.static_value(node.args[0])
                return self.reduce_mean(input_value, list(dims), False)
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
        bias = self.lower_expr(node.args[2]) if len(node.args) > 2 and not (isinstance(node.args[2], ast.Constant) and node.args[2].value is None) else None
        if len(input_value.shape) != 2 or len(weight.shape) != 2:
            raise FunctionalToLambdaError("Only rank-2 linear is supported")
        batch, in_features = input_value.shape
        out_features = weight.shape[0]
        out_shape = (batch, out_features)

        def render(coord: Sequence[str], input_value=input_value, weight=weight, bias=bias, in_features=in_features) -> str:
            b, o = coord
            bias_expr = f" + {bias.render([o])}" if bias is not None else ""
            summation = render_nested_sum(
                [in_features],
                lambda vars_: f"({weight.render([o, vars_[0]])} * {input_value.render([b, vars_[0]])})",
                "li_",
            )
            return f"({summation}{bias_expr})"

        return ValueRef(shape=out_shape, renderer=render)

    def lower_conv(self, node: ast.Call, transpose: bool) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        weight = self.lower_expr(node.args[1])
        bias = self.lower_expr(node.args[2]) if len(node.args) > 2 and not (isinstance(node.args[2], ast.Constant) and node.args[2].value is None) else None
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
                oc_local = oc if groups == 1 else f"({oc} % {out_per_group})"
                group_base = "0" if groups == 1 else f"(({oc} // {out_per_group}) * {in_per_group})"
                reduce_bounds = [in_per_group, *weight.shape[2:]]

                def body(vars_: list[str]) -> str:
                    ic_local = vars_[0]
                    kernels = vars_[1:]
                    input_coords: list[str] = []
                    conds: list[str] = []
                    for axis, (out_axis, k, in_size) in enumerate(zip(spatial, kernels, input_value.shape[2:])):
                        numer = f"({out_axis} + {padding[axis]} - {k} * {dilation[axis]})"
                        coord_expr = f"({numer} // {stride[axis]})"
                        input_coords.append(coord_expr)
                        conds.append(f"({numer} % {stride[axis]} == 0)")
                        conds.append(f"(0 <= {coord_expr} < {in_size})")
                    ic_global = ic_local if groups == 1 else f"({group_base} + {ic_local})"
                    prod_expr = (
                        f"({weight.render([ic_global, oc_local, *kernels])} * "
                        f"{input_value.render([b, ic_global, *input_coords])})"
                    )
                    return f"({prod_expr} if {' and '.join(conds)} else zero)"

                bias_expr = f" + {bias.render([oc])}" if bias is not None else ""
                return f"({render_nested_sum(reduce_bounds, body, 'tconv_')}{bias_expr})"

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
            group_base = "0" if groups == 1 else f"(({oc} // {out_per_group}) * {in_per_group})"

            def body(vars_: list[str]) -> str:
                ic_local = vars_[0]
                kernels = vars_[1:]
                ic_global = ic_local if groups == 1 else f"({group_base} + {ic_local})"
                input_coords = [
                    f"({out_axis} * {stride[axis]} - {padding[axis]} + {k} * {dilation[axis]})"
                    for axis, (out_axis, k) in enumerate(zip(spatial, kernels))
                ]
                return (
                    f"({weight.render([oc, ic_local, *kernels])} * "
                    f"{input_value.render([b, ic_global, *input_coords])})"
                )

            bias_expr = f" + {bias.render([oc])}" if bias is not None else ""
            return f"({render_nested_sum(reduce_bounds, body, 'conv_')}{bias_expr})"

        return ValueRef(shape=out_shape, renderer=render)

    def reduce_mean(self, input_value: ValueRef, dims: Sequence[int], keepdim: bool) -> ValueRef:
        out_shape = reduction_shape(input_value.shape, dims, keepdim)
        bounds = [input_value.shape[dim if dim >= 0 else dim + len(input_value.shape)] for dim in dims]
        count = product(bounds)

        def render(coord: Sequence[str], input_value=input_value, dims=dims, keepdim=keepdim, count=count, bounds=bounds) -> str:
            return (
                f"({render_nested_sum(bounds, lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)), 'mean_')}"
                f" / {count})"
            )

        return ValueRef(shape=out_shape, renderer=render)

    def lower_sum(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        dim = self.keyword_value(node, "dim")
        if dim is None and len(node.args) > 1:
            dim = self.static_value(node.args[1])
        keepdim = bool(self.keyword_value(node, "keepdim", False))
        dims = dim if isinstance(dim, (list, tuple)) else [dim]
        out_shape = reduction_shape(input_value.shape, dims, keepdim)
        bounds = [input_value.shape[d if d >= 0 else d + len(input_value.shape)] for d in dims]

        def render(coord: Sequence[str], input_value=input_value, dims=dims, keepdim=keepdim, bounds=bounds) -> str:
            return render_nested_sum(
                bounds,
                lambda vars_: input_value.render(render_reduction_input_coord(coord, len(input_value.shape), dims, vars_, keepdim)),
                "sum_",
            )

        return ValueRef(shape=out_shape, renderer=render)

    def reduce_extreme(self, input_value: ValueRef, dim: int, keepdim: bool, kind: str) -> ValueRef:
        dims = [dim]
        out_shape = reduction_shape(input_value.shape, dims, keepdim)
        axis = dim if dim >= 0 else dim + len(input_value.shape)
        bound = input_value.shape[axis]

        def render(coord: Sequence[str], input_value=input_value, dim=dim, keepdim=keepdim, bound=bound, kind=kind) -> str:
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
            renderer=lambda coord, lhs=lhs, rhs=rhs, out_shape=out_shape: (
                f"min({lhs.render(broadcast_coord(coord, out_shape, lhs.shape))}, "
                f"{rhs.render(broadcast_coord(coord, out_shape, rhs.shape))})"
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
            expr = input_value.render(coord)
            if min_expr is not None and max_expr is not None:
                return f"min(max({expr}, {min_expr}), {max_expr})"
            if min_expr is not None:
                return f"max({expr}, {min_expr})"
            if max_expr is not None:
                return f"min({expr}, {max_expr})"
            return expr

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_softmax(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        dim = self.keyword_value(node, "dim")
        if dim is None and len(node.args) > 1:
            dim = self.static_value(node.args[1])
        axis = dim if dim >= 0 else dim + len(input_value.shape)
        bound = input_value.shape[axis]

        def render(coord: Sequence[str], input_value=input_value, axis=axis, bound=bound) -> str:
            numer = f"math.exp({input_value.render(coord)})"
            denom = render_nested_sum(
                [bound],
                lambda vars_: f"math.exp({input_value.render(list(coord[:axis]) + [vars_[0]] + list(coord[axis + 1 :]))})",
                "softmax_",
            )
            return f"({numer} / {denom})"

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_group_norm(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        weight = self.lower_expr(node.args[2])
        bias = self.lower_expr(node.args[3])
        eps_name = node.keywords[0].value.id if node.keywords else "group_norm_eps"
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

                return (
                    f"({render_nested_sum([channels_per_group, *spatial], body, 'gnm_')}"
                    f" / {count})"
                )

            def var_for(g_expr: str, mean_expr: str) -> str:
                def body(vars_: list[str]) -> str:
                    channel_expr = f"(({g_expr}) * {channels_per_group} + {vars_[0]})"
                    sample_expr = input_value.render([b, channel_expr, *vars_[1:]])
                    return f"((lambda q: q * q)({sample_expr} - ({mean_expr})))"

                return (
                    f"({render_nested_sum([channels_per_group, *spatial], body, 'gnv_')}"
                    f" / {count})"
                )

            return (
                f"((lambda g: (lambda m: (lambda v: ({weight.render([c])} * ({input_value.render([b, c, *spatial_coord])} - m) / "
                f"math.sqrt(v + {eps_name}) + {bias.render([c])}))({var_for('g', 'm')}))({mean_for('g')}))({c} // {channels_per_group}))"
            )

        return ValueRef(shape=input_value.shape, renderer=render)

    def lower_instance_norm(self, node: ast.Call) -> ValueRef:
        input_value = self.lower_expr(node.args[0])
        if len(input_value.shape) != 4:
            raise FunctionalToLambdaError("Only rank-4 instance_norm is supported")
        use_input_stats = self.keyword_value(node, "use_input_stats", True)
        if not use_input_stats:
            raise FunctionalToLambdaError("Only use_input_stats=True instance_norm is supported")
        eps_name = "instance_norm_eps"
        for keyword in node.keywords:
            if keyword.arg == "eps":
                eps_name = ast.unparse(keyword.value)
        weight_name = "instance_norm_weight"
        bias_name = "instance_norm_bias"
        spatial = input_value.shape[2:]
        count = product(spatial)

        def render(coord: Sequence[str]) -> str:
            b, c = coord[0], coord[1]
            spatial_coord = list(coord[2:])
            mean_expr = (
                f"({render_nested_sum(list(spatial), lambda vars_: input_value.render([b, c, *vars_]), 'inm_')} / {count})"
            )
            var_expr = (
                f"((lambda m: ({render_nested_sum(list(spatial), lambda vars_: f'((lambda q: q * q)({input_value.render([b, c, *vars_])} - m))', 'inv_')} / {count}))({mean_expr}))"
            )
            weight_expr = f"({weight_name}[{c}] if {weight_name} is not None else 1)"
            bias_expr = f"({bias_name}[{c}] if {bias_name} is not None else 0)"
            return (
                f"((lambda m: (lambda v: ({weight_expr} * ({input_value.render([b, c, *spatial_coord])} - m) / "
                f"math.sqrt(v + {eps_name}) + {bias_expr}))({var_expr}))({mean_expr}))"
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
                    f"({out_axis} * {stride[idx]} - {padding[idx]} + {vars_[idx]} * {dilation[idx]})"
                    for idx, out_axis in enumerate(spatial_coord)
                ]
                conds = [
                    f"(0 <= {input_coords[idx]} < {input_value.shape[2 + idx]})"
                    for idx in range(spatial_ndim)
                ]
                return (
                    f"({input_value.render([*base, *input_coords])} if {' and '.join(conds)} else neg_inf)"
                )

            return render_nested_stack_reduce("max", list(kernel), body, "pool_")

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
            return (
                f"({render_nested_sum(list(input_value.shape[2:]), lambda vars_: input_value.render([b, c, *vars_]), 'aap_')} / {count})"
            )

        return ValueRef(shape=out_shape, renderer=render)


def generate_source(path: str | Path, function_name: str = "value_at") -> str:
    spec = load_module_spec(path)
    lowerer = LambdaLowerer(spec, function_name=function_name)
    return lowerer.build()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a coordinate-lambda function from a KernelBench functional model")
    parser.add_argument("input_file", help="Path to the functional model file")
    parser.add_argument("--output", help="Optional output path for the generated module")
    parser.add_argument("--function-name", default="value_at", help="Generated function name")
    args = parser.parse_args(argv)

    source = generate_source(args.input_file, function_name=args.function_name)
    if args.output:
        Path(args.output).write_text(source)
    else:
        print(source, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
