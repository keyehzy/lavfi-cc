"""Safe evaluator for the documented Week 2 lutrgb expression subset.

The same subset serves ``lutyuv``, which is ``vf_lut.c``'s other entry point.
The only difference is the range a component is defined over: ``lutrgb`` runs
every channel over ``0..255``, while ``lutyuv`` gives luma ``16..235`` and each
chroma channel ``16..240``.  That range reaches the expression as ``minval``
and ``maxval`` and, through them, as ``clipval`` and ``negval``, so
:func:`build_lut` takes it as a parameter rather than assuming full range.
"""

from __future__ import annotations

import ast
import math


class ExpressionError(ValueError):
    pass


_VARIABLES = {"val", "clipval", "negval", "minval", "maxval"}
_FUNCTION_ARITY = {"abs": 1, "min": 2, "max": 2, "clip": 3, "pow": 2}


def _validate(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("only numeric constants are supported")
        if not math.isfinite(float(node.value)):
            raise ExpressionError("numeric constants must be finite")
    elif isinstance(node, ast.Name):
        if node.id not in _VARIABLES:
            raise ExpressionError(
                f"variable {node.id!r} is unsupported; allowed variables: "
                + ", ".join(sorted(_VARIABLES))
            )
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate(node.operand)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        _validate(node.left)
        _validate(node.right)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        expected = _FUNCTION_ARITY.get(name)
        if expected is None:
            raise ExpressionError(f"function {name!r} is unsupported")
        if node.keywords or len(node.args) != expected:
            raise ExpressionError(f"function {name!r} requires {expected} positional arguments")
        for argument in node.args:
            _validate(argument)
    else:
        raise ExpressionError(
            "only numbers, supported variables, + - * /, parentheses, and "
            "abs/min/max/clip/pow calls are accepted"
        )


def parse_expression(source: str) -> ast.Expression:
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError) as error:
        detail = getattr(error, "msg", str(error))
        raise ExpressionError(f"invalid expression: {detail}") from error
    _validate(tree)
    return tree


def _evaluate(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, variables)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, variables)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, variables)
        right = _evaluate(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0.0:
            if left == 0.0:
                return math.nan
            sign = math.copysign(1.0, left) * math.copysign(1.0, right)
            return math.copysign(math.inf, sign)
        return left / right
    if isinstance(node, ast.Call):
        name = node.func.id
        arguments = [_evaluate(argument, variables) for argument in node.args]
        if name == "abs":
            return abs(arguments[0])
        if name == "min":
            return min(arguments)
        if name == "max":
            return max(arguments)
        if name == "clip":
            return max(arguments[1], min(arguments[0], arguments[2]))
        if name == "pow":
            try:
                return math.pow(arguments[0], arguments[1])
            except (OverflowError, ValueError):
                return math.nan
    raise AssertionError(f"unvalidated expression node: {ast.dump(node)}")


#: Range every 8-bit ``lutrgb`` component is defined over.
FULL_RANGE = (0, 255)


def build_lut(
    source: str,
    value_range: tuple[int, int] = FULL_RANGE,
    depth: int = 8,
    output_max: int = 255,
) -> tuple[int, ...]:
    """Materialize one component's table over ``[minval, maxval]``.

    ``clipval`` and ``negval`` are derived exactly as ``config_props`` derives
    them: both are clamped into the component's own range, so on a limited-range
    luma channel ``negval`` is ``av_clip(16 + 235 - val, 16, 235)`` rather than
    ``255 - val``.  The final clamp is ``[0, output_max]``, which upstream takes
    from the alpha component's maximum -- the same for every component, and not
    always ``(1 << depth) - 1``; see :func:`lavfi_cc.filters._lut_ranges`.

    Upstream's table always has 65536 entries whatever the depth, because it is
    indexed by a raw ``uint16``.  This one covers the format's own domain and is
    indexed through a clamp; see :mod:`lavfi_cc.interpreter` on why.
    """

    minimum, maximum = value_range
    tree = parse_expression(source)
    values: list[int] = []
    for value in range(1 << depth):
        clipped = max(minimum, min(value, maximum))
        negated = max(minimum, min(minimum + maximum - value, maximum))
        result = _evaluate(
            tree,
            {
                "val": float(value),
                "clipval": float(clipped),
                "negval": float(negated),
                "minval": float(minimum),
                "maxval": float(maximum),
            },
        )
        if not math.isfinite(result):
            raise ExpressionError(f"expression is non-finite for val={value}")
        if result < -(2**31) or result > 2**31 - 1:
            raise ExpressionError(f"expression is outside signed-int range for val={value}")
        quantized = int(result)  # C conversion and Python int both truncate toward zero.
        values.append(max(0, min(output_max, quantized)))
    return tuple(values)
