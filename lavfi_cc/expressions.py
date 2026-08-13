"""Safe evaluator for the documented Week 2 lutrgb expression subset."""

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


def build_lut(source: str) -> tuple[int, ...]:
    tree = parse_expression(source)
    values: list[int] = []
    for value in range(256):
        result = _evaluate(
            tree,
            {
                "val": float(value),
                "clipval": float(value),
                "negval": float(255 - value),
                "minval": 0.0,
                "maxval": 255.0,
            },
        )
        if not math.isfinite(result):
            raise ExpressionError(f"expression is non-finite for val={value}")
        if result < -(2**31) or result > 2**31 - 1:
            raise ExpressionError(f"expression is outside signed-int range for val={value}")
        quantized = int(result)  # C conversion and Python int both truncate toward zero.
        values.append(max(0, min(255, quantized)))
    return tuple(values)
