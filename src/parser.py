import re
from collections.abc import Callable
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def parse_equation(
    equation: str,
) -> tuple[str, list[str], sp.Expr, str, Callable[..., Any]]:
    if "=" not in equation:
        raise ValueError("Equation must contain an '=' sign.")
    left, right = (part.strip() for part in equation.split("=", 1))

    match = re.match(r"^([a-zA-Z_]\w*)\s*\(([^)]*)\)$", left)
    if not match:
        raise ValueError(
            "Left side must be a valid function signature. "
            "Expected format: function_name(x, y, ...)"
        )

    function_name = match.group(1)
    raw_args = match.group(2).strip()
    var_names = [arg.strip() for arg in raw_args.split(",") if arg.strip()]

    if not all(var.isidentifier() for var in var_names):
        raise ValueError(f"Invalid variable name in signature: {var_names}")
    if len(var_names) != len(set(var_names)):
        raise ValueError(f"Duplicate variable names in signature: {var_names}")

    symbols = [sp.Symbol(name) for name in var_names]
    local_dict = dict(zip(var_names, symbols, strict=True))

    expression = sp.parse_expr(
        right, local_dict=local_dict, transformations=TRANSFORMATIONS
    )

    # Verifica se a expressão usa variáveis que não foram declaradas em f(...)
    undeclared = expression.free_symbols - set(symbols)
    if undeclared:
        undeclared_names = ", ".join(sorted(str(s) for s in undeclared))
        raise ValueError(f"Expression uses undeclared variable(s): {undeclared_names}")

    numeric_function = sp.lambdify(symbols, expression, modules=["numpy"])

    return function_name, var_names, expression, right, numeric_function
