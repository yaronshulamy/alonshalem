"""Symbolic Z-transform solver for:

    y[n] - a**2 * y[n-2] = 1/2 * x[n]

Assumes zero initial conditions and a causal system.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ZTransformSolution:
    """Container for symbolic solution artifacts."""

    equation_in_z: sp.Equality
    transfer_function: sp.Expr
    poles: tuple[sp.Expr, ...]
    roc_causal: sp.StrictGreaterThan
    stability_condition: sp.StrictLessThan
    impulse_response: sp.Expr


def solve_question_system(
    a: sp.Symbol | sp.Expr,
    z: sp.Symbol | None = None,
    n: sp.Symbol | None = None,
) -> ZTransformSolution:
    """Solve the system in the Z-domain using SymPy symbols.

    Parameters
    ----------
    a : sympy.Symbol | sympy.Expr
        System parameter in y[n] - a**2*y[n-2] = 1/2*x[n].
    z : sympy.Symbol | None
        Z-domain variable. If omitted, a new symbol "z" is created.
    n : sympy.Symbol | None
        Time index for impulse response. If omitted, a nonnegative integer
        symbol "n" is created.
    """
    z = z or sp.symbols("z")
    n = n or sp.symbols("n", integer=True, nonnegative=True)

    Xz, Yz = sp.symbols("Xz Yz")

    # Z-transform with zero initial conditions:
    # Z{y[n-2]} = z**(-2) * Y(z)
    equation_in_z = sp.Eq(Yz - a**2 * z**-2 * Yz, sp.Rational(1, 2) * Xz)
    yz_solution = sp.solve(equation_in_z, Yz)[0]
    transfer_function = sp.together(sp.simplify(yz_solution / Xz))

    denominator = sp.denom(transfer_function)
    poles = tuple(sp.solve(sp.Eq(denominator, 0), z))

    # For causal h[n], ROC is outside outermost pole.
    roc_causal = sp.Abs(z) > sp.Abs(a)
    stability_condition = sp.Abs(a) < 1

    # h[n] = (1/2) * sum_{k=0..inf} (a^2)^k * delta[n-2k]
    k = sp.symbols("k", integer=True, nonnegative=True)
    impulse_response = sp.Sum(
        sp.Rational(1, 2) * (a**2) ** k * sp.KroneckerDelta(n, 2 * k), (k, 0, sp.oo)
    )

    return ZTransformSolution(
        equation_in_z=equation_in_z,
        transfer_function=transfer_function,
        poles=poles,
        roc_causal=roc_causal,
        stability_condition=stability_condition,
        impulse_response=impulse_response,
    )


def main() -> None:
    """Run a symbolic demonstration for the given question."""
    a, z = sp.symbols("a z", real=True)
    n = sp.symbols("n", integer=True, nonnegative=True)

    solution = solve_question_system(a=a, z=z, n=n)

    print("System: y[n] - a^2 y[n-2] = 1/2 x[n]")
    print(f"Z-domain equation: {solution.equation_in_z}")
    print(f"H(z) = Y(z)/X(z) = {solution.transfer_function}")
    print(f"Poles: {solution.poles}")
    print(f"Causal ROC: {solution.roc_causal}")
    print(f"Stability condition: {solution.stability_condition}")
    print(f"Impulse response h[n]: {solution.impulse_response}")


if __name__ == "__main__":
    main()
