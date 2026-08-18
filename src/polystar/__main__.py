"""
Command-line interface (SRS section 9.2):

    python -m polystar solve --n 1.5 --plot --export model_n1p5.csv
    python -m polystar validate
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .diagnostics import validate_analytic, validate_reference
from .lane_emden import LaneEmdenError, solve_lane_emden
from .models import SolverConfig
from .plotting import plot_theta


def _solution_to_dataframe(solution) -> pd.DataFrame:
    end = solution.xi_surface if solution.xi_surface is not None else solution.xi[-1]
    return pd.DataFrame({
        "xi": solution.xi,
        "theta": solution.theta,
        "dtheta_dxi": solution.dtheta_dxi,
        "rho_over_rho_c": solution.rho_normalized,
        "m_dimensionless": solution.mass_dimensionless,
        "r_over_R": solution.xi / end,
        "m_over_M": solution.mass_dimensionless / (solution.surface_mass_coefficient or solution.mass_dimensionless[-1]),
    })


def _cmd_solve(args: argparse.Namespace) -> int:
    config = SolverConfig(xi_max=args.xi_max, rtol=args.rtol, atol=args.atol)
    try:
        solution = solve_lane_emden(args.n, config=config)
    except LaneEmdenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"n = {solution.n}")
    if solution.xi_surface is not None:
        print(f"  xi_1 (surface)              = {solution.xi_surface:.6f}")
        print(f"  -xi_1^2 theta'(xi_1)        = {solution.surface_mass_coefficient:.6f}")
        print(f"  rho_c / <rho>               = {solution.central_concentration:.6f}")
    else:
        print("  no finite surface found within xi_max"
              f" = {args.xi_max} ({solution.metadata.get('warning', '')})")

    if args.export:
        _solution_to_dataframe(solution).to_csv(args.export, index=False)
        print(f"  exported profile to {args.export}")

    if args.plot:
        fig = plot_theta(solution)
        out = args.plot if isinstance(args.plot, str) else f"lane_emden_n{args.n}.png"
        fig.savefig(out, dpi=150)
        print(f"  saved figure to {out}")

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    ok = True
    for n in (0.0, 1.0):
        result = validate_analytic(n)
        status = "PASS" if result.passed else "FAIL"
        ok &= result.passed
        print(f"[{status}] {result.label}: {result.details}")
    for n in (1.5, 3.0):
        result = validate_reference(n)
        status = "PASS" if result.passed else "FAIL"
        ok &= result.passed
        print(f"[{status}] {result.label}: {result.details}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polystar")
    sub = parser.add_subparsers(dest="command", required=True)

    solve_p = sub.add_parser("solve", help="Solve the Lane-Emden equation for a given n")
    solve_p.add_argument("--n", type=float, required=True, help="Polytropic index")
    solve_p.add_argument("--xi-max", type=float, default=50.0)
    solve_p.add_argument("--rtol", type=float, default=1e-10)
    solve_p.add_argument("--atol", type=float, default=1e-12)
    solve_p.add_argument("--plot", nargs="?", const=True, default=False,
                          help="Save a theta(xi) figure; optionally give a filename")
    solve_p.add_argument("--export", type=str, default=None, help="Export profile to CSV")
    solve_p.set_defaults(func=_cmd_solve)

    validate_p = sub.add_parser("validate", help="Run the analytic/reference validation suite")
    validate_p.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
