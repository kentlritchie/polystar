# Numerical methods

## The ξ = 0 singularity

The Lane-Emden equation, written as a first-order system
(`y0 = θ`, `y1 = dθ/dξ`),

```
dy0/dξ = y1
dy1/dξ = -θ^n - 2 y1 / ξ
```

has a `1/ξ` term that is literally undefined at `ξ = 0` — the exact point
we need a boundary condition. `lane_emden.py` sidesteps this rather than
regularizing it numerically: integration starts at a small `ξ_0 > 0`
(`SolverConfig.xi_0`, default `1e-6`), with the initial condition supplied
by the **central series expansion** instead of the raw boundary values:

```
θ(ξ)      ≈ 1 - ξ²/6 + n ξ⁴/120
dθ/dξ(ξ)  ≈ -ξ/3 + n ξ³/30
```

This is the standard local-series treatment (Chandrasekhar 1939) — accurate
to `O(ξ_0^4)` in `θ` itself, which at `ξ_0 = 1e-6` is far below solver
tolerance.

## Integration and surface detection

`scipy.integrate.solve_ivp` (default method `RK45`, `rtol=1e-10`,
`atol=1e-12`) integrates outward from `ξ_0`. Rather than scanning the
output for a sign change after the fact (which would only be as precise as
the sample spacing), the solver uses `solve_ivp`'s **event** mechanism: a
terminal, downward-crossing event on `θ = 0` stops integration exactly at
the root, to solver tolerance, and returns that `ξ` directly as
`xi_surface`. `dense_output=True` then lets the code resample the
already-computed trajectory onto a uniform grid (`n_dense` points) for
smooth plotting, independent of wherever the adaptive stepper actually
placed its internal steps.

## Guarding fractional powers of a negative base

For non-integer `n`, `θ^n` is undefined for `θ < 0` (complex-valued). Two
guards handle this:

1. Inside the ODE right-hand side, `θ` is clamped to `max(θ, 0)` before
   being raised to the `n`-th power — this only matters in the fraction of
   a step just before the surface event fires and terminates integration.
2. After integration, the dense-output array is clamped the same way
   before computing `ρ/ρ_c = θ^n`, since dense-output interpolation can
   produce a tiny negative overshoot right at the surface.

## n approaching (or past) 5

`n = 5` is the marginal case: the analytic solution
`θ = 1/√(1 + ξ²/3)` never reaches zero, so the star's radius is formally
infinite. Numerically, this means the surface-crossing event simply never
fires within any finite `ξ_max`. Rather than treating that as failure, the
solver reports `xi_surface = None` and records a warning in
`solution.metadata`, so callers can distinguish "the model has no finite
surface" from "the ODE integration crashed." `SolverConfig.xi_max`
(default `50`) can be raised for `n` close to but below `5`, where the
surface exists but lies far out.

## Why not just check tolerances once?

`NR-006`/`VAL-007` ask for a convergence check, not a single tolerance
choice taken on faith. In practice: tightening `rtol`/`atol` by another two
orders of magnitude, or shrinking `xi_0` by a factor of 10, changes the
computed `ξ_1` for n=1.5 by less than 1e-6 relative — i.e. the defaults are
already well past the point of diminishing returns for this equation. See
`diagnostics.validate_analytic` / `validate_reference` for the checks that
exercise this in the test suite.
