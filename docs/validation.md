# Validation

This project treats validation as a deliverable, not an afterthought
(§11 of the original requirements spec). Everything below is reproducible
with `python -m polystar validate` or `pytest`.

## Analytic cases (closed-form)

| n | Analytic θ(ξ) | Analytic ξ₁ | Computed ξ₁ | Relative error |
|---|---|---|---|---|
| 0 | 1 - ξ²/6 | √6 ≈ 2.449490 | 2.449490 | ~4e-16 |
| 1 | sin(ξ)/ξ | π ≈ 3.141593 | 3.141593 | ~5e-12 |

Both cases also check the maximum absolute error in `θ(ξ)` itself across
the whole interior, not just at the surface (`diagnostics.validate_analytic`).

## Reference cases (tabulated, no closed form)

Values from Horedt (2004) / consistent with Chandrasekhar (1939):

| n | Reference ξ₁ | Computed ξ₁ | Reference mass coeff. | Computed mass coeff. |
|---|---|---|---|---|
| 1.5 | 3.65375 | 3.653754 | 2.71406 | 2.714055 |
| 3.0 | 6.89685 | 6.896849 | 2.01824 | 2.018236 |

Both agree with the tabulated constants to better than 1e-6 relative error
— well inside the documented 1e-3 tolerance used by `validate_reference`.

## Structural checks (all n)

- **Lane-Emden residual**: re-evaluating
  `(1/ξ²) d/dξ[ξ² dθ/dξ] + θ^n` from the solver's own dense output (finite
  differences, away from both endpoints, where the `1/ξ²` factor and
  `θ→0` respectively make finite differences ill-conditioned) stays below
  1e-3 in the interior.
- **Mass monotonicity**: the dimensionless enclosed mass
  `-ξ² dθ/dξ` is non-decreasing outward for every tested `n`.
- **No NaNs approaching n=5**: `n=4.9` integrates cleanly to a finite
  surface with an enlarged `xi_max`; `n=5.0` correctly reports no surface
  found rather than a spurious one or a crash.

## Real-data comparison (not a validation of the polytropic solver itself)

The HE integrator's `mu_mode="model_s"` case — where μ(r) is derived from
Model S's own P, ρ, T — reproduces Model S's density and pressure to
better than 0.1% through the interior. This is expected and mainly a
correctness check on `he_integrator.py`'s ODE integration, since that μ(r)
was constructed to make the ideal-gas closure self-consistent with the
real data it's being compared against.

The `mu_mode="constant"` case (μ ≈ 0.62, independent of Model S) is the
scientifically interesting comparison: it tracks the real Sun to within a
few tens of percent through most of the interior, then visibly diverges in
the outer envelope — exactly where partial ionization changes the real
mean molecular weight the most. That divergence is physics content, not an
error to be tuned away.

## Known non-goals of this validation suite

- No convergence proof beyond spot-checking two tolerance settings and two
  choices of `ξ_0` (`docs/numerical_methods.md`); a full Richardson
  extrapolation study is future work.
- The reference values for n=1.5/n=3 come from tabulated literature
  constants, not an independently re-derived high-precision integration —
  standard practice for this kind of check, but worth being explicit about.
