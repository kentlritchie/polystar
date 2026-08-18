# Physics notes

## From hydrostatic equilibrium to the Lane-Emden equation

A static, spherically symmetric star obeys two structure equations:

**Hydrostatic equilibrium** — the pressure gradient at radius `r` exactly
balances the local gravitational pull of everything interior to it:

```
dP/dr = -G M(r) ρ(r) / r^2
```

**Mass continuity** — the enclosed mass grows by adding successive
spherical shells:

```
dM/dr = 4 π r^2 ρ(r)
```

These two equations alone are under-determined: two unknowns (`P`, `ρ`),
one radius-dependent relation. Something has to close the system. A
**polytropic equation of state** does it with a simple power law:

```
P = K ρ^(1 + 1/n)
```

where `K` is a constant and `n` is the polytropic index. Combining all
three and substituting the dimensionless variables

```
ρ = ρ_c θ(ξ)^n,        r = a ξ,        a^2 = (n+1) K ρ_c^(1/n - 1) / (4 π G)
```

collapses hydrostatic equilibrium + mass continuity + the polytropic
closure into a single second-order ODE for the dimensionless function
`θ(ξ)`:

```
(1/ξ^2) d/dξ [ ξ^2 dθ/dξ ] = -θ^n
θ(0) = 1,   dθ/dξ|_(ξ=0) = 0
```

This is the **Lane-Emden equation**. `θ` tracks the dimensionless
density-potential from center (`θ=1`) outward; the star's surface is the
first radius where `θ` reaches zero, `ξ_1`. For `n < 5` that radius is
finite; `n = 5` is the marginal case with formally infinite radius.

## What the polytropic index means physically

| n | Physical regime |
|---|---|
| 0 | Incompressible, uniform density |
| 1 | Relativistic-degenerate matter / simple toy models |
| 1.5 | Fully convective star (adiabatic monatomic ideal gas) |
| 3 | Eddington standard model (radiation-pressure-dominated envelope) |
| 5 | Marginal case, infinite radius, most centrally concentrated |

Higher `n` means pressure resists compression less strongly as density
rises relative to the center, so mass piles up toward the middle and the
outer envelope thins out — visible directly in `docs/figures/comparison_n.png`.

## Derived dimensionless quantities

```
ρ/ρ_c            = θ^n
M_dimensionless(ξ) = -ξ^2 dθ/dξ
ρ_c / <ρ>         = ξ_1^3 / [3 (-ξ_1^2 θ'(ξ_1))]
```

The last quantity is the **central concentration**: how much denser the
core is than the star's mean density. It grows sharply as `n → 5`
(`plot_central_concentration` in `plotting.py`).

## Beyond the polytropic assumption: he_integrator.py

The Lane-Emden reduction is elegant but assumes `P(ρ)` is a clean power
law everywhere — real stars aren't. `he_integrator.py` integrates the
*same* two structure equations (hydrostatic equilibrium + mass continuity)
without that assumption, closing the system instead with the ideal-gas law

```
ρ = P μ m_H / (k_B T)
```

using the **real** temperature profile `T(r)` from Model S as an external
input. This sidesteps needing a full radiative-transport/energy-generation
model while still being a genuinely different (non-polytropic) physics
path — see `docs/validation.md` for how its output compares to the real
Sun.

## References

- Chandrasekhar, S. (1939), *An Introduction to the Study of Stellar
  Structure*, University of Chicago Press.
- Horedt, G. P. (2004), *Polytropes: Applications in Astrophysics and
  Related Fields*, Kluwer Academic Publishers.
- Christensen-Dalsgaard, J. et al. (1996), "The Current State of Solar
  Modeling," *Science* 272, 1286-1292 (Model S).
