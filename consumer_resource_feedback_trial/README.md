# Consumer–resource feedback under a changing temporal environment

This first trial asks **when a reduction in consumer attack rate increases or decreases resource variability**, and whether the answer can be expressed through biological quantities rather than a particular matrix. It includes a nonlinear consumer–resource model, exact local frequency responses, stationary responses to persistent environmental noise, structural derivatives, four PNG figures, and numerical checks.

Start with [RESULTS.md](RESULTS.md) for the interpretation and [outputs/03_environmental_persistence.png](outputs/03_environmental_persistence.png) for the clearest ecological result. These are exploratory results, not a rewritten manuscript or an empirical application.

## Run

Julia 1.12.5 and CairoMakie 0.15.8 were used. All other imported packages are Julia standard libraries. With CairoMakie already installed:

```julia
include("trial.jl")
```

Or run `julia --startup-file=no trial.jl` from this directory. For a separate Julia environment, the included Project.toml declares CairoMakie; instantiate that environment once, then run `julia --project=. --startup-file=no trial.jl`.

The script writes only PNG figure files, plus CSV numerical results and a text verification report, under `outputs/`. It uses no downloaded data and performs no random sampling. Existing outputs in this folder are overwritten on rerun. It does not change other experiments or manuscripts.

## Biological model and intervention

The resource R grows logistically and a consumer C feeds on it:

```
dR/dt = r R (1 - R/K) (1 + u(t)) - a R C / (1 + a h R)
dC/dt = s [e a R / (1 + a h R) - m] C
```

- r: resource intrinsic growth rate; K: carrying capacity.
- a: attack rate; e: conversion efficiency; m: consumer mortality parameter.
- s: relative consumer response-speed multiplier, applied to both consumer gains and losses. This is a phenomenological controlled comparison; it is not an independently energy-conserving manipulation of physiology.
- h: handling-time parameter. The primary model has h=0 (linear consumption). A small robustness comparison uses h>0 (saturating consumption).
- u(t): fractional variation in resource intrinsic growth. Neither K nor consumer mortality is environmentally forced in this trial.

Every intervention reduces a by 10%, changing resource loss and consumer gain consistently. Coexistence equilibria, Jacobians and environmental coupling are recomputed after the change. We do not freeze abundances or change only one matrix entry. Resource and consumer means are reported separately from variance.

Default parameters are r=K=1, e=0.5, m=0.2, s=1, a=1, h=0. Thus the unforced resource equilibrium is R*=0.4, and C*=0.6. Following the attack-rate reduction, R*=0.444444 and C*=0.617284. The consumer can become more abundant despite lower attack rate because the resource becomes more abundant. This is a consequence of the model, not an imposed response.

The main dimensionless quantities are:

```
q = R*/K                 resource abundance relative to carrying capacity
gamma = s m / r         consumer demographic speed relative to resource growth
Omega = omega / r      environmental angular frequency
theta = tau r          environmental persistence relative to resource growth
```

Smaller q means greater resource suppression relative to K in this model. It is not a universal empirical index of top-down control. Changing gamma at fixed q preserves the equilibrium but changes feedback dynamics. Changing q in the maps changes the baseline attack rate while holding r, K, e and m fixed.

## Environmental forcing and outcomes

### Periodic forcing

u(t)=0.02 cos(omega t). Period is 2π/omega. Linearization around each model's own equilibrium gives:

```
dx/dt = J x + b u(t)
b = [r R* (1-R*/K), 0]
y(omega) = (i omega I - J)^(-1) b
Var(x_i) = 0.02^2 |y_i|^2 / 2
```

Figures show finite-change differences between the two local models, not a norm maximized over forcing directions. The main outcome is 100(Var_modified/Var_original - 1). CV² is Var/(equilibrium abundance)². This second outcome checks whether population-size changes alter the interpretation.

The two time-series frequencies maximize the positive and negative *absolute variance differences* on the stored grid, restricted to baseline variance at least 10% of its maximum. They are illustrative selected cases, not a prevalence estimate or exact continuous optima. There is no constant forcing example. Nonlinear RK4 simulations independently check the local predictions under weak periodic forcing; the full biological attack-rate modification is retained.

### Persistent environmental noise

u is zero-mean Ornstein–Uhlenbeck forcing with covariance sigma² exp(-|lag|/tau). Environmental variance sigma² stays fixed as tau changes. The spectrum is 2 sigma² tau/(1+omega² tau²). This is a low-pass environmental spectrum, not a narrow band centred at 1/tau. Its rapid limit at fixed variance is not finite-intensity white noise.

The stationary covariance is solved using a three-state Lyapunov equation for resource, consumer and environmental forcing. No stochastic trajectory simulation is needed for this exact local calculation. For the absolute-scale SD panel, sigma=0.02. Tables give variance per unit environmental variance; effect percentages do not depend on sigma in the linear regime. Nonlinear stochastic responses have not been tested.

### Structural derivative

For H=(i omega I-J)^(-1), the full response derivative is:

```
dy/da = H (dJ/da) H b + H (db/da)
d Var(x_i)/da = 0.02^2 Re(conj(y_i) dy_i/da)
```

The first term contains the structured resolvent derivative. The Jacobian derivative is separated into a direct parameter derivative at fixed abundance and an equilibrium-mediated derivative. These are diagnostic contributions, not separately implemented interventions. The environmental-coupling derivative is essential because growth forcing acts on a changed population. These contributions are added with their signs; norms of the contributions would not predict the sign of the variance change.

## Figures

1. **01_periodic_response_map.png** — Signed resource-variance and CV² changes across environmental frequency and consumer response speed; baseline response profiles show absolute size.
2. **02_mechanism_and_time_series.png** — Nonlinear time series at two selected frequencies, changes in ecological quantities, and a first-order decomposition checked against the exact finite change.
3. **03_environmental_persistence.png** — Resource suppression and environmental persistence jointly determine the sign of the absolute-variance effect. Relative variability can tell a different story.
4. **04_robustness_and_consumer.png** — Consumer response speed, modification size, saturating consumption and consumer variance. Saturation weakens the apparent generality of the baseline reversal.

Heatmap colours encode percentage changes; black contours indicate zero change. The plotted colour ranges saturate at ±60% in figure 1 and ±35% in figure 3. Quantitative values remain in the CSVs. Times are in arbitrary model units; no species or empirical time calibration is claimed.

## Verification and boundaries

The script checks coexistence and local stability for every plotted model, analytic harmonic responses against the resolvent, analytic OU variance against Lyapunov covariance, Lyapunov residuals, covariance against independent spectral quadrature, the full structural derivative against finite differences, and an analytical dimensionless sign relationship. Four nonlinear periodic cases check local resource and consumer variances; one modified case also receives time-step refinement. All assertions must pass.

The primary model has no consumer self-limitation and exhibits perfect steady-state resource adaptation to changes in r. This matters for the very slow forcing limit. Its resource–consumer harmonic phase lag is exactly 90 degrees, so a change in phase lag alone cannot explain the results. Saturating consumption still lacks consumer self-limitation. No third species, extinction, strong environmental forcing, empirical validation, or general food-web survey is included. No novelty claim is made without a literature assessment.

See `outputs/checks_and_findings.txt` for numerical errors and selected effect sizes. All CSVs are generated directly by `trial.jl`; they are not required as inputs for rerunning.
