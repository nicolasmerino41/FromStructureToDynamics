# Three species with demonstrations at maximum structural effects

The main figure uses a three-species extension, larger interaction modifications, and frequencies chosen at the largest signed difference between modification effects in each direction, restricted to omega >= 0.5. The earlier two-species example is preserved in its separate folder.

## Run

Requires Julia and CairoMakie. Run `julia --startup-file=no peak_effects.jl`, or `include("peak_effects.jl")` in Julia. Outputs are saved beside the script under `outputs`. The script performs scientific checks, saves numerical data, and generates a four-panel main figure and a separate approximation diagnostic.

## Controlled model

The equation is `T dx/dt = A x + a b cos(omega t)`:

```
A = [-1  -2   0
      2  -1   0
      1   1  -1]
T = diag(1, 1, 2)
b = [1, 0, 0]
a = 0.20
```

The original interacting pair remains unchanged. Species 3 receives equal positive effects from species 1 and species 2, responds more slowly, and does not feed back to the pair. It is a downstream receiver of both signals, not a fitted trophic species. This is an explicit idealization: a one-way coupling is used to keep the mechanism transparent.

Species 3 receives `x1+x2` through the filter `1/(1+2*i*omega)`. The two contributions can reinforce or partially cancel depending on their phases. It therefore contributes differently to the consequences of the two interaction modifications. It does not change the focal species 2 trajectory. In particular, a third species is not guaranteed to produce a dramatic focal/community difference.

Two alternatives to the baseline are compared:

- A: A[1,2] increases by 0.6, from -2 to -1.4 (weaker negative effect).
- B: A[2,1] increases by 0.6, from +2 to +2.6 (stronger positive effect).

Both changes are 30 percent of the original interaction magnitude. All three systems receive the same forcing amplitude and phase on species 1. The same focal species 2 is displayed in both time-series panels. T and the reference equilibrium are held fixed. x denotes displacements in common scaled units, not raw biomass.

## Predictions for the actual finite modification

Define `R=(i*omega*T-A)^(-1)`. For each modification P, the exact complex trajectory difference in the linear model is

```
d = a * (R_modified - R_original) * b
```

The focal RMS departure is `abs(d[2])/sqrt(2)`. The community RMS distance is `norm(d)/sqrt(2)`, equivalently `sqrt(mean(delta_x1^2 + delta_x2^2 + delta_x3^2))`. It is not total biomass and not the mean of the instantaneous distance. These expressions use the actual forcing b and the specified output, with no input maximization.

The first-order approximation is `a*epsilon*R*P*R*b`. For single-link changes, the exact expression also obeys `epsilon*R*P*R/(1-epsilon*R[j,i])` before multiplication by forcing amplitude and direction. We compute the finite-change result using direct inverses rather than assuming the derivative remains accurate at 30 percent.

The main figure uses exact finite-change predictions. The approximation diagnostic reports the first-order error, including phase, for 10, 20, and 30 percent changes at the corresponding modification's own peak. A smaller RMS-magnitude error need not imply a small complex-response error, because phase can be wrong.

## Maximum difference selection

The main figure now maximizes the signed ABSOLUTE difference of the two exact community RMS curves: first `D_A(omega)-D_B(omega)`, then `D_B(omega)-D_A(omega)`. This is the vertical gap, not a ratio, not either curve's individual peak, and not the distance between the two modified trajectories. The focal species is unchanged, but frequency selection uses community RMS, so the displayed focal-species contrast need not itself be maximal.

The user-selected comparison range is omega >= 0.5, so both examples remain periodic. A's maximum signed advantage is at the lower boundary (0.5); B's is an interior optimum. These are restricted-range maxima, not unrestricted global maxima. Without this constraint A's largest advantage occurs at zero.

The algorithm scans 0.5 to 20, refines all resolved local maxima, includes endpoints, and verifies convergence with a doubled-resolution grid. An upper bound proves the omitted high-frequency tail cannot beat the selected positive gap. Dense-grid tests check both signed objectives and verify that both selected frequencies satisfy the lower bound. The parameter `MIN_FREQUENCY` controls this scientifically explicit restriction.

## Figure and simulation

Panel B marks the largest vertical gap in each direction within omega >= 0.5. Panels C and D both show species 2 under periodic forcing, at those two frequencies. The forcing direction, amplitude, and phase are identical; the focal species and vertical scales are unchanged.

Both time-series panels display two complete cycles after transients. RMS measurements use four complete cycles. Independent RK4 simulations start from zero, discard at least 35 slowest decay times, and are checked against exact finite-change frequency predictions. There is no constant-forcing example in this figure.

The amplitude diagnostic and parameter sweep remain explicitly labelled auxiliary analyses evaluated at each modification's own response peak. They do not choose the main figure's demonstration frequencies.

## Exploration and reproducibility

The exploration first considered a third species receiving only species 1. That mostly adds a common contribution to the two effects and does not provide the desired distinct community-level signal. Two equal incoming effects were then used so that the downstream species combines different phase relationships. This is a purposeful explanatory construction, not a random or unbiased ecological sample.

The chosen parameters (incoming strengths both 1, third timescale 2) are simple central values of a documented 3-by-3 sensitivity sweep: incoming strengths 0.5, 1, 1.5 and timescales 1, 2, 3. All cases, including zero-frequency maxima, are saved. The sweep quantifies how the outcome depends on those assumptions; it is not used to select an isolated best-looking case.

Outputs:

- `peak_effects.png` and `.pdf`: main four-panel figure.
- `amplitude_diagnostic.png` and `.pdf`: effect size and first-order error at 10, 20, and 30 percent changes.
- `profiles.csv`: exact and first-order focal and community profiles.
- `amplitude_comparison.csv`: numerical amplitude sweep, with separate complex and RMS errors.
- `parameter_sweep.csv`: all nine downstream coupling/timescale combinations.
- `trajectories.csv`: all simulated species trajectories.
- `summary.txt`: settings, frequencies, RMS effects, and approximation errors.

## Scope

This illustrates practical structural effects within a local linear model. Increasing the interaction modification does not validate nonlinear ecological behaviour, equilibrium movement, positivity of raw abundances, or an empirical intervention. Exact finite-change prediction means exact for this linear model only. The downstream species changes the community metric but cannot change species 2 because it has no feedback. Stronger differentiation is an outcome to inspect, not something guaranteed merely by adding a species.
