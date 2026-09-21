# A minimal demonstration of forcing specific structural effects

This experiment compares the original community with two single-interaction modifications under the same environmental forcing. It uses the same focal species at both frequencies, measures whole-community RMS departure separately, and predicts each observable using the specified forcing. There is no maximization over forcing vectors.

## Run and outputs

Requires Julia and CairoMakie, consistent with the other scripts in this repository. Run `julia --startup-file=no forcing_specific.jl`, or `include("forcing_specific.jl")` in a Julia REPL. The script checks the calculations, generates the figure as PNG and vector PDF, and saves all data in its own `outputs` folder.

- `forcing_specific_effects.png` and `.pdf`: the complete practical demonstration.
- `frequency_predictions.csv`: first-order and exact finite-change predictions for both observables across frequency.
- `trajectories.csv`: original and modified simulated trajectories for both species.
- `simulation_metrics.csv`: predicted and measured RMS effects and relative errors.
- `results.txt`: parameters, environment versions, and quantitative findings.

## Why two species suffice

The model is `dx/dt = A x + a b cos(omega t)` with

```
A = [-1  -2
      2  -1]
b = [1, 0]
a = 0.20
```

Species 1 has a positive effect on species 2; species 2 has a negative effect on species 1. These are predator-prey signs, with self-regulation in both species. `x` is displacement from a reference equilibrium in common scaled units. This is an illustrative local linear model, not a fitted ecological population model or a nonlinear model of positive abundances. T is the identity. Baseline eigenvalues are -1 +/- 2i, so the equilibrium is stable.

Environmental forcing always acts on species 1 with the same amplitude and phase. The focal observable is always species 2. Two separate modifications are applied:

- A: increase A[1,2] by 0.20, weakening the negative effect of species 2 on species 1 from -2 to -1.8.
- B: increase A[2,1] by 0.20, strengthening the positive effect of species 1 on species 2 from 2 to 2.2.

These are equal absolute changes, and both happen to be 10 percent of the baseline link magnitude. They are alternative changes to phenomenological coefficients, not a claim that one biological trait can independently alter either coefficient. Both perturbed systems remain stable.

## Analytical prediction and transparent design

Write the common baseline interaction magnitude as k=2, z=1+i*omega, and D=z^2+k^2. Then

```
R = [z  -k; k  z] / D
```

For unit forcing on species 1, the first-order complex response changes per unit interaction change are

```
R P_A R b = [k*z, k^2] / D^2
R P_B R b = [-k*z, z^2] / D^2
```

For the fixed focal species 2, the ratio of predicted RMS departures A/B is

```
k^2 / (1 + omega^2)
```

For the whole community, the ratio of predicted RMS distances A/B is

```
k / sqrt(1 + omega^2)
```

Both ratios cross 1 at `omega = sqrt(k^2-1) = sqrt(3)`. A is predicted to matter more below this frequency, and B above it. This is possible for any k>1 in this family, not just one optimized parameter value. The model, input, focal species, and demonstration frequencies (0.5 and 4) are specified directly; the script performs no search or outcome-dependent selection. The frequencies straddle the known analytical crossover.

This is still an intentionally constructed illustration, not an estimate of the prevalence of reversals in nature. The finite-change crossover can differ from the first-order crossover.

## Match each prediction to its observable

Let `h = R P R b`, `epsilon=0.20`, and `a=0.20`. The first-order periodic difference is `a*epsilon*real(h*exp(i*omega*t))`.

- Focal species RMS departure: `a*epsilon*abs(h[2])/sqrt(2)`.
- Whole-community RMS distance: `a*epsilon*norm(h)/sqrt(2)`.

The latter equals the square root of the time mean of `delta_x1^2 + delta_x2^2`. It is neither the time mean of the instantaneous norm, nor the RMS of total biomass, nor an average per species. Equal species weights are appropriate to the common scaled units used here; different units or ecological priorities require explicit weighting.

Exact finite-change frequency predictions replace `epsilon*h` with `(R_modified-R_original)*b`. These are used for verification and saved in the CSV, but the main figure compares first-order predictions with independent ODE simulations.

## Simulations and figure

All three systems start from zero displacement and are integrated separately using RK4. At least 35 model time units of transient decay are discarded, rounded up to a whole number of forcing cycles. RMS measurements use four complete cycles and trapezoidal time integration. The time-series panels display the first two measured cycles.

- A and B: forcing-specific first-order predictions for the focal species and whole community. Dotted lines mark the demonstration frequencies; the dashed line marks the analytical crossover.
- C and D: original and modified species 2 trajectories under slow and fast forcing. All lines are direct ODE simulations. Lower strips show the differences from the original trajectory, without arbitrary magnification. Both panels use matching vertical limits, including their departure strips. The horizontal axis is real model time, so two cycles span different durations under slow and fast forcing.
- E and F: measured RMS departures (coloured bars) and first-order predictions (white diamonds). The same colour always represents the same modification.

The finite change is intentionally large enough to see its effects while still reasonably approximated by the first-order response. Differences between bars and diamonds are reported rather than hidden. The code also verifies improved approximation for a tenfold smaller interaction change.

## Verification and limits

Checks cover baseline and perturbed stability, single-link normalization, both analytical ratios, finite-difference derivatives, the fixed-input spectral-norm bound, ODE agreement with exact periodic response, both observed ranking reversals, convergence as the interaction change decreases, and time-step refinement.

The result demonstrates a practical consequence within a local linear model: changing the forcing timescale changes which interaction modification produces the larger departure, both for the same focal species and for the whole community. It does not establish nonlinear validity, an empirical intervention effect, or a universal reversal. Equilibrium movement and coupled changes in T are outside this controlled experiment.
