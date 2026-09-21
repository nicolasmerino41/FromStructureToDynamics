# Single link sensitivity across forcing frequencies

This experiment asks why changing one interaction alters a community's response differently at different frequencies. It separates source responsiveness from propagation through the recipient, then checks predictions against independently integrated time-domain dynamics.

## Run

The script follows the repository's Julia and CairoMakie workflow. With CairoMakie installed in your active Julia environment, run from this folder:

```julia
julia --startup-file=no single_link_analysis.jl
```

Or from a Julia REPL:

```julia
include("single_link_analysis.jl")
```

Outputs are always saved under this script's `outputs` directory, regardless of the working directory. The run executes scientific assertions before creating the figures. No random sampling or smoothing is used.

## Model and conventions

We use `T dx/dt = A x + a b cos(omega t)`, where `x` is displacement from a fixed reference equilibrium. `A[i,j]` is the effect of source species `j` on recipient species `i`. The matrix reproduces the illustrative three-module system in Figure5.jl before its display permutation. The main experiment uses `T = I`; additional checks use heterogeneous positive timescales.

The model is illustrative, with no calibrated physical time unit or fitted species. `omega` is angular frequency and the period is `2 pi / omega`. All existing nonzero off-diagonal links are analysed. Absent links are excluded, although the same formula could quantify adding a link.

The forcing direction is fixed at `b = [1,0,0,0,0,0]`. Both original and structurally modified systems receive exactly the same forcing. An interaction change is an equal absolute increment `epsilon` to `A[i,j]`, with fixed T. A positive increment strengthens a positive link and weakens a negative link. Absolute first-order magnitudes are the same for either infinitesimal sign. Equal proportional changes would require multiplication by `abs(A[i,j])`.

The forcing norm is defined on the right-hand side of `T dx/dt`. Physical additive forcing in `dx/dt = J x + f` instead enters the resolvent as `T f`. This distinction matters when T is heterogeneous. The code does not claim fixed-T perturbations are full parameter changes of a nonlinear ecological model.

## Analytical result

Let `R = (i omega T - A)^(-1)` and let P contain a single 1 at `(i,j)`. Then

```
R P R = R[:,i] * R[j,:]
potential sensitivity = norm(R[:,i]) * norm(R[j,:])
fixed-input sensitivity = norm(R[:,i]) * abs((R*b)[j])
```

The row here is a literal row, not a conjugation of its entries. Its Euclidean norm supplies the correct complex induced norm.

`norm(R[j,:])` is the greatest possible amplitude in source j over unit complex forcing vectors. `norm(R[:,i])` is the community response magnitude to an input entering recipient i. The product is exactly the spectral norm of RPR. Potential sensitivity maximizes over input amplitudes and relative phases independently at every frequency and for every link. Those optima need not be a single pulse or a common disturbance.

The fixed-input result uses the actual source response under b. Neither quantity is total biomass: the output metric is the Euclidean norm of the species response vector. For real sinusoidal forcing, the complex response-vector norm divided by sqrt(2) is the time RMS of that vector. A specified biomass observable would require an output weighting vector and a different sensitivity metric.

The factorisation is exact; predicting a finite change via `epsilon RPR` is first order. For this single link, the exact resolvent difference is `epsilon RPR / (1 - epsilon R[j,i])` when the denominator is nonzero. This provides additional context for the finite-change approximation, but the code validates it using direct inverses rather than that identity.

## Figures

`outputs/single_link_mechanism.png` and `.pdf`:

- A: all existing links ranked within each frequency. Rank 1 is highest; ties are assigned their average rank (relative tolerance 1e-10). This shows relative importance, not absolute effect size. Ties in this symmetric example are not treated as biological rank changes.
- B: maximum source responsiveness for the highlighted links.
- C: propagation from each highlighted recipient.
- D: the product of B and C, giving potential link sensitivity.
- E: the same two links under fixed forcing on species 1 (solid), compared with their potential sensitivities (dashed).
- F: relative error of the first-order complex response, compared with the exact perturbed resolvent, for a range of interaction changes. Both amplitude and phase contribute to the error.

Highlighted links are selected by an explicit algorithm: choose the pair with the strongest minimum opposing dominance across two frequencies. At eligible frequencies both links must retain at least 10 percent of their own peak sensitivity, excluding negligible tails. This selection is intentionally illustrative, not an unbiased estimate of how often reversals occur. The program stops if no reversal exists.

`outputs/time_domain_validation.png` and `.pdf`:

Direct RK4 integrations start at zero and discard 40 slowest decay times before showing three cycles. Solid curves show the difference between perturbed and original trajectories. Dashed curves show the first-order periodic prediction. For readability each panel displays the species with the largest exact complex response difference in that comparison; all reported errors use all species.

## Checks and numeric outputs

The run checks stability of the baseline and perturbed systems, the factorisation against direct spectral norms, the fixed-input identity and upper bound, the selected ranking reversal, central finite differences with heterogeneous T, finite-change error reduction, and time-domain agreement with the exact frequency response.

- `profiles.csv`: all links, frequencies, the two factors, potential and fixed-input sensitivity.
- `validation.csv`: finite-change prediction errors for the highlighted links and frequencies.
- `summary.txt`: matrices, forcing vector, selected links, frequencies, and error summaries.
- `time_traces.csv`: displayed time-domain traces and their first-order predictions.

These are tests of the mathematics and implementation in a linear model. They do not demonstrate empirical validity, nonlinear validity, or a universal frequency-dependent ranking reversal. A next scientific extension would use predefined ensembles and report reversals and non-reversals without selecting systems for the outcome.
