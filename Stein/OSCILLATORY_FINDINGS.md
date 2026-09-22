# Stein: final bounded oscillatory test

## Decision

The selected frequency-dependent reversal is reproducible in the fitted nonlinear model, but it is not a clean illustration of the direct structural contribution at a fixed reference state. It depends strongly on the equilibrium and input-coupling changes that accompany a biological interaction-coefficient change. Given the time constraint, stop developing this selected Stein example as the main validation of the current framework. Retain it as a possible application of an extended biological-parameter sensitivity calculation.

This test does not establish that every possible Stein application fails. It tests the two links selected from the earlier screen. These are model predictions and internal numerical checks, not validation against observed responses to modified interactions.

## What was tested

- Published Stein gLV parameters, on the feasible, fully stable seven-group pathogen-present resident support.
- Nonnegative scaled antibiotic exposure: u(t) = 0.001 + 0.0002 cos(2*pi*t/period), ranging from 0.0008 to 0.0012. These are hypothetical scaled inputs, not clinical doses.
- Periods of 2 and 30 days, selected from the previous screen before this test.
- Separate 0.1% increases in the magnitude of the Blautia-to-C. difficile and C. difficile-to-Akkermansia coefficients. An arrow means the first group's effect on the second group.
- Recomputed positive equilibria and full 11-group stability, including invasion directions, for each modification.
- Nonlinear trajectories after transients. Each trajectory is centred on its own background-exposure equilibrium, so the plotted differences isolate oscillatory responses rather than baseline abundance shifts.

The code is `oscillatory_test.jl`; run `julia --startup-file=no oscillatory_test.jl` after preparing the source parameters. Results are in `outputs/oscillatory_test/`.

## Results

| Period | Larger C. difficile response departure | Ratio to the other modification |
|---|---|---:|
| 2 days | Blautia-to-C. difficile | 4.21 |
| 30 days | C. difficile-to-Akkermansia | 2.82 |

These ratios use exact local response differences at finite coefficient changes. Nonlinear-versus-local community response error is below 0.09% in all four cases. The full first-order coefficient prediction differs from the exact local result by less than 0.46% for either the community or focal output. The numerical checks, including time-step refinement, pass.

The reversal also persists in exact local calculations across all nine combinations of background exposure (0.0005, 0.001, 0.002) and coefficient changes (0.05%, 0.1%, 0.25%). All tested states are stable. This robustness grid is a local-model calculation; nonlinear simulations were performed for the four main examples.

However, the direct structural contribution ranks Blautia-to-C. difficile above C. difficile-to-Akkermansia at both selected periods, throughout that robustness grid. Using only that contribution produces 90–126% relative error in the focal complex response difference for the four main examples. The error includes phase as well as amplitude.

Absolute oscillatory departures are small under these deliberately weak inputs and coefficient changes. These checks establish numerical consistency, not a substantial ecological or clinical effect.

## Why the distinction matters

For the resident interaction matrix B and equilibrium n under background exposure, J = diag(n) B and b = n .* epsilon. If one biological interaction coefficient changes in direction E, then

```
dn = -B^(-1) E n
dJ_direct = diag(n) E
dJ_equilibrium = diag(dn) B
db = dn .* epsilon
H = (i*omega*I - J)^(-1)
y = H b
dy = H dJ_direct y + H dJ_equilibrium y + H db
```

The first term is the direct interaction contribution with the reference abundance and forcing input held fixed. The second captures how shifted abundances change the Jacobian. The third captures how shifted abundances change the absolute effect of per-capita antibiotic forcing. All three are induced by the interaction modification, but only the first matches that particular fixed-reference structural comparison.

The three contributions are complex vectors: their phases can reinforce or cancel. Their RMS magnitudes cannot be added as ordinary percentage shares. For the focal output, signed projections onto the total derivative give:

| Link | Period | Direct | Equilibrium/Jacobian | Input coupling |
|---|---:|---:|---:|---:|
| Blautia-to-C. difficile | 2 days | 13.30% | -7.87% | 94.57% |
| Blautia-to-C. difficile | 30 days | -25.24% | 129.91% | -4.67% |
| C. difficile-to-Akkermansia | 2 days | -0.081% | -67.12% | 167.20% |
| C. difficile-to-Akkermansia | 30 days | -0.048% | 86.14% | 13.91% |

Each row sums to approximately 100%. Negative entries indicate opposition to the total response in this projection; values above 100% reflect cancellation. These are diagnostic projections, not independent fractions of variance.

## Figure and manuscript implications

`oscillatory_decomposition.png` and `.svg` show the full derivative (A), direct contribution alone (B), and nonlinear centred-response differences at the two periods (bottom). The top panels have different vertical scales. Black dashed traces show exact local response differences and nearly coincide with the nonlinear curves as expected under weak forcing.

The framework's resolvent remains useful in the full calculation; this result does not invalidate it. What fails is the proposed clean interpretation of the selected reversal as a direct fixed-reference interaction effect. Treating the full biological change correctly requires the additional equilibrium and input terms above, or an explicit mapping into the manuscript's A/T parametrization.

For a revision with limited development time, retain the revised illustrative Figure 3 as the controlled demonstration. Use Stein only if the paper can accommodate this broader parameter-sensitivity application and label it as a published-model application. Do not present the synthetic sinusoidal trajectories as new empirical validation. No parameter uncertainty or independent intervention observations were assessed here.
