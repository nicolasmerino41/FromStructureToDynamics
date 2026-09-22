# Stein empirical-model screening

This is a first screening of the original Stein et al. (2013) fitted microbiome model for an interaction-structure application. It checks feasibility before interpreting response sensitivities. It does not refit the model or validate interaction changes against new experimental observations.

## Reproduce

From this directory, run:

```
python prepare_data.py
julia --startup-file=no screen_stein.jl
```

Dependencies: Python numpy/openpyxl and Julia CairoMakie (tested runtime versions are written to outputs/summary.txt). The scripts preserve Data/. The original workbook and paper are downloaded from PLOS; processed/source_inventory.json records the workbook URL, hash, units and exact cell ranges. The workbook reader warning about unsupported Excel extensions does not alter the source file, which is never saved by the reader.

## Reference state and model

The model is dN/dt = N .* (r + A*N + epsilon*u(t)), with time in days. N represents the source's scaled DNA-density proxy, not biomass. A[i,j] is the fitted effect of group j on group i. Some groups are taxonomically aggregated, including an Other category.

All 2,047 nonempty resident supports are screened for positive abundance and stability of the full 11-group Jacobian, including invasion by absent groups. The selected state is the unique fully stable support containing C. difficile. The all-11 coexistence equilibrium is infeasible. This is a published-model equilibrium interpretation, not a claim that the exact seven-group state was experimentally observed.

For the resident submatrix B, n = -B\r, J=diag(n)*B and b=n.*epsilon. Fitted antibiotic susceptibilities determine the input direction. The source antibiotic variable is scaled; it cannot be read as a human dose or intestinal drug concentration. The model does not predict toxin production or clinical infection severity.

## Sensitivity calculation

For a unit change E to B[i,j], within a fixed resident support:

```
dn = -B^(-1)*E*n
dJ = diag(dn)*B + diag(n)*E
db = dn .* epsilon
H(omega) = (i*omega*I-J)^(-1)
dy = H*(dJ*H*b + db)
```

For comparison across links, multiply dy by 0.01*abs(B[i,j]). This compares equal fractional coefficient changes; it is not a measurement-uncertainty estimate. The equal-absolute-coefficient community derivative is also exported. Nonzero off-diagonal resident links only are ranked. Frequency curves use periods 1–180 days and omega=2*pi/period; they represent response per unit sinusoidal input amplitude. No sinusoidal antibiotic exposure is claimed to have been measured or physically administered.

For community output, sensitivity is norm(dy)/sqrt(2); for C. difficile, abs(dy[focal])/sqrt(2). Both are forcing-specific; neither is a maximum over arbitrary input directions. Abundance coordinates are kept in the source's common density units; rare and common groups are not equalized by relative-abundance normalization.

## Finite pulses

Pulse durations are fixed in advance at 0.25, 0.5, 1, 2, 4, 7, 14 and 28 days. Every rectangular pulse has the same integrated scaled exposure, 0.001 days, so its height is 0.001/duration. These are hypothetical weak, nonnegative disturbances.

The linear response and its coefficient derivative are propagated by matrix exponentials of the augmented system:

```
dx/dt = J*x + b*u(t)
dz/dt = J*z + dJ*x + db*u(t)
```

Here z is the derivative of the centred response with respect to one coefficient. This is the time-domain counterpart of the frequency derivative above. Piecewise-constant inputs are propagated exactly on the output grid. Ranking integrates the squared response difference over the same 365-day observation window for every pulse. Community RMS is sqrt(mean_over_time(sum_over_groups(D_i(t)^2))); it is the time RMS Euclidean distance, not total abundance and not the mean across groups. Pathogen RMS uses only C. difficile. A long shared window includes the slow relaxation; it dilutes displayed RMS magnitudes equally across durations.

## Nonlinear verification

At predefined durations of 0.5 and 14 days, select each duration's highest pathogen-sensitivity link and deduplicate. Test each selected link at both durations, using 0.1%, 0.5%, 1%, 5% and 10% increases in coefficient magnitude. The two smaller changes were added after the initial screen found that 5% and 10% changes failed equilibrium checks and a 1% change already had appreciable derivative error. Selection never requires different winning links or a ranking reversal. Time panels show the 1% change. A bisection locates the selected link's first feasibility/stability boundary within the tested 0–10% interval.

Recompute each modified equilibrium and full-system stability; infeasible or unstable cases are recorded and excluded from response comparison. Integrate nonlinear gLV trajectories with RK4, dt=0.05 days, exactly aligning pulse discontinuities. Both original and modified models start at their own equilibria. Compare the centred response difference against (a) the difference between the two recomputed local responses, and (b) the first-order coefficient derivative. Baseline equilibrium shifts are exported separately. Also report the error produced by freezing equilibrium and input coupling to clarify the consequence of that simplification.

Checks cover analytical frequency derivatives against central differences at four periods for all links, all +/-1% stability checks, nonlinear integration refinement, RMS quadrature refinement, the zero-frequency identity, and nonlinear/local agreement at the chosen small input amplitude. Exact finite-change local rankings at 0.1% and 1% are also compared with derivative rankings across every frequency and for both illustration pulses, for all 42 links.

## Outputs and decision

* 01_sensitivity_screen.png/svg: separate community/pathogen profiles, rank changes and finite-pulse priorities. The rank heatmap deliberately shows the 12 links with greatest rank ranges; full tables contain all links.
* 02_pulse_checks.png/svg: same focal group in both time panels, separate community departure and approximation errors.
* frequency_sensitivity.csv and pulse_sensitivity.csv: full rankings and values.
* equilibria.csv, nonlinear_checks.csv and pulse_trace_*.csv: reference states and verification.
* summary.txt: screening metrics and runtime versions.
* FINDINGS.md: interpretation of the completed screening.

The aim is to determine whether the model produces an informative, interpretable application. Stable rankings are valid findings; large reversals are not a success condition. The original data are sparse and irregular over weeks; periods extending to 180 days and the 365-day response window are model extrapolations, not experimentally resolved frequency bands. Do not claim protection against invasion from a pathogen-free state, eradication, movement between attraction basins, empirical validation of modified coefficients, or posterior certainty. This is a local application to the pathogen-present fitted state.

## Final oscillatory test

Run `julia --startup-file=no oscillatory_test.jl` for the nonnegative sinusoidal-input test and decomposition into direct interaction, equilibrium-Jacobian and input-coupling contributions. Figures and tables are saved under `outputs/oscillatory_test/`. See [OSCILLATORY_FINDINGS.md](OSCILLATORY_FINDINGS.md) for the decision and limitations: the selected reversal survives full biological changes, but is not reproduced by the direct fixed-reference contribution alone.

## References

Stein et al. (2013), PLOS Computational Biology 9:e1003388. https://doi.org/10.1371/journal.pcbi.1003388 . Original Dataset S1, MmuE worksheet, contains the parameters. The source paper discusses equilibrium interpretation and limitations in its Stability and Discussion sections.

Jones and Carlson (2018), PLOS Computational Biology 14:e1006001, is a later application of the same fitted model: https://doi.org/10.1371/journal.pcbi.1006001 . No parameters from its extended sporulation/resistance models are used here.
