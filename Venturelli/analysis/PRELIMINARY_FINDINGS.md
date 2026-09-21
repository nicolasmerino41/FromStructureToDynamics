# Venturelli: preliminary findings and decision

The supplied material supports a reproducible application to a published, experimentally fitted model. It does not supply an independent experiment changing an interaction coefficient, so simulated interventions cannot establish empirical validation of structural sensitivity.

## The main feasibility constraint

The full 12-species autonomous equilibrium is infeasible in both T3 and T4: the smallest calculated abundances are -1.514 and -2.794, respectively. Negative abundances cannot be used as a biological reference state. Enumerating all 4,095 nonempty species subsets identifies one stable equilibrium resistant to invasion by absent species in each model. T3 and T4 share six residents: BU, BO, EL, FP, DP and ER.

These are predictions without dilution. The experiments use serial transfer, and the supplied models encode 95% removal at 24 and 48 hours. Therefore the six-resident equilibrium is not the experimental full community. Figure 1 makes this distinction explicit. A periodic-transfer response analysis would require a substantive methodological extension.

## What the framework adds in the conditional six-species application

We impose a hypothetical common sinusoidal change in per-capita growth rates, amplitude 0.001 per hour, with periods from 6 to 336 hours. For each nonzero resident interaction, we predict the change in community fluctuations produced by a 1% coefficient change. The equilibrium, Jacobian and input coupling all change consistently.

The output is a practical ranking: which coefficients most need accurate estimation to predict fluctuations under this stated input and timescale? It is conditional on the model, the forcing and fractional-change normalization; it is not a ranking of measured parameter uncertainty.

T3 has 19 nonzero off-diagonal resident interactions. Rankings at 6 and 336 hours are fairly similar (Spearman correlation 0.932; largest endpoint shift five places). However, the leading interaction does change at intermediate periods: BO → BU leads at both endpoints, while BU → BO leads at sampled periods of approximately 32–69 hours. The arrow means the influence of the first species on the growth of the second species; it does not imply feeding or movement of biomass. An earlier progress description suggesting the same winner throughout the range was too broad.

Figure 2 shows all-period rankings for the links with the largest rank ranges, the strongest response profiles, comparison against coefficient magnitudes, and comparison between T3 and T4. T3 and T4 are different fitted parameter sets, not uncertainty draws. Complete tables include every analyzed link and both fractional and absolute coefficient sensitivities.

## Numerical checks

Analytical total derivatives pass central finite-difference checks for every T3 resident link at four periods. All four inferred resident communities also pass full-Jacobian stability checks, including absent-species invasion directions.

For BO → BU at 24- and 168-hour periods, nonlinear simulations are compared against recomputed local responses for 1%, 5% and 10% coefficient changes. The largest relative waveform discrepancy is approximately 0.0694% at the small forcing amplitude used. This checks the approximation within the model, not against new experimental observations.

The first-order coefficient derivative becomes less accurate as the coefficient change increases: errors are approximately 1.1–1.9% for a 1% change and 13.5–18.5% for a 10% change. Figure 3 documents this limitation. Equilibrium shifts are exported separately; plotted departures compare fluctuations centred on each model's own unforced equilibrium.

## Recommended use in the revision

This is a defensible exploratory example of how empirical parameter estimates can inform timescale-specific priorities for interaction estimation. It also extends the simple fixed-matrix demonstration by consistently propagating a biological coefficient change through equilibrium, Jacobian and forcing coupling.

It is not yet a clean empirical case study of the original experimental protocol. Given limited time, use the current analysis to decide whether the conditional no-dilution question is compelling enough for the manuscript. If the revision needs a direct match to observed community dynamics, a model with an experimentally relevant stable equilibrium would be a better target than developing a new periodic-transfer framework here.

See README.md for assumptions, equations, source provenance and reproduction. Figures and numerical tables are in outputs/; original Data/ files are unchanged.
