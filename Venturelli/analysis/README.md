# Venturelli worked example feasibility and preliminary analysis

This analysis asks which estimated interaction coefficients matter for predictions of community fluctuations at different timescales. It is an empirically parameterised model application. The coefficient-change trajectories are model predictions, not new observations or independent empirical validation.

## Reproduce

Run these from this `analysis` folder:

```
python prepare_data.py
julia --startup-file=no explore_venturelli.jl
```

Python extraction requires `numpy` and `openpyxl`. Julia analysis requires `CairoMakie`; other dependencies are standard libraries. The first execution used Julia 1.12.5 and CairoMakie 0.15.8. Python is used only to read Excel and SBML sources; numerical dynamics and figures follow the repository's Julia workflow. No source files in `../Data` are edited.

`processed/source_inventory.json` records original filenames, SHA256 hashes, workbook descriptions, species ordering, initial conditions, and dilution events. Extraction verifies all 48 supplied SBML reaction formulas against `N .* (mu + alpha*N)` using independent MathML evaluation. It also verifies the 24 h and 48 h events multiply all 12 populations by 0.05. Parameters are matched by named species identifiers, not XML element order.

## What is in the supplied data

- `MOESM6_ESM.zip`: four SBML models, T1 through T4, each with growth rates, interaction coefficients, initial conditions and dilution events.
- `MOESM2_ESM.xlsx`: 66 pairwise relative-abundance series, seven samples each (PW1).
- `MOESM3_ESM.xlsx`: 30 pairwise relative-abundance series, seven samples each (PW2).
- `MOESM4_ESM.xlsx`: 13 multispecies communities, seven samples each, including the full community and single-species dropouts. Twelve species columns are stored as text and are explicitly parsed.
- `MOESM5_ESM.xlsx`: metabolite log2 fold changes, not interaction coefficients or a population fluctuation spectrum.
- Paper, appendix and review history PDFs.

The Excel files contain relative abundance means and do not provide a complete set of absolute abundances, initial total OD, exact sample timestamps, replicate-level observations or posterior parameter draws. Accordingly the analysis does not reconstruct experimental initial conditions, re-fit the models, infer a spectrum from seven samples, or turn T3/T4 differences into a posterior uncertainty interval.

The paper defines T1 as monospecies, T2 as monospecies plus PW1, T3 as monospecies plus PW1 and PW2, and T4 as those data plus the full multispecies time series. T4 is therefore not independent of the full-community observations. The model estimates absolute population proxies as relative abundance times OD600. OD is a biomass proxy, not a species-specific conversion to mass.

## Feasibility finding

The no-dilution coexistence equilibrium is `N* = -alpha \ mu`. T2, T3 and T4 yield negative abundances for some species when all 12 are forced to coexist. Such algebraic equilibria are not valid reference communities. T1 is feasible but has no inferred interspecific network, so it is not an appropriate workaround.

We enumerate every nonempty species subset, require positive resident abundances, negative real parts of resident Jacobian eigenvalues, and negative invasion growth rates for all absent species. There is a single stable uninvadable equilibrium support in each fitted model. T3 and T4 share six residents: BU, BO, EL, FP, DP and ER. This is a mathematical property of the autonomous model without dilution, not an observed six-species endpoint or a newly validated community.

The first figure also runs the supplied 72 h SBML protocol from its stored initial values (0.01 per species), with 95 percent removals at 24 and 48 h. This reproduces the encoded setup, not the paper's experimental fit: experimental starting proportions and total OD are not reconstructed. The observed relative-abundance series are shown separately with sample index rather than invented exact timestamps.

## Precise ecological question and scenario

For the resident subsystem, which fitted coefficient estimates would need to be accurate to predict the response to a hypothetical common oscillation in per-capita growth rates?

The scenario is

```
dN/dt = N .* (mu + alpha*N + a*q*cos(omega*t))
a = 0.001 per hour
q = ones(number of residents)
```

Every resident experiences the same small per-capita forcing. This is not measured environmental forcing or an inferred nutrient effect. Periods span 6 to 336 hours, with `omega=2*pi/period`. Outputs are expressed in the model's OD-derived abundance units.

The default link comparison is the magnitude of the response to a 1 percent change in each nonzero off-diagonal coefficient. This is a standardized fractional perturbation, not an uncertainty estimate. Equal absolute-coefficient sensitivities are also exported because normalization changes the question. Zero coefficients are excluded from percentage-change ranking; they would need an absolute-change analysis to assess uncertain missing links.

## Full parameter derivative

We perturb the fitted coefficient `alpha[i,j]`, allowing equilibrium, Jacobian and input coupling to change. For a unit direction E at (i,j), within a fixed feasible resident support:

```
dN = -alpha^(-1) * E * N*
J = diag(N*) * alpha
dJ = diag(dN)*alpha + diag(N*)*E
b = diag(N*)*q
db = diag(dN)*q
H = (i*omega*I-J)^(-1)
y = H*b
dy = H*(dJ*y + db)
```

The predicted RMS community departure for a fractional coefficient change f is `a*abs(f*alpha[i,j])*norm(dy)/sqrt(2)`. The summed OD-proxy output instead uses `abs(sum(dy))`; its separate sensitivity is exported. The community norm is not total biomass. These calculations describe changes in fluctuations about each model's own unforced equilibrium; changes in the equilibrium mean are recorded separately.

For connection to the manuscript, we export `tau_i=-1/J_ii`, `T=diag(tau)`, and `A=T*J`, so `A_ii=-1`. The parameter derivative does not artificially keep this T fixed. The forcing term `db` is necessary because a fixed per-capita forcing corresponds to abundance-dependent additive forcing in local coordinates.

## Figures and interpretation

1. `01_feasibility.png` / `.svg`: full-coexistence failure, common resident state, encoded dilution protocol, and observed composition. This is the feasibility assessment, not a spectral validation.
2. `02_interaction_priorities.png` / `.svg`: sensitivity rankings across periods, response profiles, comparison with coefficient-magnitude rankings, and dependence on fitting set T3 versus T4. Panel A displays the twelve links with the largest rank ranges to make variation readable; the complete results are in CSV. Panel B selects the largest maximum effects. Neither is an unbiased sample of links.
3. `03_nonlinear_check.png` / `.svg`: selected 1, 5 and 10 percent changes checked against nonlinear gLV simulations at 24 and 168 h periods. Links are the top-ranked ones at the short and long ends of the predefined range, with duplicates removed. This selection is not based on a desired reversal. Displayed species are selected by largest exact response change per example and labelled; quantitative errors use all resident species.

Exact finite-change local responses recompute the equilibrium and Jacobian in each modified model. Independent nonlinear RK4 simulations are run around each model's own equilibrium under the same forcing. Stationary response differences are measured after discarding 25 slowest relaxation times, rounded up to full forcing cycles. Four cycles are retained. Full parameter derivatives are checked against central finite differences across all nonzero resident links and four periods.

Agreement with these simulations validates the approximation within the fitted model. It does not validate a real intervention or recover observed spectra from the supplied time series. No claim of a universal ranking reversal is made; rank stability is also a legitimate result.

## Scope decision

Venturelli supports a transparent model-based worked example, but the serial-transfer protocol is a real obstacle to interpreting a stationary equilibrium calculation as the experiment itself. A protocol-faithful analysis of periodic dilution would require a periodic-orbit or discrete transfer-map response framework. That extension is intentionally not undertaken here because the purpose is a time-limited first exploration.

Before investing in a manuscript case study, decide whether the no-dilution resident-community interpretation answers a useful ecological question. If not, favour another published model with an experimentally relevant stable equilibrium rather than extending the current paper into a new periodic-system study.

## Sources

Venturelli et al. (2018), Molecular Systems Biology 14:e8157, DOI 10.15252/msb.20178157. Supplied `Paper.pdf`: Fig. 1 and Methods for dilution design; Fig. 3 and Appendix Fig. S15 for training and evaluation; Methods pp. 15-16 for gLV equations and absolute-abundance proxy. Supplied `Appendix.pdf`: S20 for uncertainty analysis and S25-S28 for fitted parameters. Numerical parameters are extracted from supplied Code EV1 SBML files; observation meanings and species order come from each workbook's Description sheet.
