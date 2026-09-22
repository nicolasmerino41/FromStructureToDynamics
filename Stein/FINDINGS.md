# Stein screening: useful frequency dependence, weak pulse contrast

## Assessment

**Qualified yes: this is a viable empirically parameterized example of frequency- and output-specific interaction sensitivity. It is not a compelling short-versus-long pulse demonstration in its current form.** No parameters were refitted or tuned to produce this assessment.

The original dataset and paper are in Data/. All 42 directed resident interactions were screened. The final pipeline passes 258 assertions covering extraction-dependent equilibrium checks, derivatives, small finite changes, and numerical verification. See README.md for equations, reproduction and assumptions.

## What is promising

The source coefficients give a feasible seven-group pathogen-present equilibrium, stable also against invasion by absent groups. The fitted antibiotic susceptibilities specify the input direction. All +/-1% resident coefficient changes remain feasible and stable. This retains the published model's post-disturbance equilibrium interpretation without removing a recurring serial-transfer protocol.

For the C. difficile output, the interaction ranked first changes across the sampled forcing-period range:

| Sampled periods | Leading interaction |
|---|---|
| Approximately 1–2.5 days | Blautia → C. difficile |
| Approximately 2.6–11.8 days | C. difficile → Other |
| Approximately 12.2–15.8 days | Other → C. difficile |
| Approximately 16.3–180 days | C. difficile → Akkermansia |

Intervals are sampled-grid classifications, not accurately located crossover thresholds. The arrow denotes the influence of the first group on the growth of the second; the pathway can feed back indirectly to the focal pathogen.

The same sequence of four winning interactions survives recomputation using exact finite-change local responses at both 0.1% and 1% coefficient changes. At every sampled period, the correlation between derivative and exact pathogen rankings exceeds 0.9995 for 0.1% changes and 0.9941 for 1% changes. This is strong numerical support for the ranking pattern within this fitted model, though not for its robustness to parameter-estimation uncertainty.

Community and pathogen outputs give different priorities. Ranking coefficient magnitudes alone is a poor substitute: for the seven-day pulse, its rank correlation with pathogen sensitivity is only 0.102. Some links shift as many as 34 places across the frequency range, but the overall short/long pathogen rankings remain fairly correlated (0.912). Thus frequency matters selectively; the whole ranking is not completely reorganized.

## What did not work as hoped

Equal-exposure rectangular pulses of 0.25–28 days give almost unchanged pathogen priorities: C. difficile → Akkermansia ranks first throughout, and the endpoint rank correlation is 0.998. Community pulse ranks are also similar (correlation 0.992).

The first-order community calculation shows a change of winner between C. difficile → Other for short pulses and C. difficile → Akkermansia for longer pulses. This is a near tie: the effect ratio is about 0.986 for a half-day pulse and 1.008 for a 14-day pulse. It disappears when finite 0.1% changes are evaluated: the Akkermansia link wins at both example durations. **Do not use this near-tie reversal as the main result.**

A plausible interpretation is that changing pulse duration spreads input across many frequencies while retaining a strong slow-response contribution, unlike the frequency-resolved diagnostic. That explanation has not been isolated by a separate spectral decomposition here.

## Locality is a substantive limitation

Akkermansia is rare in the selected equilibrium (0.02138 source abundance units). Increasing the magnitude of the C. difficile → Akkermansia coefficient reduces its equilibrium abundance. The seven-group state reaches its feasibility/stability boundary at about a **2.77%** coefficient increase. Changes of 5% and 10% are therefore recorded as invalid for this local comparison, not silently plotted or replaced with a different community.

For this selected interaction and the two pulse examples:

| Coefficient change | Largest first-order waveform error |
|---|---|
| 0.1% | 2.12% |
| 0.5% | 10.90% |
| 1% | 22.64% |

In contrast, the exact recomputed local response differs from the nonlinear simulations by at most 0.184% under the small pulse exposure used. These are distinct approximations: linearity in disturbance amplitude works very well here, while linearity in the coefficient change has a narrower range. Stable rankings do not guarantee accurate first-order magnitudes.

Freezing equilibrium and input coupling gives errors around 100% for the selected case. Therefore the full parameter derivative is essential to the present interpretation. The analysis includes changes in equilibrium, Jacobian and additive input coupling; it is not a pure fixed-equilibrium modification of a single Jacobian entry.

## What we can claim, and the next decision

The defensible result is: **in a published fitted community model with a biologically specified disturbance direction, the interaction most consequential for a focal response depends on forcing period and on the output being predicted.** The frequency-ranking sequence is supported by small finite-change calculations.

The current pulses do not add a robust dramatic contrast. A minimal next development would be to assess this frequency-specific pattern under bounded, nonnegative oscillatory exposure with an explicitly stated small baseline, checking the corresponding reference equilibrium and finite changes. This is a possible next experiment, not a completed result or a condition to conceal an unhelpful outcome.

Important limits remain: the parameters are a point estimate, groups are taxonomically coarse, long periods extend well beyond the source's weeks-long observations, and the exact seven-group equilibrium is model-predicted. Neither the parameter-change trajectories nor the frequency ranking have independent experimental validation. The work does not support conclusions about clinical dosing, toxin production, infection onset or transitions between stable states.

## Files to inspect

* outputs/01_sensitivity_screen.png: main screen; first-order frequency profiles include every frequency winner, and panel D explicitly labels the non-robust pulse reversal.
* outputs/02_pulse_checks.png: both time panels show C. difficile in nonlinear simulations, with community departure and approximation error quantified separately.
* outputs/finite_frequency_rank_summary.csv: finite-change verification of the frequency ranking.
* outputs/finite_pulse_rank_summary.csv: finite-change check that rejects the small community pulse reversal.
* outputs/nonlinear_checks.csv and selected_link_boundary.csv: locality limits and model verification.

Source: Stein et al. (2013), PLOS Computational Biology 9:e1003388, https://doi.org/10.1371/journal.pcbi.1003388 . All numerical statements above are from the screening scripts using its original Dataset S1.
