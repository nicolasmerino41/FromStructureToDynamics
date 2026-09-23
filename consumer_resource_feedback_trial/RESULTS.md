# First assessment of the consumer–resource trial

## What is promising

The first trial supports a sharper ecological question: **does weakening consumption increase or decrease resource variability, and how does the answer depend on resource suppression and environmental persistence?** It gives a simple relationship involving resource abundance relative to carrying capacity, environmental persistence, and relative consumer response speed. It also rejects an overly simple interpretation in terms of changing consumer delay.

This is a promising analytical starting point, not yet a general ecological result. The strongest qualification is that conclusions depend on whether variability means absolute variance or variance relative to mean abundance. A second is that a modest change in the consumption function removes the baseline persistence reversal in the tested range.

## The preliminary findings

### The periodic reversal is real within the model

At the default parameters, reducing attack rate by 10% increases resource variance by about 11.7% at omega/r=0.1574, but decreases it by about 15.3% at omega/r=0.3724. These cases were selected using the largest positive and negative absolute variance differences on the sampled grid, with a minimum baseline-response threshold. They are illustrative, not evidence that most frequencies have large effects.

Nonlinear resource and consumer variances agree with the local predictions to within 0.093% across the four checked cases. This verifies the weak-forcing approximation for those cases. It does not establish universality or empirical validity.

The resource mean rises by 11.1% and the consumer mean by 2.9%. Relative resource variance (CV²) decreases at both selected frequencies: approximately 9.5% and 31.4%. Thus the first example is not an increase in variability relative to the resource population's own size.

### Environmental persistence and the ecological starting state both matter

At fixed consumer speed gamma=0.2, under OU environmental fluctuations of equal variance:

| Initial resource abundance / K | Short persistence theta=0.01 | Long persistence theta=100 | Interpretation for this comparison |
|---|---:|---:|---|
| 0.2 | +5.0% | +7.8% | Weaker consumption increases absolute variance at both ends |
| 0.4 | -4.8% | +2.2% | The direction changes with environmental persistence |
| 0.6 | -22.9% | -9.5% | Weaker consumption decreases absolute variance at both ends |

These are finite effects of 10% lower attack rate. The first map explores all intervening values rather than only these three examples. At q=0.4 the positive effect under persistent forcing is modest, and appears after the peak in baseline variance; the absolute-scale panel makes that limitation visible.

Relative resource variance declines throughout the sampled OU map. In fact, this monotonicity follows analytically for the type-I model (below). We should not call the absolute-variance reversal a general reversal of ecological stability.

### Feedback timing is not sufficient as an explanation

In the type-I model, the local consumer equation contains only the resource fluctuation: dx_C/dt = k x_R. The consumer therefore lags the resource by 90 degrees at every positive sinusoidal frequency. The delay expressed in time changes with the forcing period, but the phase lag does not. The observed reversal cannot be attributed simply to a frequency-dependent phase angle.

Reducing attack rate increases resource damping, reduces the feedback product, and changes how strongly growth-rate fluctuations act on the resource. At the baseline, resource damping rises 11.1%, the feedback product falls 7.4%, and the environmental input coefficient rises 2.9%. Equilibrium-mediated effects are substantial and can oppose the direct parameter contribution. The figure's decomposition exposes this rather than treating a single-link norm as the total biological effect.

### Some robustness, and an informative failure

The q=0.4 OU reversal persists across the tested consumer speeds gamma=0.05, 0.2 and 0.8 and attack-rate reductions of 1%, 5%, 10% and 20% within the plotted range, although its boundary and magnitude move. It is therefore not only an artefact of a 10% modification.

At the same initial q and gamma, adding saturation at chi=m h/e=0.15 or 0.30 removes that reversal over theta=0.01 to 100: weaker consumption reduces absolute resource variance throughout the tested range. This does not show that saturation prevents reversals everywhere; it shows the baseline result is conditional. The consumer has its own variance response, which need not equal the resource's.

## A relationship that does not require referring to a matrix

For the type-I model, define q=R*/K, gamma=s m/r, and theta=tau r. Resource variance under weak OU growth-rate forcing has the closed form:

```
Var(R) / (K² sigma²) = q (1-q)² / [q + 1/theta + gamma (1-q) theta].
```

This expression is independently checked against the covariance solver. It separates the ecological starting state q from environmental persistence theta and relative consumer speed gamma. Attack-rate reduction increases q, because R*=m/(e a).

For an infinitesimal reduction in attack rate, the sign of the absolute-variance change is the sign of:

```
S(q,gamma,theta) = (1-3q)/theta - 2q²
                  + gamma theta (1-q)(1-2q).
```

No fitted threshold is involved. The expression was differentiated analytically and checked against numerical derivatives. It is not a claimed novel theorem; it is an explanatory result of this trial that requires a literature comparison before publication claims.

It gives several testable statements **within this model and forcing convention**:

- With very short environmental persistence, the sign is controlled by 1-3q. Weakening consumption increases absolute resource variance for q<1/3 and decreases it for q>1/3, in the small-change limit away from the boundary.
- With very long persistence, the sign is controlled by 1-2q. The corresponding threshold is q=1/2.
- Consequently, for 1/3<q<1/2, the infinitesimal effect must change from negative to positive as persistence increases. The speed ratio gamma determines where the switch occurs.
- For q>1/2 all terms in S are negative: no reversal is possible for an infinitesimal weakening in the type-I OU model.
- For q<1/3 both limiting effects are positive, but intermediate behaviour can still depend on gamma. We must not infer that positive endpoints rule out an intermediate negative effect.

For q=0.4 and gamma=0.2 the infinitesimal boundary solves 0.024 theta² - 0.32 theta - 0.2=0, giving theta about 13.93. A finite 10% reduction has a shifted boundary; the figures correctly use exact modified covariances rather than this derivative threshold.

At fixed environmental variance, absolute variance tends to zero at both persistence extremes. The long-persistence limit reflects perfect resource adaptation to growth rate in this model; it does not imply large absolute effects under arbitrarily slow forcing.

### Why relative variability gives a different answer

Dividing by R*² gives:

```
CV² / sigma² = (1-q)² / {q [q + 1/theta + gamma(1-q)theta]}.
```

This decreases strictly with q for 0<q<1 and positive gamma and theta. To see this, let D=q+1/theta+gamma(1-q)theta. The logarithmic derivative with respect to q is:

```
-2/(1-q) - 1/q - D'/D,  with D'=1-gamma theta.
```

If D' is nonnegative every term is nonpositive. If D'<0, the positive final term is less than 1/(1-q), because D+(1-q)D'=1+1/theta>0. The entire derivative is still strictly negative. Thus, for any feasible finite attack-rate reduction in this type-I OU model, relative resource variance decreases even where absolute variance increases.

This is potentially as informative as the reversal: conclusions about an interaction's buffering role depend on the ecological outcome being measured. It also warns us that a manuscript centred only on the colourful absolute-variance map would be incomplete.

## Assessment against the four agreed criteria

| Criterion | First-trial assessment |
|---|---|
| Ecological meaning | Yes: a coherent consumption change alters resource variability, with the extent of resource suppression predicting contrasting responses. Absolute and relative outcomes must be distinguished. |
| Explanatory simplicity | Promising: q, theta and gamma give an explicit sign condition without a particular matrix. The original changing-delay explanation is insufficient. |
| Persistence across assumptions | Partial: speed and intervention size checks are encouraging; saturating consumption changes the conclusion. Consumer self-limitation and additional species remain untested. |
| Elegance, generality and explanatory value | The structural derivative gives one consistent calculation for direct and equilibrium-mediated effects. The minimal model yields a clean ecological relationship. Its extension beyond this special case is the next scientific question. |

## The most useful next investigation

Before enlarging the network, determine how the sign condition changes with consumer self-limitation and saturation. Those changes directly challenge the two-species mechanism, including its perfect adaptation property. Preserve the comparison of absolute and relative variability and keep a biologically coherent intervention.

If a compact criterion survives, we can test it in larger food webs. If it does not, the contribution may instead be a classification of when common intuitions about consumer buffering hold or fail. This trial supplies a concrete lead and meaningful counterexamples; it does not yet justify a general statement that weaker interactions stabilise or destabilise communities under a given environmental timescale.
