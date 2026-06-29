---
name: explain-smooth
description: Explain and interpret smooth (ADAM) state-space forecasting outputs in plain language, and pick the right model function — ADAM/AutoADAM, ES, CES/AutoCES, MSARIMA/AutoMSARIMA, SMA, the occurrence models OM/OMG/AutoOM for intermittent demand, msdecompose, and the sim_* simulators. Covers ETS model notation (the three-letter code and Z/X/Y/C/F selection placeholders), persistence/smoothing parameters (alpha, beta, gamma, phi) and their constraints, ARIMA orders, error distributions, information-criteria model selection, point forecasts and prediction intervals, component/state decomposition, and holdout accuracy. Use when the user asks what a fitted model means, how to read a summary/forecast/plot, why a model or distribution was selected, or which function fits their data — in either the R package or the Python port.
---

# Explaining smooth (ADAM)

This skill makes **smooth** results understandable. The goal is not to add
features but to translate the package's state-space output into clear, correct,
plain-language explanations grounded in what the code actually computes.

`smooth` implements **Single Source of Error (SSOE)** state-space models. The
centrepiece is **ADAM** (Augmented Dynamic Adaptive Model), which unifies ETS
(Error/Trend/Seasonal), ARIMA, and regression in one framework. It exists as an
R package (repo root: `R/`, `src/` C++) and a Python port
(`python/src/smooth/`); the two are designed for equivalence, so answer in the
user's language idiom but the statistics are the same.

## How to explain (principles)

1. **Read the code, don't guess.** When a number's meaning is unclear, trace the
   producing function (R `R/adam.R`, `R/es.R`, … or Python
   `python/src/smooth/adam_general/core/`) before explaining.
2. **State the definition, then the interpretation.** First what the quantity
   *is*, then what *this value* says about the data (stable vs reactive, strong
   vs weak seasonality, decisive vs marginal IC difference).
3. **Lead with the takeaway.** Start with the one-sentence conclusion ("ADAM
   chose additive trend, multiplicative seasonality, and forecasts continued
   growth"), then justify it from the parameters.
4. **Map the model code to words.** Always expand the ETS three-letter notation
   and the chosen orders into prose (see below) — users find the codes opaque.
5. **Respect the conventions.** Multiplicative/non-normal errors give
   **asymmetric** prediction intervals; smoothing parameters live on `[0,1]`
   with ordering constraints; information criteria are only meaningful as
   *differences*.
6. **Prefer a plot.** Fitted models expose plots of the forecast, the fitted
   values, and the decomposed states/components — suggest them and say what to
   look for.
7. **Flag uncertainty honestly.** Wide intervals, tiny IC gaps, near-boundary
   smoothing parameters (α≈1, φ≈1) all deserve a caveat rather than overclaiming.

## Reading the model notation

### ETS three-letter code (`model=`)
Position = **Error**, **Trend**, **Seasonal**. Component letters:
- **Error**: `A` additive, `M` multiplicative.
- **Trend**: `N` none, `A` additive, `Ad` additive damped, `M` multiplicative,
  `Md` multiplicative damped.
- **Seasonal**: `N` none, `A` additive, `M` multiplicative.

Example: `model="MAdM"` = multiplicative error, additive **damped** trend,
multiplicative seasonality. Additive seasonality = roughly constant seasonal
swing; multiplicative = swing scales with the level. A **damped** trend means
the slope flattens out into the future (φ < 1).

### Selection placeholders
A letter can be a *request to select* rather than a fixed component:
- `Z` — let the model **select** that element from all options by IC.
- `X` — select among **additive-only** options; `Y` — among
  **multiplicative-only** options.
- `C` — **combine** candidates (forecast = IC-weighted average);
  `F` — full pool; `P` — pure (no mixed additive/multiplicative).

So `model="ZZZ"` = select everything; `model="ZXZ"` = select error and seasonal
freely but restrict the trend to additive. When a model was selected, explain
*which* concrete code won and that the IC drove the choice.

### ARIMA orders
`orders=list(ar=p, i=d, ma=q)` (R) / `orders={"ar":p,"i":d,"ma":q}` (Python),
optionally per-lag for seasonal ARIMA. Explain `i`=differencing (removes
trend/unit root), `ar`=dependence on past values, `ma`=dependence on past
errors. With `select=TRUE` the orders are chosen by IC.

## Interpreting the main outputs

### Smoothing / persistence parameters (`alpha`, `beta`, `gamma`, `phi`)
The heart of an ETS explanation. Each is on `[0,1]`:
- **α (level)** — how fast the level tracks new data. Near 0 = stable/slow,
  near 1 = highly reactive (almost a random walk).
- **β (trend)** — how fast the slope updates. Constraint `β ≤ α`.
- **γ (seasonal)** — how fast the seasonal profile updates. Constraint
  `γ ≤ 1 − α`. One γ per seasonal lag for multiple seasonality.
- **φ (damping)** — `φ < 1` flattens the trend; `φ = 1` is undamped.

Translate values, e.g. "α=0.08 → the level barely reacts to noise (smooth,
stable series); β≈0 → an essentially fixed slope." Parameters are estimated by
maximum likelihood (C++ optimiser in R; Python core), subject to the constraints
above (violations are penalised, not allowed).

### Forecasts and prediction intervals
`forecast(model, h=)` (R) / `model.predict(h=)` (Python) give point forecasts
and, with an interval level, **prediction intervals** (uncertainty of a future
*observation*, wider than confidence bands). The interval shape follows the
error `distribution` and the model type: multiplicative or skewed distributions
(`dlnorm`, `dgamma`, `dinvgauss`) yield **asymmetric** intervals — explain why
the bounds are not symmetric around the point forecast.

### Error distribution (`distribution=`)
`dnorm` (default additive), `dlaplace`, `ds`, `dgnorm`, `dlnorm`, `dgamma`
(default multiplicative), `dinvgauss`. Explain the tail/support trade-off:
heavier tails (`dlaplace`, `ds`) for outlier-prone data; positive-only families
(`dgamma`, `dinvgauss`, `dlnorm`) for strictly positive / multiplicative data.
The choice changes the likelihood and the interval widths.

### Information criteria and selection (`ic`, `aic`/`aicc`/`bic`/`bicc`)
Used to pick the model / orders / distribution. Only **differences** matter:
ΔIC < 2 ≈ indistinguishable, 2–6 positive, 6–10 strong, >10 decisive. AICc for
short series; BIC/BICc penalise complexity more. With `AutoADAM`/`auto.adam`,
`AutoCES`, `AutoMSARIMA`, explain that many candidates were fit and the lowest
IC won (or were combined under `C`).

### Components / states and residuals
ADAM decomposes the series into level, trend, seasonal (and ARIMA/regression)
states — the states plot shows how each evolves. For diagnostics, read the
residuals for remaining autocorrelation or heteroscedasticity; a good model
leaves approximately unstructured residuals consistent with the chosen
distribution.

### Holdout accuracy
When a holdout is used, accuracy measures (MAE, RMSE, MASE/RMSSE vs naive, and
percentage/relative measures) come from **greybox** — see the
[explain-greybox skill] in that repo for how to read each. Prefer scale-free
measures (MASE, rMAE) to compare across series.

## The model functions (what each one is, when to use it, how to read it)

Quick picker:

| Goal | R | Python |
|---|---|---|
| General-purpose forecasting (recommended default) | `adam()` | `ADAM` |
| Automatic ADAM (select orders + distribution) | `auto.adam()` | `AutoADAM` |
| Pure exponential smoothing (ETS only) | `es()` | `ES` |
| Complex Exponential Smoothing | `ces()` | `CES` / `AutoCES` |
| State-space (multiple seasonal) ARIMA | `(ms/ss)arima()` | `MSARIMA` / `AutoMSARIMA` |
| Simple moving average | `sma()` | `SMA` |
| Intermittent demand — occurrence probability | `oes()` | `OM` / `OMG` / `AutoOM` |
| Multiple-seasonal decomposition | `msdecompose()` | `msdecompose` |
| Simulate data from a model | `sim_*()` | `sim_es`, `sim_ces`, `sim_ssarima`, `sim_sma`, `sim_gum`, `sim_oes` |

### `ADAM` — the unified model
The recommended default and the engine behind every other class here. Combines
**ETS + ARIMA + regression** in one SSOE state space. Fitted attributes to read:
the ETS code chosen, the persistence parameters (`persistence_level_` = α,
`persistence_trend_` = β, `persistence_seasonal_` = γ, `phi_` = φ), the ARIMA
orders, the distribution, and `aic`/`aicc`/`bic`/`bicc`. Use it unless the user
wants the simpler, more constrained behaviour of one of the wrappers below.

### `AutoADAM` — automatic ADAM
`ADAM` with **automatic selection of ARIMA orders and the error distribution**
(ETS selection via the `Z/X/Y/C/F` codes is already handled inside ADAM). Explain
the result by naming the winning ARIMA orders and distribution and noting the IC
chose them. `AutoMSARIMA` is the same machinery restricted to pure ARIMA
(`model="NNN"`, `dnorm`).

### `ES` — Exponential Smoothing (ETS only)
A thin `ADAM` wrapper for **pure ETS with no ARIMA**, normal errors, default
`model="ZXZ"`. Use when the user wants classic ETS. Interpret exactly as the ETS
notation + persistence parameters above; everything is level/trend/seasonal with
no AR/MA dynamics.

### `CES` — Complex Exponential Smoothing
Uses **complex-valued smoothing parameters** to capture the level *and the
"potential"* (rate of change) jointly, so it does not need an explicit
trend/seasonal classification. `seasonality` ∈ `{"none","simple","partial",
"full"}`; the smoothing parameters are `a` (complex) and, for partial/full, `b`
(real for partial, complex for full). Stability is governed by `bounds=
"admissible"` (eigenvalues inside the unit circle). When explaining, focus on the
seasonality mode chosen and whether the complex roots imply stable, slowly
evolving dynamics. `AutoCES` fits the candidate seasonality types and keeps the
lowest-IC one — report which mode won.

### `MSARIMA` / `AutoMSARIMA` — state-space ARIMA
Pure (multiple-seasonal) ARIMA in SSOE form (`ADAM` with `model="NNN"`,
`dnorm`); default is **ARIMA(0,1,1)**. Orders are `orders={"ar":p,"i":d,"ma":q}`
(int or per-lag list) with `lags` giving the seasonal periods, e.g.
`lags=[1,12]`. Explain `i`=differencing, `ar`=past-value dependence, `ma`=
past-error dependence; `constant` adds a drift/intercept. `AutoMSARIMA` (or
`orders={"select":True}`) chooses the orders by IC.

### `SMA` — Simple Moving Average
SMA(m) is an **AR(m) model with every coefficient fixed at 1/m**, implemented over
`ADAM` so it still gives multi-step forecasts and intervals. If `order` is
omitted it is **selected by IC** (ternary search when `fast=True`, full scan when
`fast=False`). Explain the chosen order as the averaging window.

### Occurrence models — `OM`, `OMG`, `AutoOM` (intermittent demand)
For intermittent series these model the **probability that demand occurs** (the
`o_t` Bernoulli term), independently of its size; pair them with a size model
(e.g. an `ADAM`) for the full intermittent forecast.
- **`OM`** — single occurrence ETS model on the `plogis` link. `occurrence` ∈
  `{"fixed","odds-ratio","inverse-odds-ratio","direct"}`; model name renders as
  `oETS(MNN)[O]`-style. Interpret `fitted` as occurrence probabilities ∈ (0,1).
- **`OMG`** — the **general** occurrence model: two parallel ETS sub-models
  (odds-ratio `model_a` and inverse-odds-ratio `model_b`) combined, name
  `oETS[G](MNN)(MNN)`. Use when a single mechanism does not fit the on/off
  pattern.
- **`AutoOM`** — fits an `OM`/`OMG` for each requested `occurrence` type and
  returns the lowest-IC one directly. (In R this is `oes(..., occurrence="auto")`;
  in Python also reachable via `OM(occurrence="auto")`.)

### `msdecompose` — multiple-seasonal decomposition
Classical decomposition for series with one or more frequencies (`lags=[12]`,
`type` ∈ `{"additive","multiplicative"}`, `smoother="lowess"`). Returns the
level/trend, the seasonal profile(s) and the remainder — useful for an
exploratory "what is in this series?" answer before modelling. (Compare with
greybox's `stick()`, which quantifies the *strength* of each component.)

### `sim_*` — simulation
`sim_es`, `sim_ces`, `sim_ssarima`, `sim_sma`, `sim_gum`, `sim_oes` generate
synthetic series from each model type (returning a `SimulateResult`). Use for
explaining model behaviour, building intuition, or generating test data — not for
forecasting an observed series.

> **Note:** GUM (Generalised Uniform Model) currently has **no fitting class in
> the Python port** — only `sim_gum`. It is available as `gum()` in R. Do not
> confuse it with `OMG`, which is the *general occurrence* model.

## Reference map

| Topic | Wiki page | R source | Python module |
|---|---|---|---|
| ADAM unified model | `ADAM` | `R/adam.R`, `src/adamGeneral.cpp` | `adam_general/core/adam.py` |
| Auto selection | `ADAM` | `R/adam.R` (`auto.adam`) | `core/auto_adam.py`, `auto_msarima.py`, `auto_om.py` |
| Exponential smoothing | `es` | `R/es.R`, `R/adam-es.R` | `core/es.py` |
| State-space ARIMA | `ssarima` | `R/ssarima.R`, `R/adam-ssarima.R` | `core/msarima.py` |
| CES | `ces` | `R/ces.R`, `R/adam-ces.R` | `core/ces_model.py` |
| Occurrence / intermittent | `oes` | `R/oes.R` | `core/om.py`, `core/omg.py`, `core/auto_om.py` |
| Moving average | `sma` | `R/sma.R` | `core/sma.py` |
| GUM (R-only fitting; sim only in Python) | `gum` | `R/gum.R`, `R/adam-gum.R` | `core/simulate/gum.py` (`sim_gum`) |
| Simulation | `simulate` | `R/sim*.R` | `core/simulate/` |
| Decomposition | — | `R/msdecompose.R` | `core/utils/utils.py` (`msdecompose`) |

`smooth` depends on **greybox** for distributions, information criteria, and
accuracy measures — when an explanation reaches those, defer to the
`explain-greybox` skill and the greybox wiki. The authoritative reference is the
ADAM book: Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented
Dynamic Adaptive Model (ADAM)*, https://openforecast.org/adam/.

When the explanation hinges on a numeric value the user pasted, recompute or
re-read the producing function before committing to an interpretation, and note
any R-vs-Python differences if relevant.
