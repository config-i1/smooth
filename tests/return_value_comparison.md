Fixes in om():

1. Change the name from "iConstant[F]" to "oETS[F](MNN)" in case of occurrence="fixed"
2. Return the occurrence type in "occurence", not in "occurrenceType"
3. Return constant value, ets, ICs, ICw similar to how it is done in adam().
4. Remove iprob.
5. Add models in case of combined forecasts, similar to adam().
6. Add other parameter, similar to how it is done in adam()
7. Add profileInitial (as in adam())
8. Return res as the result of the nloptr run (see how it is done in adam()).
9. Don't return y.

1. Implement the Fisher Information calculation, the same way it is done in adam(), passing the FI in ellipses and returning FI.
2. Implement the combined forecasts, similar to how it is done in adam().



# Return-value comparison: `oes()`, `om()`, `adam()`

This document compares the named lists returned by the three modeling functions in the **smooth** package and highlights naming and semantic differences. Field lists were extracted directly from:

- `oes()` — `R/oes.R` (full output assembled around lines 925, 1245–1290; `accuracy` set at 1229–1236)
- `om()` — `R/om.R`, two return paths: trivial `iConstant[F]` early return at lines 225–254 (22 fields) and full return at lines 922–973 (36 fields)
- `adam()` — `R/adam.R`: base list built in `preparator()` at lines 2168–2179, augmented at 2905–2943 (post-fit) and 3096–3103 (post-selection), plus a combined-model branch at 3084–3094

## Class hierarchy

| Function | Class vector |
|----------|--------------|
| `oes()`  | `c("oes", "smooth", "occurrence")` |
| `om()`   | `c("om", "adam", "smooth")` |
| `adam()` | `c("adam", "smooth")` (or `c("adamCombined", "adam", "smooth")` for combined) |

`om()` is intentionally part of the `adam` class chain; `oes()` is its own lineage rooted in a generic `occurrence` class.

## Field inventories

### `oes()`
Source: `R/oes.R:925` (full path), 1245–1290 (post-fit additions), 1228–1236 (accuracy).

`accuracy`, `B`, `distribution`, `fitted`, `fittedModel`, `forecast`, `forecastModel`, `initial`, `initialSeason`, `initialX`, `logLik`, `loss`, `lower`, `lowerModel`, `model`, `nParam`, `occurrence`, `persistence`, `persistenceX`, `phi`, `residuals`, `s2`, `states`, `timeElapsed`, `transitionX`, `updateX`, `upper`, `upperModel`, `xreg`, `y`

Notes:
- `accuracy` is always present (set to `NA` when `holdout=FALSE`).
- `distribution` is hardcoded to `"plogis"`.
- `loss` is hardcoded to `"likelihood"`.
- For `occurrence="n"` (none) the output collapses: `lower`/`upper` become `NA`, `persistence`/`initial`/`initialSeason` become `NULL`.

### `om()`
Two return paths.

**Early return (`R/om.R:225-254`, fixed-probability `iConstant[F]` model)** — 22 fields:

`B`, `call`, `distribution`, `fitted`, `forecast`, `iprob`, `lags`, `lagsAll`, `logLik`, `loss`, `model`, `nParam`, `occurrence`, `occurrenceType`, `orders`, `persistence`, `phi`, `residuals`, `scale`, `states`, `timeElapsed`, `y` (plus `holdout`, `accuracy` if `holdout=TRUE`).

**Full return (`R/om.R:922-973`)** — 36 fields:

`adamCpp`, `arma`, `B`, `bounds`, `call`, `data`, `distribution`, `fitted`, `forecast`, `formula`, `initial`, `initialEstimated`, `initialType`, `iprob`, `lags`, `lagsAll`, `logLik`, `loss`, `lossFunction`, `lossValue`, `measurement`, `model`, `nParam`, `occurrence`, `occurrenceType`, `orders`, `persistence`, `phi`, `profile`, `regressors`, `residuals`, `scale`, `states`, `timeElapsed`, `transition`, `y` (plus `holdout`, `accuracy` if `holdout=TRUE`).

Notes:
- `occurrence` is always `NULL` (the wrapper does not nest an oes-style sub-object).
- `occurrenceType` carries the string identifier ("fixed", "odds-ratio", etc.).
- `distribution` is hardcoded to `"plogis"`.
- `scale` is hardcoded to `NA`.

### `adam()`
Source: `R/adam.R:2168-2179` (preparator base), 2905–2943 (post-fit augmentation), 3096–3103 (post-selection augmentation), 3084–3094 (combined-model branch).

Always present (non-combined path): `adamCpp`, `arma`, `B`, `bounds`, `call`, `constant`, `data`, `distribution`, `ets`, `FI`, `fitted`, `forecast`, `formula`, `holdout`, `ICs`, `initial`, `initialEstimated`, `initialType`, `lags`, `lagsAll`, `logLik`, `loss`, `lossFunction`, `lossValue`, `measurement`, `model`, `nParam`, `occurrence`, `orders`, `other`, `persistence`, `phi`, `profile`, `profileInitial`, `regressors`, `res`, `residuals`, `scale`, `states`, `timeElapsed`, `transition`.

Conditional:
- `accuracy` — only when `holdout=TRUE` (line 3103).
- `models`, `ICw` — only when `modelDo == "combine"` (combined output adds class `"adamCombined"`).
- `occurrence` — populated with a nested oes-style model object when `occurrence != "none"`; otherwise effectively a flag.

## Comparison table

`✓` = present always; `✓ᶜ` = present conditionally (see footnotes); `—` = absent. Differences in semantics are noted inline.

| Field | `oes()` | `om()` | `adam()` |
|-------|:-------:|:------:|:--------:|
| `accuracy` | ✓ (NA if no holdout) | ✓ᶜ¹ | ✓ᶜ¹ |
| `adamCpp` | — | ✓² | ✓ |
| `arma` | — | ✓² | ✓ |
| `B` | ✓ | ✓ | ✓ |
| `bounds` | — | ✓² | ✓ |
| `call` | — | ✓ | ✓ |
| `constant` | — | — | ✓ |
| `data` | — | ✓² | ✓ |
| `distribution` | ✓ (`"plogis"`) | ✓ (`"plogis"`) | ✓ (varies³) |
| `ets` | — | — | ✓ |
| `FI` | — | — | ✓ |
| `fitted` | ✓ (probabilities) | ✓ (probabilities) | ✓ (demand or prob × demand) |
| `fittedModel` | ✓ | — | — |
| `forecast` | ✓ (probabilities) | ✓ (probabilities) | ✓ |
| `forecastModel` | ✓ | — | — |
| `formula` | — | ✓² | ✓ |
| `holdout` | — | ✓ᶜ¹ | ✓ |
| `ICs` | — | — | ✓ |
| `ICw` | — | — | ✓ᶜ⁴ |
| `initial` | ✓ | ✓² | ✓ |
| `initialEstimated` | — | ✓² | ✓ |
| `initialSeason` | ✓ | — | — |
| `initialType` | — | ✓² | ✓ |
| `initialX` | ✓ | — | — |
| `iprob` | — | ✓ | — |
| `lags` | — | ✓ | ✓ |
| `lagsAll` | — | ✓ | ✓ |
| `logLik` | ✓ | ✓ | ✓ |
| `loss` | ✓ (`"likelihood"`) | ✓ | ✓ |
| `lossFunction` | — | ✓² | ✓ |
| `lossValue` | — | ✓² | ✓ |
| `lower` | ✓ | — | — |
| `lowerModel` | ✓ | — | — |
| `measurement` | — | ✓² | ✓ |
| `model` | ✓ | ✓ | ✓ |
| `models` | — | — | ✓ᶜ⁴ |
| `nParam` | ✓ | ✓ | ✓ |
| `occurrence` | ✓ (string) | ✓ (always `NULL`) | ✓ (nested oes-object or flag) |
| `occurrenceType` | — | ✓ (string) | — |
| `orders` | — | ✓ | ✓ |
| `other` | — | — | ✓ |
| `persistence` | ✓ | ✓ | ✓ |
| `persistenceX` | ✓ | — | — |
| `phi` | ✓ | ✓ | ✓ |
| `profile` | — | ✓² | ✓ |
| `profileInitial` | — | — | ✓ |
| `regressors` | — | ✓² | ✓ |
| `res` | — | — | ✓ |
| `residuals` | ✓ | ✓ | ✓ |
| `s2` | ✓ | — | — |
| `scale` | — | ✓ (always `NA`) | ✓ (estimated) |
| `states` | ✓ | ✓ | ✓ |
| `timeElapsed` | ✓ | ✓ | ✓ |
| `transition` | — | ✓² | ✓ |
| `transitionX` | ✓ | — | — |
| `updateX` | ✓ | — | — |
| `upper` | ✓ | — | — |
| `upperModel` | ✓ | — | — |
| `xreg` | ✓ | — | — |
| `y` | ✓ | ✓ | — |

Footnotes:

1. `accuracy` and `holdout` on `om()`/`adam()` appear only when the user passed `holdout=TRUE`. `oes()` always sets `accuracy` (as `NA` when no holdout).
2. `om()` has two return paths. Fields marked ²  are only present in the **full** return path (`R/om.R:922-973`). The trivial `iConstant[F]` early return omits them. The 22 fields produced by the early return are: `model`, `occurrenceType`, `occurrence`, `loss`, `distribution`, `timeElapsed`, `fitted`, `residuals`, `forecast`, `states`, `B`, `persistence`, `phi`, `lags`, `lagsAll`, `orders`, `logLik`, `nParam`, `scale`, `iprob`, `y`, `call`.
3. `adam()` `distribution` reflects the chosen error distribution (`dnorm`, `dlaplace`, `ds`, `dgnorm`, `dlnorm`, `dgamma`, `dinvgauss`).
4. `models` and `ICw` only exist when `modelDo == "combine"`, in which case the class becomes `c("adamCombined","adam","smooth")`.

## Semantic differences worth flagging

- **`occurrence` field is overloaded.** In `oes()` it is a string; in `om()` it is always `NULL` (the string lives in `occurrenceType`); in `adam()` it is a nested oes-style model object (or a flag/`NULL`). Code consuming any of these objects must dispatch on class before reading `$occurrence`.
- **Probability handling.** `oes()` and `om()` return probabilities in `fitted`/`forecast`. `adam()` may return demand-level forecasts that already incorporate occurrence probabilities; the underlying probability model is reachable via `adam_object$occurrence$fitted`.
- **`iprob` is unique to `om()`.** It records the in-sample empirical probability `mean(oInSample)`. `oes()` exposes the analogous quantity through the `initial` component of its state vector; `adam()` does not surface it directly.
- **Underlying-model exposure.** `oes()` exports `fittedModel`, `forecastModel`, `lowerModel`, `upperModel` so callers can introspect the latent ETS model fit. `om()` and `adam()` deliberately do not — `om()` is the wrapper-style replacement and `adam()` only nests an oes-object inside `$occurrence` when needed.
- **Prediction intervals.** `oes()` returns `lower`/`upper` directly when `interval != "none"`. `om()` and `adam()` route intervals through `forecast()`/`forecast.adam()` instead and therefore omit those fields from the model object.
- **xreg representation.** `oes()` carries the older `updateX` / `initialX` / `persistenceX` / `transitionX` / `xreg` quartet plus matrix. `om()` and `adam()` use the modern ADAM scheme with `data`, `formula`, `regressors` and a unified state vector.
- **Profile tables.** Only ADAM-family functions (`om()`, `adam()`) populate `profile`; `adam()` additionally populates `profileInitial`. `oes()` predates the profile-table machinery.
- **Scale.** `om()` always returns `scale = NA` (no scale parameter for a binary likelihood). `adam()` returns the estimated/computed residual scale. `oes()` instead returns `s2` (mean squared residual).
- **`adamCpp` flag.** Internal switch indicating which C++ engine the fit used. ADAM-only.
- **`y` in `om()` and `oes()`.** Both return the binary occurrence vector (zeros and ones). `adam()` does not return `y` separately; the original series is reachable via `data` (column or vector depending on input shape).
- **Loss field.** `oes()` is fixed to `"likelihood"`. `om()`/`adam()` carry whatever loss was requested.
