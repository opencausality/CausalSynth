<div align="center">

# CausalSynth

**Causally-faithful synthetic data generation.**

Generate synthetic data that preserves the *causal structure* of your original data —
not just its statistical correlations.

[![CI](https://github.com/opencausality/causalsynth/actions/workflows/ci.yml/badge.svg)](https://github.com/opencausality/causalsynth/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is CausalSynth?

CausalSynth generates synthetic tabular data by sampling from a **Structural Causal Model (SCM)**
calibrated to your real data. Unlike standard synthetic data generators, it preserves the
*causal directions* and *interventional distributions* — not just the marginal statistics.

```
Real data  +  Causal DAG
       │
       ▼ SCM fitting
 Structural equations per variable
 (X_i = f(parents) + noise)
       │
       ▼ ancestral sampling
 Synthetic data where:
   - causal structure is preserved
   - privacy constraints apply
   - distributions match original
```

### Key Features

- 🏗️ **SCM-based generation** — samples from structural equations, not from joint distributions
- 📐 **Causal fidelity validation** — quantifies how well synthetic data preserves your DAG
- 🔒 **Privacy mode** — optional differential privacy via Laplace noise
- 🔍 **DAG auto-discovery** — can infer causal structure from data if no DAG provided
- 📊 **Validation report** — KS tests, MMD score, causal_fidelity_score
- 🏠 **Local-first** — fully offline, no API key needed

---

## Why Causal Structure Matters in Synthetic Data

Standard synthetic data generators (CTGAN, SDV, TabPFN, GANs) preserve **marginal distributions**
and **pairwise correlations**. They do not preserve **causal directions**.

**The umbrella problem:**

In real data: `rain → umbrella` (rain causes umbrella use).

A GAN trained on this data sees high correlation between `rain` and `umbrella` and reproduces it.
But it has no concept of direction. A model trained on that synthetic data may learn:
`umbrella → rain` (umbrellas cause rain — the causal direction is reversed).

This is not a toy problem. In healthcare, drug treatment is often confounded by disease severity.
Synthetic data that doesn't preserve this causal structure will produce models that recommend
treatment for the wrong reasons.

| | CTGAN / SDV / GANs | CausalSynth |
|---|---|---|
| **Marginal distributions** | ✅ Preserved | ✅ Preserved |
| **Pairwise correlations** | ✅ Preserved | ✅ Preserved |
| **Causal directions** | ❌ Not guaranteed | ✅ Enforced by construction |
| **Interventional distributions** | ❌ Destroyed | ✅ Preserved |
| **do-calculus consistency** | ❌ Breaks | ✅ Maintained |
| **Causal fidelity score** | Not measured | ✅ 0–1 score reported |
| **Privacy mode** | Depends on tool | ✅ Differential privacy built-in |

### Why This Matters

**Downstream models trained on causally-correct synthetic data will:**
- Learn the right causal mechanisms, not spurious shortcuts
- Generalize better under distribution shift
- Produce valid counterfactual predictions

**Downstream models trained on causally-incorrect synthetic data will:**
- Exploit spurious correlations from the synthetic generation process
- Fail when deployed in environments where the spurious correlation doesn't hold
- Produce systematically wrong counterfactual predictions

---

## Installation

```bash
pip install causalsynth
# or
uv add causalsynth
```

**Requirements**: Python 3.10+. No API key required.

---

## Quick Start

### 1. Generate synthetic data with a known DAG

```bash
causalsynth generate \
  --data real_health_data.csv \
  --dag health_dag.json \
  --n 1000 \
  --output synthetic_health.csv
```

Output:
```
CausalSynth — Generation Report
══════════════════════════════════

Input data: real_health_data.csv (300 samples, 5 variables)
DAG: health_dag.json (5 nodes, 5 edges)
Target samples: 1000

Fitting Structural Causal Model...
  age:              root node — Gaussian noise (std=15.2)
  bmi:              age → bmi  (β=0.148, noise_std=2.9)
  blood_pressure:   age,bmi → blood_pressure  (β_age=0.31, β_bmi=0.49, noise_std=7.8)
  cholesterol:      blood_pressure → cholesterol  (β=0.82, noise_std=19.4)
  diagnosis:        cholesterol → diagnosis  (β=0.021, logistic)

Generating 1000 samples via ancestral sampling...

Validating causal structure...
  Causal fidelity score: 0.95  ✅ EXCELLENT
  KS test (marginals):   all p > 0.15  ✅ PASS
  MMD score:             0.031  ✅ PASS

Synthetic data saved → synthetic_health.csv
```

### 2. Auto-discover DAG from data

```bash
causalsynth generate \
  --data real_data.csv \
  --auto-discover \
  --n 1000 \
  --output synthetic.csv
```

### 3. Generate with differential privacy

```bash
causalsynth generate \
  --data sensitive_data.csv \
  --dag dag.json \
  --n 1000 \
  --privacy-epsilon 1.0 \
  --output private_synthetic.csv
```

### 4. Validate existing synthetic data

```bash
causalsynth validate \
  --real real.csv \
  --synthetic synthetic.csv \
  --dag dag.json
```

Output:
```
Validation Report
═════════════════

Causal fidelity score: 0.95
  Preserved edges (4/5):
    ✅ age → bmi
    ✅ age → blood_pressure
    ✅ bmi → blood_pressure
    ✅ blood_pressure → cholesterol
    ⚠️ cholesterol → diagnosis  (partial: confidence 0.48, threshold 0.5)

Marginal distributions (KS test):
  age:            p=0.42  ✅ similar
  bmi:            p=0.31  ✅ similar
  blood_pressure: p=0.28  ✅ similar
  cholesterol:    p=0.19  ✅ similar
  diagnosis:      p=0.11  ✅ similar

MMD score: 0.031  ✅ (threshold: < 0.1)

Verdict: PASSED (fidelity: 0.95, threshold: 0.8)
```

---

## CLI Reference

```bash
# Generate synthetic data
causalsynth generate --data real.csv --dag dag.json --n 1000 --output synth.csv
causalsynth generate --data real.csv --dag dag.json --n 1000 --seed 42
causalsynth generate --data real.csv --auto-discover --n 1000 --output synth.csv
causalsynth generate --data real.csv --dag dag.json --n 1000 --privacy-epsilon 1.0
causalsynth generate --data real.csv --dag dag.json --n 1000 --noise laplace

# Validate synthetic data
causalsynth validate --real real.csv --synthetic synth.csv --dag dag.json

# Visualize the causal DAG
causalsynth show-dag --data real.csv --dag dag.json

# REST API server
causalsynth serve --port 8000
```

---

## Architecture

```
Real data + DAG
      │
      ▼
┌──────────────────┐
│   DAG Loader /   │  ← Load from JSON, or auto-discover via
│   Discoverer     │    independence tests on data
└──────────────────┘
      │ CausalDAG
      ▼
┌──────────────────┐
│   SCM Builder    │  ← For each node in topological order:
│                  │    fit OLS on parents, fit noise to residuals
└──────────────────┘
      │ SCM (equations + noise params)
      ▼
┌──────────────────┐
│    Sampler       │  ← Ancestral sampling: traverse topological
│                  │    order, evaluate each structural equation
└──────────────────┘
      │ raw samples
      ▼
┌──────────────────┐
│  Post-processor  │  ← Round integers, clip to valid ranges,
│  + Privacy       │    apply differential privacy if configured
└──────────────────┘
      │ synthetic DataFrame
      ▼
┌──────────────────┐
│   Validator      │  ← KS tests, MMD, causal fidelity check
│                  │    Re-discover DAG from synthetic, compare
└──────────────────┘
```

---

## DAG Format

```json
{
  "nodes": ["age", "bmi", "blood_pressure", "cholesterol", "diagnosis"],
  "edges": [
    ["age", "bmi"],
    ["age", "blood_pressure"],
    ["bmi", "blood_pressure"],
    ["blood_pressure", "cholesterol"],
    ["cholesterol", "diagnosis"]
  ]
}
```

---

## How SCM Generation Works

For each variable in **topological order** (parents before children):

1. **Fit structural equation**: `X_i = β_1 * parent_1 + β_2 * parent_2 + ... + intercept + noise`
   - Coefficients fit by OLS regression on real data
2. **Fit noise distribution**: Gaussian (default), Laplace, or Uniform — fitted to OLS residuals
3. **Sample**: For each new synthetic row, compute `X_i` from (already sampled) parent values + fresh noise

This preserves the causal structure by construction: the generating process mirrors the real
causal mechanism.

---

## Privacy Mode

```bash
causalsynth generate --data sensitive.csv --dag dag.json --privacy-epsilon 1.0
```

With `--privacy-epsilon ε`, CausalSynth adds Laplace noise to generated values with scale
`sensitivity / ε`. Lower ε = stronger privacy = lower fidelity. Typical values: 0.1 (strong), 1.0 (moderate), 10.0 (weak).

Note: differential privacy is applied as post-processing and adds noise on top of the SCM's
natural noise — causal structure is still preserved to within the privacy budget.

---

## Configuration

```env
CAUSALSYNTH_DEFAULT_NOISE=gaussian  # "gaussian", "laplace", "uniform"
CAUSALSYNTH_N_SAMPLES=1000
CAUSALSYNTH_SEED=42
CAUSALSYNTH_PRIVACY_EPSILON=       # empty = no differential privacy
CAUSALSYNTH_LOG_LEVEL=INFO
```

---

## Data Model

```python
@dataclass
class StructuralEquation:
    variable: str
    parents: list[str]
    coefficients: dict[str, float]  # parent → coefficient
    intercept: float
    noise_type: str                 # "gaussian", "laplace", "uniform"
    noise_std: float

@dataclass
class ValidationReport:
    causal_fidelity_score: float     # 0–1, fraction of edges preserved
    ks_test_results: dict[str, float] # variable → p-value
    mmd_score: float
    edges_preserved: list[tuple]
    edges_broken: list[tuple]
    verdict: str                     # "PASSED", "WARNING", "FAILED"
```

---

## Philosophy

CausalSynth is built on the principle that **synthetic data should preserve mechanism, not just statistics**.

- 🏠 **Local-first**: Fully offline — your sensitive data never leaves your machine
- 🔓 **Open source**: All SCM logic and validation code is MIT licensed
- 🚫 **No telemetry**: Zero data collection
- 🧠 **Causal, not correlational**: Structural equation sampling, not joint distribution mimicry

---

## Contributing

CausalSynth is free for research, privacy engineering, and educational use.
If you're building production privacy-preserving data pipelines on top of CausalSynth,
consider contributing domain-specific DAG templates and SCM calibration benchmarks.

*"Good synthetic data teaches the right lesson."*
