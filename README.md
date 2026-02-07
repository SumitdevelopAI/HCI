# Uncertainty-Aware Human Perception for Trustworthy HCI

**Project short name:** UncertaintyPose  
**Author:** Sumit Sharma  
**Target group:** Cognition-oriented HCI / perception research 
**Goal:** Produce a research-grade implementation and evaluation that demonstrates calibrated, decomposed uncertainty (aleatoric + epistemic) for keypoint/pose estimation and shows how uncertainty can be used by HCI systems to improve decision outcomes.

## Abstract

Standard human pose and landmark predictors produce point estimates with a scalar "confidence" that is often miscalibrated and uninformative under occlusion, motion blur, sensor noise, or domain shift. This project develops a principled pipeline that:

1. Decomposes uncertainty into aleatoric and epistemic components
2. Produces calibrated probabilistic predictions for keypoints
3. Demonstrates HCI utility via adaptive UI behaviors and a small user/simulated study

**Deliverables:** Reproducible code, figures (calibration curves, AUROC, ablation tables), reproducible demo, and a short research report.

## 1. Problem Statement (Core Depth)

Humans and assistive systems need to know when a perception model is likely to be wrong. For pose/landmark tasks, the common outputs (heatmap peaks, single-value confidence) do not separate model uncertainty (epistemic) from data noise (aleatoric), and they are frequently miscalibrated relative to downstream error measures (PCK/OKS). This makes it unsafe or inefficient to build adaptive HCI behaviors (e.g., automatic actions, nudges, fallback requests for clarification).

### We aim to solve:

- **Accurate probabilistic localization:** Produce per-keypoint predictive distributions (mean + variance)
- **Uncertainty decomposition:** Separate aleatoric vs epistemic uncertainty so different interventions are possible
- **Calibration:** Map predicted uncertainty to empirical error probabilities
- **HCI integration:** Define and validate interface behaviors that use uncertainty to improve task outcomes

## 2. Key Claims to Test

1. Explicit aleatoric + epistemic modeling yields better failure detection (higher AUROC for large errors) than baseline scalar confidence
2. CalibrationNet (post-hoc learned calibration) reduces Expected Calibration Error (ECE) and improves decision thresholds for adaptive UIs
3. Using uncertainty in the interface (visual cues + adaptive automation) improves user decision accuracy and reduces catastrophic actions in an assistive task

## 3. Theoretical Background (Concise, Testable)

### Uncertainty Decomposition

Let *y* denote true keypoint coordinates and *ŷ* the model prediction. We model predictive uncertainty as:

```
Var(y|x) = E[Var(y|x,θ)] + Var(E[y|x,θ])
           └─ aleatoric (σ²ₐ) ─┘   └─ epistemic (σ²ₑ) ─┘
```

where *θ* are model parameters and *x* the input image. Aleatoric uncertainty captures irreducible noise (occlusion, motion blur); epistemic captures model uncertainty (lack of data, OOD).

### Aleatoric Modeling (Heteroscedastic NLL)

For a single keypoint modeled as Gaussian:

```
L_ale = (1/2σ²ₐ)||ŷ - y||² + (1/2)log(σ²ₐ)
```

This follows Kendall & Gal (heteroscedastic regression): predicting σ²ₐ allows the network to downweight noisy labels.

### Epistemic Estimation (MC Dropout / Ensembles)

Use *T* stochastic forward passes (dropout active) producing ŷ⁽ᵗ⁾. Estimate:

```
μ̂ = (1/T)Σᵗ ŷ⁽ᵗ⁾
σ̂²ₑ = (1/T)Σᵗ ||ŷ⁽ᵗ⁾ - μ̂||²
```

### Calibration and Expected Calibration Error (ECE)

For binned predicted confidence/probability *p* and observed accuracy *a*:

```
ECE = Σₘ (|Bₘ|/n)|acc(Bₘ) - conf(Bₘ)|
```

adapted for continuous error via thresholds on OKS/PCK.

### Decision Rule for HCI

Given predicted error probability *p_err* from calibration, interface selects action *A ∈ {auto, ask, safe-mode}* by thresholding:

```
A(x) = {
  auto       if p_err ≤ τ₁
  ask        if τ₁ < p_err ≤ τ₂
  safe-mode  if p_err > τ₂
}
```

Thresholds tuned by validation to balance user effort vs risk.

## 4. Related Methods and Limitations (What Others Do Now)

- **Heatmap peak amplitude as confidence** — Simple but miscalibrated under occlusion
- **Ensembles / Deep ensembles** — Strong epistemic estimates but expensive
- **MC Dropout** — Cheap, approximate epistemic estimate; variance depends on dropout setting
- **Variance heads / heteroscedastic losses** — Model aleatoric noise but do not capture epistemic
- **Conformal prediction / post-hoc calibration** — Provide calibrated intervals but seldom combined with joint aleatoric/epistemic decomposition for pose

**Limitations to address:** Computational cost (keep feasible on RTX 3050), interpretability of per-keypoint distributions, and realistic HCI evaluation (need small user/sim study).

## 5. Proposed Method (Architecture + Math)

### 5.1 High-level Pipeline

Input image *x* → lightweight backbone *f_θ(x)* → three heads:

1. **Mean heatmap / keypoint head** *h_μ* → produces per-keypoint heatmaps / coordinates ŷ
2. **Aleatoric head** *h_σₐ* → produces per-keypoint variance σ̂²ₐ (or per-pixel variance for heatmaps)
3. **Optional attention / explanation head** *h_attn* → saliency map to explain uncertainty

At inference: run *T* stochastic forward passes (dropout) to compute epistemic variance σ̂²ₑ. Aggregate into total predictive variance:

```
σ̂²_pred = σ̂²ₐ + σ̂²ₑ
```

### 5.2 Loss Function

Total loss for training on dataset *D*:

```
L = L_heat + λ_ale·L_ale + λ_reg||θ||²
```

where:
- **L_heat** is standard heatmap MSE / focal loss for mean heatmap
- **L_ale** = Σₖ (||ŷₖ - yₖ||² / 2σ̂²ₐ,ₖ + (1/2)log(σ̂²ₐ,ₖ)) summed over keypoints *k*
- **λ_ale** balances aleatoric training

Epistemic variance is not directly in the loss; encourage epistemic reduction via data augmentation and regular training (ensembles or dropout approximate posterior uncertainty).

### 5.3 CalibrationNet

Train a small calibration network *g_φ* that inputs (ŷ, σ̂²ₐ, σ̂²ₑ) and outputs a calibrated error probability *p_err*. Optimize cross-entropy vs binary label 1(||ŷ - y|| > ε) or regression vs continuous error with ECE reduction objective.

Loss for calibration:

```
L_cal = BCE(g_φ(·), 1(error > ε)) + λ_ece·ECE(g_φ)
```

## 6. Implementation Plan (Code + Practical Tricks)

### 6.1 Backbone and Heads

- **Backbone:** MobileNetV3-Small or Mobile HRNet variant
- **Heatmap head:** Deconvolution / upsample to produce heatmaps; argmax or soft-argmax for coordinates
- **Aleatoric head:** Predict log-variance log(σ²ₐ) to ensure positivity
- **Dropout:** Apply spatial dropout in several backbone blocks for MC passes

### 6.2 Training

- **Framework:** PyTorch. Use `torch.cuda.amp` for mixed precision
- **Batch strategy:** Batch accumulation if GPU memory limits. Default batch 8 at 256×256 crops
- **Augmentations:** Random occluders, motion blur, brightness/contrast, random crops. Use Albumentations
- **MC passes** *T=8* for inference epistemic estimate (tunable). For faster experimentation use *T=4*

### 6.3 Datasets

- **Primary:** COCO Keypoints subset for initial experiments
- **Secondary:** MPII, and synthetic occlusion dataset (apply randomized masks) to test robustness
- **Small held-out OOD test set** (different scenes / lighting) for epistemic evaluation

### 6.4 HCI Demo

- Simple web UI or Jupyter demo that overlays predicted keypoints and visualizes per-keypoint uncertainty (opacity, blur radius, or heatmap spread)
- Adaptive behavior simulation: automated action triggered only when *p_err < τ*; otherwise show a confirm dialog

## 7. Evaluation Protocol and Metrics (Research Proof)

### 7.1 Perceptual Accuracy

- OKS / PCK / mAP on standard validation sets

### 7.2 Uncertainty Quality

**NLL** under predicted Gaussian distributions:
```
NLL = -Σₖ log N(yₖ | ŷₖ, σ̂²_pred,k)
```

**ECE:** Bin predicted error probabilities and compute calibration gap (adapted to continuous error via thresholds)

**AUROC:** Use uncertainty to classify whether ||ŷ - y|| > δ (failure) — AUROC quantifies detection power

### 7.3 HCI Effectiveness

- **Simulated decision accuracy:** Simulate users acting with/without uncertainty cues and measure correct vs incorrect decisions
- **Small user study (optional):** Within-subjects test (N=12–20) comparing baseline UI vs uncertainty-aware UI on a simple annotation or verification task
  - Metrics: task accuracy, decision time, subjective trust (Likert)
  - Use nonparametric tests (Wilcoxon) for significance

### 7.4 Ablations

- Aleatoric only vs epistemic only vs both
- MC Dropout (T variants) vs small ensemble (2–3 models)
- CalibrationNet vs temperature scaling vs isotonic regression

## 8. Expected Results (Benchmarks to Aim For)

- **ECE reduced substantially** (target: 30–60% relative reduction over raw heatmap confidence on corrupted test sets)
- **AUROC for failure detection:** baseline ~0.6 → target ≥0.75 under severe occlusion (dataset dependent)
- **HCI simulated outcome:** Measurable improvement in decision accuracy when using uncertainty thresholds (effect size moderate)

These are realistic research targets for a compact, well-engineered implementation.

## 9. Reproducibility & Repository Structure

```
README.md
paper/                      # 4–6 page research writeup
src/
  data/
  models/
    backbone.py
    heads.py
    calibration.py
  train.py
  eval.py
  inference.py
notebooks/
  demo.ipynb
  hci_simulation.ipynb
results/
  figures/
  metrics/
requirements.txt
run_demo.sh
```

Include `configs/` for small and full runs. Default branch should run a tiny experiment that finishes on an RTX 3050 within a few hours.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick demo experiment
bash run_demo.sh

# Open interactive demo
jupyter notebook notebooks/demo.ipynb
```

## Citation

```bibtex
@misc{uncertaintypose2025,
  author = {Sharma, Sumit},
  title = {Uncertainty-Aware Human Perception for Trustworthy HCI},
  year = {2025}
}
```

## License

[Specify your license here]
