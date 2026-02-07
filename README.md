# Uncertainty-Aware Human Perception for Trustworthy HCI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-HCI%20%2B%20ML-green.svg)](https://github.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Research-76B900.svg)](https://www.nvidia.com/research/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)](https://github.com)
[![COCO](https://img.shields.io/badge/Dataset-COCO-red.svg)](https://cocodataset.org/)
[![MPII](https://img.shields.io/badge/Dataset-MPII-purple.svg)](http://human-pose.mpi-inf.mpg.de/)
[![DOI](https://img.shields.io/badge/DOI-Pending-lightgrey.svg)](https://github.com)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

### 🎯 Project Highlights

- ✅ **Uncertainty Decomposition**: Separate aleatoric (data noise) from epistemic (model uncertainty)
- ✅ **Calibrated Predictions**: Post-hoc calibration for reliable confidence estimates
- ✅ **HCI Integration**: Adaptive UI behaviors based on uncertainty thresholds
- ✅ **Efficient**: Runs on RTX 3050 (consumer GPU)
- ✅ **Reproducible**: Complete code, configs, and evaluation pipeline

---

**Project short name:** UncertaintyPose  
**Author:** Sumit Sharma
**Goal:** Produce a research-grade implementation and evaluation that demonstrates calibrated, decomposed uncertainty (aleatoric + epistemic) for keypoint/pose estimation and shows how uncertainty can be used by HCI systems to improve decision outcomes.

## Abstract

> **Note:** This README contains mathematical equations in LaTeX format. For best viewing experience on GitHub, use a browser extension like [MathJax Plugin for GitHub](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima) or view the rendered documentation.

Standard human pose and landmark predictors produce point estimates with a scalar "confidence" that is often miscalibrated and uninformative under occlusion, motion blur, sensor noise, or domain shift. This project develops a principled pipeline that:

1. Decomposes uncertainty into aleatoric and epistemic components
2. Produces calibrated probabilistic predictions for keypoints
3. Demonstrates HCI utility via adaptive UI behaviors and a small user/simulated study

**Deliverables:** Reproducible code, figures (calibration curves, AUROC, ablation tables), reproducible demo, and a short research report.

---

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#1-problem-statement-core-depth)
- [Key Claims to Test](#2-key-claims-to-test)
- [Theoretical Background](#3-theoretical-background-concise-testable)
- [Related Methods](#4-related-methods-and-limitations-what-others-do-now)
- [Proposed Method](#5-proposed-method-architecture--math)
- [Implementation Plan](#6-implementation-plan-code--practical-tricks)
- [Evaluation Protocol](#7-evaluation-protocol-and-metrics-research-proof)
- [Expected Results](#8-expected-results-benchmarks-to-aim-for)
- [Repository Structure](#9-reproducibility--repository-structure)
- [Quick Start](#quick-start)
- [Citation](#citation)

---

## 📊 Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT IMAGE (x)                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Backbone f_θ(x)     │
         │   (MobileNetV3/HRNet) │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    ┌─────────┐           ┌──────────────┐
    │ Mean    │           │ Aleatoric    │
    │ Head    │           │ Variance     │
    │ (ŷ)     │           │ Head (σ²_a)  │
    └────┬────┘           └──────┬───────┘
         │                       │
         │    MC Dropout (T=8)   │
         │    ────────────►      │
         │    Epistemic (σ²_e)   │
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ σ²_pred = σ²_a + σ²_e │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   CalibrationNet      │
         │   g_φ(ŷ, σ²_a, σ²_e)  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Error Probability     │
         │      p_err            │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  HCI Decision Layer   │
         │  • auto (p_err ≤ τ₁)  │
         │  • ask  (τ₁ < p_err)  │
         │  • safe (p_err > τ₂)  │
         └───────────────────────┘
```

---

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

Let $y$ denote true keypoint coordinates and $\hat{y}$ the model prediction. We model predictive uncertainty as:

$$
\text{Var}(y|x) = \underbrace{\mathbb{E}[\text{Var}(y|x,\theta)]}_{\text{aleatoric } (\sigma_a^2)} + \underbrace{\text{Var}(\mathbb{E}[y|x,\theta])}_{\text{epistemic } (\sigma_e^2)}
$$

where $\theta$ are model parameters and $x$ the input image. 

- **Aleatoric uncertainty** $\sigma_a^2$ captures irreducible noise (occlusion, motion blur)
- **Epistemic uncertainty** $\sigma_e^2$ captures model uncertainty (lack of data, OOD)

### Aleatoric Modeling (Heteroscedastic NLL)

For a single keypoint modeled as Gaussian:

$$
\mathcal{L}_{\text{ale}} = \frac{1}{2\sigma_a^2}\|\hat{y} - y\|^2 + \frac{1}{2}\log(\sigma_a^2)
$$

This follows Kendall & Gal (heteroscedastic regression): predicting $\sigma_a^2$ allows the network to downweight noisy labels.

### Epistemic Estimation (MC Dropout / Ensembles)

Use $T$ stochastic forward passes (dropout active) producing $\hat{y}^{(t)}$. Estimate:

$$
\begin{aligned}
\hat{\mu} &= \frac{1}{T}\sum_{t=1}^{T} \hat{y}^{(t)} \\
\hat{\sigma}_e^2 &= \frac{1}{T}\sum_{t=1}^{T} \|\hat{y}^{(t)} - \hat{\mu}\|^2
\end{aligned}
$$

### Calibration and Expected Calibration Error (ECE)

For binned predicted confidence/probability $p$ and observed accuracy $a$:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|
$$

adapted for continuous error via thresholds on OKS/PCK.

### Decision Rule for HCI

Given predicted error probability $p_{\text{err}}$ from calibration, interface selects action $A \in \{\text{auto}, \text{ask}, \text{safe-mode}\}$ by thresholding:

$$
A(x) = \begin{cases}
\text{auto} & \text{if } p_{\text{err}} \leq \tau_1 \\
\text{ask} & \text{if } \tau_1 < p_{\text{err}} \leq \tau_2 \\
\text{safe-mode} & \text{if } p_{\text{err}} > \tau_2
\end{cases}
$$

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

Input image $x$ → lightweight backbone $f_\theta(x)$ → three heads:

1. **Mean heatmap / keypoint head** $h_\mu$ → produces per-keypoint heatmaps / coordinates $\hat{y}$
2. **Aleatoric head** $h_{\sigma_a}$ → produces per-keypoint variance $\hat{\sigma}_a^2$ (or per-pixel variance for heatmaps)
3. **Optional attention / explanation head** $h_{\text{attn}}$ → saliency map to explain uncertainty

At inference: run $T$ stochastic forward passes (dropout) to compute epistemic variance $\hat{\sigma}_e^2$. Aggregate into total predictive variance:

$$
\hat{\sigma}_{\text{pred}}^2 = \hat{\sigma}_a^2 + \hat{\sigma}_e^2
$$

### 5.2 Loss Function

Total loss for training on dataset $\mathcal{D}$:

$$
\mathcal{L} = \mathcal{L}_{\text{heat}} + \lambda_{\text{ale}} \cdot \mathcal{L}_{\text{ale}} + \lambda_{\text{reg}}\|\theta\|^2
$$

where:
- $\mathcal{L}_{\text{heat}}$ is standard heatmap MSE / focal loss for mean heatmap
- $\mathcal{L}_{\text{ale}} = \sum_{k} \left(\frac{\|\hat{y}_k - y_k\|^2}{2\hat{\sigma}_{a,k}^2} + \frac{1}{2}\log(\hat{\sigma}_{a,k}^2)\right)$ summed over keypoints $k$
- $\lambda_{\text{ale}}$ balances aleatoric training

Epistemic variance is not directly in the loss; encourage epistemic reduction via data augmentation and regular training (ensembles or dropout approximate posterior uncertainty).

### 5.3 CalibrationNet

Train a small calibration network $g_\phi$ that inputs $(\hat{y}, \hat{\sigma}_a^2, \hat{\sigma}_e^2)$ and outputs a calibrated error probability $p_{\text{err}}$. Optimize cross-entropy vs binary label $\mathbb{1}(\|\hat{y} - y\| > \epsilon)$ or regression vs continuous error with ECE reduction objective.

Loss for calibration:

$$
\mathcal{L}_{\text{cal}} = \text{BCE}(g_\phi(\cdot), \mathbb{1}(\text{error} > \epsilon)) + \lambda_{\text{ece}} \cdot \text{ECE}(g_\phi)
$$

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
- **MC passes** $T=8$ for inference epistemic estimate (tunable). For faster experimentation use $T=4$

### 6.3 Datasets

- **Primary:** COCO Keypoints subset for initial experiments
- **Secondary:** MPII, and synthetic occlusion dataset (apply randomized masks) to test robustness
- **Small held-out OOD test set** (different scenes / lighting) for epistemic evaluation

### 6.4 HCI Demo

- Simple web UI or Jupyter demo that overlays predicted keypoints and visualizes per-keypoint uncertainty (opacity, blur radius, or heatmap spread)
- Adaptive behavior simulation: automated action triggered only when $p_{\text{err}} < \tau$; otherwise show a confirm dialog

## 7. Evaluation Protocol and Metrics (Research Proof)

### 7.1 Perceptual Accuracy

- OKS / PCK / mAP on standard validation sets

### 7.2 Uncertainty Quality

**Negative Log-Likelihood (NLL)** under predicted Gaussian distributions:

$$
\text{NLL} = -\sum_{k} \log \mathcal{N}(y_k \mid \hat{y}_k, \hat{\sigma}_{\text{pred},k}^2)
$$

**Expected Calibration Error (ECE):** Bin predicted error probabilities and compute calibration gap (adapted to continuous error via thresholds)

**AUROC:** Use uncertainty to classify whether $\|\hat{y} - y\| > \delta$ (failure) — AUROC quantifies detection power

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

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/uncertainty-pose.git
cd uncertainty-pose

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### ⚡ Quick Demo (5 minutes)

```bash
# Run quick demo experiment (finishes in ~2 hours on RTX 3050)
bash run_demo.sh

# Open interactive demo
jupyter notebook notebooks/demo.ipynb
```

### 🎓 Training

```bash
# Train with default config (small model)
python src/train.py --config configs/small.yaml

# Train full model
python src/train.py --config configs/full.yaml --gpu 0

# Resume from checkpoint
python src/train.py --config configs/full.yaml --resume checkpoints/latest.pth
```

### 📈 Evaluation

```bash
# Evaluate uncertainty quality
python src/eval.py --checkpoint results/best_model.pth --metrics all

# Run HCI simulation
python notebooks/hci_simulation.ipynb

# Generate calibration plots
python src/eval.py --checkpoint results/best_model.pth --plot-calibration
```

---

## 📊 Performance Targets

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| **ECE Reduction** | - | 30-60% ↓ | 🎯 In Progress |
| **AUROC (Failure Detection)** | ~0.60 | ≥0.75 | 🎯 In Progress |
| **OKS/PCK** | Competitive | Match SOTA | 🎯 In Progress |
| **Inference Time** | - | <100ms (T=8) | 🎯 In Progress |
| **GPU Memory** | - | <4GB (RTX 3050) | ✅ Achieved |

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{uncertaintypose2025,
  author       = {Sharma, Sumit},
  title        = {Uncertainty-Aware Human Perception for Trustworthy HCI},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/uncertainty-pose}},
  note         = {Research-grade implementation of calibrated uncertainty decomposition for pose estimation}
}
```

### Related Publications

This work builds upon:

- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision?
- Guo, C., et al. (2017). On calibration of modern neural networks.
- [Add your publications here]

---

## 📞 Contact

**Author**: Sumit Sharma  
**Email**: [your.email@example.com]  
**Project Link**: [https://github.com/your-username/uncertainty-pose](https://github.com/your-username/uncertainty-pose)

For questions, issues, or collaboration opportunities, please:
- 🐛 Open an [issue](https://github.com/your-username/uncertainty-pose/issues)
- 💬 Start a [discussion](https://github.com/your-username/uncertainty-pose/discussions)
- 📧 Email directly for research collaborations

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/uncertainty-pose&type=Date)](https://star-history.com/#your-username/uncertainty-pose&Date)

---

## 🔧 System Requirements

### Hardware
- **Minimum**: NVIDIA RTX 3050 (4GB VRAM) or equivalent
- **Recommended**: NVIDIA RTX 3080 (10GB VRAM) or higher
- **RAM**: 16GB+ recommended
- **Storage**: 10GB for datasets + models

### Software
- Python 3.8 or higher
- CUDA 11.0+ and cuDNN
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

---

## 📚 Key Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
albumentations>=1.3.0
numpy>=1.23.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
tensorboard>=2.12.0
scikit-learn>=1.2.0
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- 🐛 Bug fixes and performance improvements
- 📊 Additional evaluation metrics
- 🎨 Visualization enhancements
- 📖 Documentation improvements
- 🧪 New datasets or augmentation strategies

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- COCO Dataset team for benchmark data
- MPII Human Pose Dataset contributors
- Kendall & Gal for uncertainty quantification foundations
- NVIDIA Research for infrastructure support

---
