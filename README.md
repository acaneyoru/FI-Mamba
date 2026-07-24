# FI-Mamba

This repository contains the official implementation of the paper:

**"FI-Mamba: Frequency-Domain Lossless Mamba with Prior Collaboration, Isotropic Unbiased State-Space Learning and Multi-Scale CLIP for Perinatal Brain Ultrasound Image Classification"**

## Overview

Perinatal brain ultrasound classification is challenging because subtle ventricular abnormalities coexist with speckle noise, weak boundaries, view-dependent appearance, and strong left-right anatomical symmetry. FI-Mamba addresses these factors with four coupled components:

1. **Frequency-Prior Collaborative Module (FPCModule).** A reconstructable Laplacian-residual decomposition separates low-frequency anatomy and high-frequency detail without discarding either component. A local difference prior emphasizes diagnostically relevant boundaries.
2. **Isotropic Unbiased Scan (IU-Scan).** Forward and reverse selective state-space updates are adaptively fused to reduce the directional bias caused by flattening a two-dimensional image into a one-dimensional token sequence.
3. **Orthogonal Frequency Spectrum Gating Block (OFSG-Block).** The complete spectrum is organized into three complementary soft bands whose masks sum to one. Multi-window local amplitudes and content-adaptive gates suppress non-diagnostic responses while retaining a residual information path.
4. **Multi-scale multi-head CLIP-interaction.** CLIP-style normalized cosine interaction is reinterpreted as a pure-vision operation. It aligns low-frequency, high-frequency, and prior branches across four encoder stages without requiring a text encoder or paired reports.

## Requirements

- Python 3.10+
- PyTorch 2.1+
- torchvision 0.16+
- Linux or Windows
- CUDA-capable GPU recommended for training

Install the minimal dependencies:

```bash
git clone https://github.com/acaneyoru/FIMamba.git
cd FIMamba
pip install -r requirements.txt
```

## Quick start

Run the built-in shape demonstration:

```bash
python FI-Mamba.py --num-classes 4 --image-size 256 --batch-size 2
```

Because the requested filename contains a hyphen, load it from another Python script with `importlib`:

```python
import sys
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("fi_mamba", "FI-Mamba.py")
fi_mamba = module_from_spec(spec)
sys.modules[spec.name] = fi_mamba
spec.loader.exec_module(fi_mamba)

model = fi_mamba.build_fimamba(
    num_classes=4,
    class_counts=[139, 139, 139, 139],
)
```

The forward method returns a dictionary so that the classification output and the intermediate interaction weights can be inspected:

```python
output = model(images, targets=labels, epoch=120, max_epochs=300)

logits = output["logits"]
probabilities = output["probabilities"]
loss = output["loss"]
branch_weights = output["branch_weights"]
scale_weights = output["scale_weights"]
```

## Architecture configuration

| Component | Default setting in `FI-Mamba.py` |
|---|---|
| Input | Grayscale image, `1 × 256 × 256` |
| FPC branches | Low frequency, high-frequency residual, local prior |
| Encoder stages | 4 |
| Stage dimensions | 48, 96, 192, 384 |
| Stage depths | 2, 2, 4, 2 |
| IU-Scan | Shared forward/reverse selective SSM with learned fusion |
| OFSG bands | 3 soft, complementary radial bands |
| Local amplitude windows | 1, 3, 5, 7 |
| Interaction head | Within-scale branch fusion followed by cross-scale fusion |
| Default classes | 4 |
| Objective | Class-balanced cross-entropy + cosine alignment loss |

The dimensions above are an explicit reference configuration. Reported parameter count and latency depend on the exact state-space kernel, token compaction strategy, and deployment environment.

## Frequency-domain lossless design

FI-Mamba does not use a fixed Fourier or wavelet filter to cut off a frequency interval. The FPCModule first constructs

\[
X_{\mathrm{low}}=\operatorname{Up}\!\left(\operatorname{AvgPool}(X)\right),
\qquad
X_{\mathrm{high}}=X-X_{\mathrm{low}},
\]

so the input feature is exactly reconstructable as

\[
X=X_{\mathrm{low}}+X_{\mathrm{high}}.
\]

The following OFSG-Block uses three soft spectral masks normalized by a softmax partition:

\[
\sum_{b=1}^{3}M_b(u,v)=1.
\]

All frequency branches remain available to the model and are recombined through content-adaptive gates and a residual connection. “Lossless” therefore refers to retaining a complete reconstructable frequency path rather than compressing files or claiming mathematically invertible nonlinear classification features.

The optional hard token mask is disabled by default. Zeroing tokens in a dense tensor does **not** reduce actual FLOPs; computation decreases only when selected tokens are physically packed and processed by a compatible sparse/variable-length kernel.

## Datasets

### Anterior Horn

- 139 neonates aged 1–7 days.
- 556 left/right images after subject-preserving preprocessing and mirroring.
- Four classes: left abnormal (`LAHD`), right abnormal (`RAHD`), left normal (`LAHN`), and right normal (`RAHN`).
- The manuscript reports approval by the institutional ethics committee (Fast2023013).

### Posterior Horn

- 278 parasagittal perinatal brain ultrasound images.
- Four classes: left abnormal (`LPHD`), right abnormal (`RPHD`), left normal (`LPHN`), and right normal (`RPHN`).

### Fetal Brain Biometry

- 3,832 fetal head ultrasound images from the public large-scale fetal head biometry annotation dataset.
- Four categories used in the manuscript: thalamic, ventricular, cerebellar, and original/other.
- Download the public dataset from its original provider and comply with the provider's license and citation requirements.

Private clinical images are not redistributed by this repository. A suggested local organization is:

```text
data/
├── AnteriorHorn/
│   ├── LAHD/
│   ├── LAHN/
│   ├── RAHD/
│   └── RAHN/
├── PosteriorHorn/
│   ├── LPHD/
│   ├── LPHN/
│   ├── RPHD/
│   └── RPHN/
└── FetalBrainBiometry/
    ├── thalamic/
    ├── ventricular/
    ├── cerebellar/
    └── original/
```

Keep all images from the same subject in the same fold to prevent patient-level leakage.

## Training protocol used in the manuscript

| Item | Setting |
|---|---|
| Framework | PyTorch |
| Input resolution | `256 × 256` |
| Optimizer | Adam |
| Initial learning rate | `1 × 10⁻⁴` |
| Weight decay | `1 × 10⁻⁵` |
| Batch size | 32 |
| Epochs | 300 |
| Augmentation | Random rotation, horizontal flip, brightness adjustment |
| Private datasets | Patient-level five-fold cross-validation |
| Public dataset | Five random seeds with a 6:2:2 train/validation/test split |
| Hardware reported | NVIDIA GeForce RTX 4090 |
| Statistical analysis | Stratified bootstrap (300 repeats) and paired Wilcoxon signed-rank test |

For class counts \(n_k\), the implementation uses effective-number weights

\[
w_k=\frac{1-\beta}{1-\beta^{n_k}},
\]

normalized across classes. The total objective is

\[
\mathcal{L}=\mathcal{L}_{\mathrm{CBCE}}
+\lambda(e)\mathcal{L}_{\mathrm{align}},
\]

where \(\lambda(e)\) increases gradually so that early optimization emphasizes classification and later optimization strengthens multi-branch and multi-scale alignment.

## Results reported in the manuscript

Values are mean ± standard deviation. They are manuscript results, not automatically reproduced by the current architecture-only release.

| Dataset | AUROC | AUPRC | OA | F1 | Youden index | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anterior Horn | 0.934 ± 0.010 | 0.737 ± 0.012 | 0.807 ± 0.013 | 0.581 ± 0.016 | 0.508 ± 0.015 | 0.589 ± 0.012 | 0.596 ± 0.014 |
| Posterior Horn | 0.703 ± 0.008 | 0.469 ± 0.010 | 0.596 ± 0.018 | 0.333 ± 0.012 | 0.149 ± 0.015 | 0.348 ± 0.017 | 0.395 ± 0.020 |
| Fetal Brain Biometry | 0.969 ± 0.004 | 0.904 ± 0.007 | 0.893 ± 0.010 | 0.881 ± 0.009 | 0.835 ± 0.012 | 0.879 ± 0.010 | 0.886 ± 0.011 |

| Parameters | FLOPs | Inference time |
|---:|---:|---:|
| 14.18 M | 3.79 G | 10.88 ms/image |

## Mapping from the paper pseudocode to the code

| Pseudocode operation | Implementation |
|---|---|
| Laplacian residual and local visual prior | `FrequencyPriorCollaborativeModule` |
| Forward/reverse state updates | `IsotropicUnbiasedScan` |
| Three-band spectrum and local amplitudes | `OrthogonalFrequencySpectrumGating` |
| Four-stage encoder | `IOFSMambaEncoder` |
| Within-scale and cross-scale cosine fusion | `MultiScaleCLIPInteraction` |
| Class-balanced classification | `ClassBalancedCrossEntropy` |
| Branch/scale alignment | `MultiScaleAlignmentLoss` |
| Complete network | `FIMamba` |

## Acknowledgments

This implementation is conceptually related to:

- [Mamba](https://github.com/state-spaces/mamba)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [CLIP](https://github.com/openai/CLIP)
- [MedMamba](https://github.com/YubiaoYue/MedMamba)

Questions and reproducibility issues can be submitted through the repository's [GitHub Issues](https://github.com/acaneyoru/FIMamba/issues).

