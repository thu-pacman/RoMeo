# RoMeo-AE
Artifact for PPoPP'26 "RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization"

## C. Prepare Software Environment

**For AE reviewers, please skip this step and use the provided environment.**

### C1. Prepare codebase

Download this repository and its submodules:

```bash
git clone --recursive https://github.com/zqh-wz/RoMeo-AE.git
git submodule update --init --recursive
```

### C2. Installation

We manage python virtual environments with `uv`.

```bash
bash ./scripts/create_env.sh .venv
source ./scripts/activate_env.sh .venv
source ./scripts/install.sh

deactivate
```

## D. Reproduce Experimental Results

#### Table 1: Comparison of measured perplexity on WikiText2 dataset.

```bash
source ./scripts/reproduce.sh tab1
```

#### Table 2: Comparison of zero-shot accuracy on four downstream tasks.

```bash
source ./scripts/reproduce.sh tab2
```