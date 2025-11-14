# RoMeo-AE
Artifact for PPoPP'26 "RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization"

## C. Prepare Software Environment

**For AE reviewers, please skip this step and use the provided environment.**

### C1. Prepare codebase

Download this repository and its submodules:

```bash
git clone --recursive https://github.com/zqh-wz/RoMeo-AE.git
cd RoMeo-AE/
```

Download val.jsonl for SmoothQuant:

```bash
wget https://hf-mirror.com/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst
zstd -d --rm val.jsonl.zst
```

Then, apply nessesary patches to submodules.

```bash
cd third_party/omniserve
git apply ../patches/omniserve.patch
cd ../../

cd third_party/cutlass
git apply ../patches/cutlass.patch
cd ../../

cd third_party/fast-hadamard-transform
git apply ../patches/fast-hadamard-transform.patch
cd ../../

cd third_party/QuaRot
git apply ../patches/QuaRot.patch
cd ../../
```

### C2. Installation

We manage python virtual environments with `uv`.

```bash
bash ./scripts/create_env.sh .venv
source ./scripts/activate_env.sh .venv
source ./scripts/install.sh

deactivate

bash ./scripts/create_env.sh .venv_qfactory
source ./scripts/activate_env.sh .venv_qfactory
source ./scripts/install_qfactory.sh

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

#### Figure 7: Normalized layer-level latency on Qwen3 models of different input batch sizes.

```bash
source ./scripts/reproduce.sh fig7
```

#### Figure 8: Normalized kernel performance on various matrix shapes.

```bash
source ./scripts/reproduce.sh fig8
```

#### Figure 9. Layer-level latency breakdown for Qwen3-8B across different batch sizes with progressive optimizations.

```bash
source ./scripts/reproduce.sh fig9
```

#### Figure 10. Scaling the percentage of outliers.

```bash
source ./scripts/reproduce.sh fig10
```