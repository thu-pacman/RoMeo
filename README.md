# RoMeo-AE

## A. Abstract

This repository contains the code for the reproduction of the paper "RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization" at PPoPP'26.

The reproduction includes Tables 1 and 2 and Figures 7, 8, 9, and 10 from the submitted version of the paper.

## B. Prepare Hardware Environment

To reproduce this work, a GPU server with NVIDIA 4090 and H100 GPUs is required.

**For AE Reviewers, please check the HotCRP website for instructions on how to access the provided GPU servers.**

To avoid issues of environment and network, we strongly recommend reviewers to use our provided environment.


## C. Prepare Software Environment

**For AE reviewers, please skip this step and use the provided environment.**

### C1. Prepare codebase

Download this repository and its submodules:

```bash
git clone --recursive https://github.com/zqh-wz/RoMeo-AE.git
cd RoMeo-AE/
```

Download calibration dataset for SmoothQuant:

```bash
wget https://hf-mirror.com/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst
zstd -d --rm val.jsonl.zst
```

Then, apply nessesary patches to submodules.

```bash
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
bash ./scripts/install.sh
deactivate
```

## D. Reproduce Experimental Results

#### Table 1: Comparison of measured perplexity on WikiText2 dataset.

Estimated runtime: ~90 minutes.

```bash
bash ./scripts/reproduce.sh tab1 | tee tab1.log
```

Result summary will be generated at `reproduce/tab1/perplexity_summary.log`.

#### Table 2: Comparison of zero-shot accuracy on four downstream tasks.

Estimated runtime: ~45 minutes.

```bash
bash ./scripts/reproduce.sh tab2 | tee tab2.log
```

Result summary will be generated at `reproduce/tab2/zero_shot_summary.log`.

> Note: Due to the long runtime of full zero-shot evaluation, this script only runs partial evaluation (partial models and benchmarks) for quick verification.
> 
> To run the full evaluation, simply modify `reproduce/tab2/run.sh` and `reproduce/tab2/run_acc_allbench.sh`.

#### Figure 7: Normalized layer-level latency on Qwen3 models of different input batch sizes.

Estimated runtime: ~25 minutes.

```bash
bash ./scripts/reproduce.sh fig7 | tee fig7.log
```

Result figure will be generated at `reproduce/fig7/layer_latency.pdf`.

#### Figure 8: Normalized kernel performance on various matrix shapes.

Estimated runtime: ~25 minutes.

```bash
bash ./scripts/reproduce.sh fig8 | tee fig8.log
```

Result figure will be generated at `reproduce/fig8/bench_kernels_results.pdf`.

#### Figure 9. Layer-level latency breakdown for Qwen3-8B across different batch sizes with progressive optimizations.

Estimated runtime: ~5 minutes.

```bash
bash ./scripts/reproduce.sh fig9 | tee fig9.log
```

Result figure will be generated at `reproduce/fig9/plot_breakdown.pdf`.

#### Figure 10. Scaling the percentage of outliers.

Estimated runtime: ~50 minutes.

```bash
bash ./scripts/reproduce.sh fig10 | tee fig10.log
```

Result figure will be generated at `reproduce/fig10/percent-ppl.pdf`.
