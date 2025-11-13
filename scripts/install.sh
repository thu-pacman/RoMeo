#!/bin/bash

# Install dependencies
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install fast-hadamard-transform 
cd third_party/fast-hadamard-transform
uv pip install . --no-build-isolation
cd ../../

# Install faster-hadamard-transform
cd third_party/applied-ai/kernels/cuda/inference/hadamard_transform/
uv pip install . --no-build-isolation
cd ../../../../../../

# Install QuaRot
cd third_party/QuaRot/
uv pip install . --no-build-isolation
cd ../../

# Install qfactory
uv pip install . --no-build-isolation