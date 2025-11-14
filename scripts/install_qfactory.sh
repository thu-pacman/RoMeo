#!/bin/bash

# Install dependencies
uv pip install torch==2.8.0 torchvision torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
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
uv pip install -e . --no-build-isolation
cd ../../

# Install Qfactory
uv pip install . --no-build-isolation

# # Install omniserve
# cd third_party/omniserve
# uv pip install .
# cd ../../