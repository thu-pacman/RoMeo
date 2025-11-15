source ../../scripts/activate_env.sh ../../.venv
spack load cuda@12.8
rm -rf ./*.log
export HF_ENDPOINT=https://hf-mirror.com
export NO_USE_FASTER_HADAMARD_TRANSFORM=1
export QFACTORY_LOG_LEVEL=ERROR
export QFACTORY_ARCH=89
export QFACTORY_FAST_PROFILE=1

model=/home/dataset/Qwen3-8B srun --pty --gres=gpu:4090:1 -c 32 -p Long bash ./run_e2e_allbench.sh
model=/home/dataset/Qwen3-14B srun --pty --gres=gpu:4090:1 -c 32 -p Long bash ./run_e2e_allbench.sh
model=/home/dataset/Qwen3-32B srun --pty --gres=gpu:4090:1 -c 32 -p Long bash ./run_e2e_allbench.sh
srun --pty --gres=gpu:4090:1 -c 32 -p Long bash ./run_e2e_quarot.sh

python3 ./plot_layer_latency.py

cd ../../
deactivate