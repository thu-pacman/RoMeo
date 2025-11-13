export QFACTORY_LOG_LEVEL=ERROR
export QFACTORY_ARCH=89
export QFACTORY_FAST_PROFILE=1

model=/home/dataset/Qwen3-8B srun --pty --gres=gpu:4090:1 -c 32 -p Long bash scripts/run_e2e_allbench.sh
model=/home/dataset/Qwen3-14B srun --pty --gres=gpu:4090:1 -c 32 -p Long bash scripts/run_e2e_allbench.sh
model=/home/dataset/Qwen3-32B srun --pty --gres=gpu:4090:1 -c 32 -p Long bash scripts/run_e2e_allbench.sh
