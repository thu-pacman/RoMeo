source ../../scripts/activate_env.sh ../../.venv

rm -rf ./*.log
export HF_ENDPOINT=https://hf-mirror.com
export NO_USE_FASTER_HADAMARD_TRANSFORM=1
task=perplexity model=Qwen3-8B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=perplexity model=Llama-3.1-8B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
model=/home/dataset/Qwen3-8B bash ./run_acc_scale.sh
model=/home/dataset/Llama-3.1-8B bash ./run_acc_scale.sh


cd ../../
deactivate
