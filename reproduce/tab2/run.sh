source ../../scripts/activate_env.sh ../../.venv

rm -rf ./*.log
export HF_ENDPOINT=https://hf-mirror.com
task=zero-shot batch=32 model=Qwen3-8B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=zero-shot batch=32 model=Qwen3-14B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=zero-shot batch=32 model=Qwen3-32B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=zero-shot batch=32 model=Llama-3.1-8B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=zero-shot batch=32 model=Llama-3.1-70B srun --gres=gpu:H100:2 -p Long bash ./run_acc_allbench.sh


cd ../../
deactivate