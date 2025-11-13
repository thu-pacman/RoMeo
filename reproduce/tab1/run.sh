source ../../scripts/activate_env.sh ../../.venv

rm -rf ./*.log
export HF_ENDPOINT=https://hf-mirror.com
export NO_USE_FASTER_HADAMARD_TRANSFORM=1
task=perplexity model=Qwen3-8B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=perplexity model=Qwen3-14B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=perplexity model=Qwen3-32B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=perplexity model=Llama-3.1-8B srun --gres=gpu:H100:1 -p Long bash ./run_acc_allbench.sh
task=perplexity model=Llama-3.1-70B srun --gres=gpu:H100:2 -p Long bash ./run_acc_allbench.sh
python3 ./read_perplexity.py --file_path perplexity_Qwen3-8B.log,perplexity_Qwen3-14B.log,perplexity_Qwen3-32B.log,perplexity_Llama-3.1-8B.log,perplexity_Llama-3.1-70B.log

cd ../../
deactivate
