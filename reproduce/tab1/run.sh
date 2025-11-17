source ../../scripts/activate_env.sh ../../.venv

export HF_ENDPOINT=https://hf-mirror.com
export NO_USE_FASTER_HADAMARD_TRANSFORM=1

models=("Qwen3-8B" "Qwen3-14B" "Qwen3-32B" "Llama-3.1-8B" "Llama-3.1-70B")

for model in "${models[@]}"; do
    log_file="perplexity_${model}.log"

    if [ "$model" == "Llama-3.1-70B" ];
    then
        num_gpus=2
    else
        num_gpus=1
    fi

    if [ ! -f ./"$log_file" ]; then
        task=perplexity model=$model srun --gres=gpu:H100:$num_gpus -p Long bash ./run_acc_allbench.sh
    else
        echo "$log_file exists, skipping..."
    fi
done

python3 ./read_perplexity.py --file_path perplexity_Qwen3-8B.log,perplexity_Qwen3-14B.log,perplexity_Qwen3-32B.log,perplexity_Llama-3.1-8B.log,perplexity_Llama-3.1-70B.log | tee perplexity_summary.log

cd ../../
deactivate
