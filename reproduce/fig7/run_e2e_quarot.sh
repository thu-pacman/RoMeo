logfilename="bench_e2e_quarot_$(date +%Y%m%d_%H%M%S).log"

batch_list=(16 64 256)

seqlen=128

all_models=(
    "/home/dataset/Qwen3-8B"
    "/home/dataset/Qwen3-14B"
    "/home/dataset/Qwen3-32B"
)

for model in "${all_models[@]}"; do
    for batch in "${batch_list[@]}"; do
        echo "============================================" | tee -a "$logfilename"
        echo "Running benchmark with model=$model, batch=$batch" | tee -a "$logfilename"
        echo "============================================" | tee -a "$logfilename"
        
        python3 benchmark/bench_module.py \
            --model "$model" \
            --nbatch "$batch" \
            --seqlen "$seqlen" \
            --baseline quarot 2>&1 | tee -a "$logfilename"
    done
done

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "============================================" | tee -a "$logfilename"
echo "Benchmark completed at: $end_time" | tee -a "$logfilename"
echo "Results saved to: $logfilename" | tee -a "$logfilename"
echo "============================================" | tee -a "$logfilename"
