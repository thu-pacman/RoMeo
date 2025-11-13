if [ -z "$model" ]; then
    model="/home/dataset/Qwen3-8B"
fi

if [ -z "$seqlen" ]; then
    seqlen=128
fi

model_name=$(basename "$model")
logfilename="bench_e2e_${model_name}_$(date +%Y%m%d_%H%M%S).log"

batch_list=(16 64 256)

baseline_flags=(
    "--baseline bf16"
    "--baseline bitweaver --multistream"
    # "--baseline bitweaver --unifiedkernel"
)

for baseline_flag in "${baseline_flags[@]}"; do
    for batch in "${batch_list[@]}"; do
        echo "============================================" | tee -a "$logfilename"
        echo "Running benchmark with batch=$batch, baseline_flag=$baseline_flag" | tee -a "$logfilename"
        echo "============================================" | tee -a "$logfilename"
        
        python3 benchmark/bench_module.py \
            --model "$model" \
            --nbatch "$batch" \
            --seqlen "$seqlen" \
            $baseline_flag 2>&1 | tee -a "$logfilename"
    done
done

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "============================================" | tee -a "$logfilename"
echo "Benchmark completed at: $end_time" | tee -a "$logfilename"
echo "Results saved to: $logfilename" | tee -a "$logfilename"
echo "============================================" | tee -a "$logfilename"

cat "$logfilename" | grep Layer | awk -F'|' '{
    gsub(/ms/,"",$1); gsub(/Layer /,"",$1);
    gsub(/ms/,"",$2); gsub(/attn /,"",$2);
    gsub(/ms/,"",$3); gsub(/gemm /,"",$3);
    gsub(/ms/,"",$4); gsub(/quant /,"",$4);
    gsub(/ms/,"",$5); gsub(/post_mul /,"",$5);
    gsub(/ms/,"",$6); gsub(/hadamard /,"",$6);
    print $1","$2","$3","$4","$5","$6
}' | column -s',' -t | tee -a "$logfilename"
