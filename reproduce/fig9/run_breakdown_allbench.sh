if [ -z "$model" ]; then
    model="/home/dataset/Qwen3-8B"
fi

if [ -z "$batch" ]; then
    batch=64
fi

if [ -z "$seqlen" ]; then
    seqlen=128
fi

model_name=$(basename "$model")
logfilename="bench_breakdown_${model_name}_b${batch}.log"

basic_flags="--model ${model} --nbatch ${batch} --seqlen ${seqlen}"

export QFACTORY_LOG_LEVEL=ERROR
export QFACTORY_ARCH=89
export QFACTORY_FAST_PROFILE=1

srun --gres=gpu:4090:1 -c 32 -p Long \
    python3 ../../bench_module.py $basic_flags \
    --baseline bf16 2>&1 | tee -a "$logfilename"

QFACTORY_NO_PIPELINE=1 srun --gres=gpu:4090:1 -c 32 -p Long \
    python3 ../../bench_module.py $basic_flags \
    --baseline bitweaver --unifiedkernel 2>&1 | tee -a "$logfilename"

srun --gres=gpu:4090:1 -c 32 -p Long \
    python3 ../../bench_module.py $basic_flags \
    --baseline bitweaver --unifiedkernel 2>&1 | tee -a "$logfilename"

QFACTORY_NO_PIPELINE=1 srun --gres=gpu:4090:1 -c 32 -p Long \
    python3 ../../bench_module.py $basic_flags \
    --baseline bitweaver 2>&1 | tee -a "$logfilename"

srun --gres=gpu:4090:1 -c 32 -p Long \
    python3 ../../bench_module.py $basic_flags \
    --baseline bitweaver 2>&1 | tee -a "$logfilename"

srun --gres=gpu:4090:1 -c 32 -p Long \
    python3 ../../bench_module.py $basic_flags \
    --baseline bitweaver --multistream 2>&1 | tee -a "$logfilename"

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
