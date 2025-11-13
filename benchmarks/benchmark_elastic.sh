#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <N> <K>"
    exit 1
fi

n=$1
k=$2

div_values=(1 2 4 8)

for d in "${div_values[@]}"; do
    log_file="log_gemm_N${n}_K${k}_div${d}.log"
    
    echo "Running: N=$n, K=$k, div=$d => $log_file"
    
    QFACTORY_LOG_LEVEL=ERROR QFACTORY_FAST_PROFILE=1 srun --gres=gpu:a100:1 -p a100 -c 16 \
        python benchmarks/benchmark_gemm_elastic.py \
        --n "$n" \
        --k "$k" \
        --outlier 256 \
        --div "$d" > "$log_file"
done

echo "All benchmark jobs done for N=$n, K=$k!"
