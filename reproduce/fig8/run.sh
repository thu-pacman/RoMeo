if [ -f bench_kernels_results.pdf ]; then
    echo "bench_kernels_results.pdf exists"
    exit 0
fi

source ../../scripts/activate_env.sh ../../.venv

export HF_ENDPOINT=https://hf-mirror.com
export NO_USE_FASTER_HADAMARD_TRANSFORM=1

srun --pty --gres=gpu:4090:1 -c 32 -p Long python3 ./bench_kernels.py
python3 ./plot_bench_kernels_results.py --file bench_kernels_results

cd ../../
deactivate