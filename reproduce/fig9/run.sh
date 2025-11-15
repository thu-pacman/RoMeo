source ../../scripts/activate_env.sh ../../.venv

rm -rf ./*.log
export HF_ENDPOINT=https://hf-mirror.com
export NO_USE_FASTER_HADAMARD_TRANSFORM=1

batch=16 source ./run_breakdown_allbench.sh
batch=64 source ./run_breakdown_allbench.sh
python3 ./plot_breakdown.py --file1 bench_breakdown_Qwen3-8B_b16.log --file2 bench_breakdown_Qwen3-8B_b64.log

cd ../../
deactivate