export QFACTORY_LOG_LEVEL=ERROR
export QFACTORY_ARCH=89
export QFACTORY_FAST_PROFILE=1

export SGLANG_TORCH_PROFILER_DIR=~/profile_log


python -m sglang.bench_one_batch --model-path ~/datasets/Qwen3-8B/ --load-format dummy --batch 64 --input-len 128 --output-len 1 --disable-cuda-graph --quantization romeo
python -m sglang.bench_one_batch --model-path ~/datasets/Qwen3-8B/ --load-format dummy --batch 64 --input-len 128 --output-len 1 --disable-cuda-graph
python -m sglang.bench_one_batch --model-path ~/datasets/Qwen3-14B/ --tp 2 --load-format dummy --batch 64 --input-len 128 --output-len 1 --disable-cuda-graph --quantization romeo
python -m sglang.bench_one_batch --model-path ~/datasets/Qwen3-14B/ --tp 2 --load-format dummy --batch 64 --input-len 128 --output-len 1 --disable-cuda-graph
python -m sglang.bench_one_batch --model-path ~/datasets/Qwen3-32B/ --tp 4 --load-format dummy --batch 64 --input-len 128 --output-len 1 --disable-cuda-graph --quantization romeo
python -m sglang.bench_one_batch --model-path ~/datasets/Qwen3-32B/ --tp 4 --load-format dummy --batch 64 --input-len 128 --output-len 1 --disable-cuda-graph
