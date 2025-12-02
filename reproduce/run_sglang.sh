LOG_DIR="logs"
mkdir -p "$LOG_DIR"

export QFACTORY_LOG_LEVEL=ERROR
export QFACTORY_ARCH=89
export QFACTORY_FAST_PROFILE=1
export SGLANG_TORCH_PROFILER_DIR="profile_log"

BATCH_SIZES=(8 16 32 64 128)
INPUT_LENS=(128)
OUTPUT_LEN=1

MODELS=(
  "Qwen3-8B|1|$HOME/datasets/Qwen3-8B/"
  "Qwen3-14B|2|$HOME/datasets/Qwen3-14B/"
  "Qwen3-32B|4|$HOME/datasets/Qwen3-32B/"
)

QUANTS=(romeo unquantized)

for SPEC in "${MODELS[@]}"; do
  IFS='|' read -r NAME TP MPATH <<< "$SPEC"
  for B in "${BATCH_SIZES[@]}"; do
    for IL in "${INPUT_LENS[@]}"; do
      for Q in "${QUANTS[@]}"; do
        SUFFIX="$Q"
        LOG_FILE="$LOG_DIR/${NAME}_tp${TP}_b${B}_in${IL}_${SUFFIX}.log"
        if [[ -f "$LOG_FILE" ]]; then
          echo "[SKIP] exists: $LOG_FILE"
          continue
        fi

        echo "==== $(date) | ${NAME} tp=${TP} batch=${B} in=${IL} quant=${Q} ====" | tee -a "$LOG_FILE"

        ARGS=(
          --model-path "$MPATH"
          --load-format dummy
          --batch "$B"
          --input-len "$IL"
          --output-len "$OUTPUT_LEN"
          --disable-cuda-graph
          --tp "$TP"
        )
        if [[ "$Q" == "romeo" ]]; then
          ARGS+=(--quantization romeo)
        fi

        python -m sglang.bench_one_batch "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
      done
    done
  done
done

echo "Logs saved to: $LOG_DIR"
