if [ -z "$model" ]; then
    model="Qwen3-8B"
fi

if [ -z "$batch" ]; then
    batch=2
fi

if [ -z "$task" ]; then
    echo "Error: The 'task' environment variable must be set."
    exit 1
fi

case "$task" in
    perplexity)
        basic_flags="--batch-size $batch --eval-perplexity"
        ;;
    zero-shot)
        basic_flags="--batch-size $batch --eval-zero-shot --eval-zero-shot-dataset arc_challenge,arc_easy,lambada_openai,piqa,winogrande"
        ;;
    zero-shot-hellaswag)
        basic_flags="--batch-size $batch --eval-zero-shot --eval-zero-shot-dataset hellaswag"
        ;;
    *)
        echo "Error: Invalid task '$task'. Must be 'perplexity' or 'zero-shot'." >&2
        exit 1
        ;;
esac

model_name=$(basename "$model")
logfilename="${task}_${model_name}.log"

bf16_flags="--a-bits 16 --w-bits 16"
mixq_flags="--a-bits 4 --w-bits 4 --mixed-precision mixq --threshold-policy percentage --activation-threshold 0.1"
quarot_flags="--a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl"
bitweaver_flags="--a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --activation-threshold 0.05 --weight-threshold 0.05"
int8_flags="--a-bits 8 --w-bits 8"

echo "TASK: BF16" | tee -a $logfilename
python3 ../../eval.py --model /home/dataset/$model $bf16_flags $basic_flags 2>&1 | tee -a $logfilename

echo "TASK: MixQ-W4A4O16" | tee -a $logfilename
python3 ../../eval.py --model /home/dataset/$model $mixq_flags $basic_flags 2>&1 | tee -a $logfilename

echo "TASK: Quarot-W4A4" | tee -a $logfilename
python3 ../../eval.py --model /home/dataset/$model $quarot_flags $basic_flags 2>&1 | tee -a $logfilename

echo "TASK: BitWeaver-W4A4O8" | tee -a $logfilename
python3 ../../eval.py --model /home/dataset/$model $bitweaver_flags $basic_flags 2>&1 | tee -a $logfilename

echo "TASK: Int8-W8A8" | tee -a $logfilename
python3 ../../eval.py --model /home/dataset/$model $int8_flags $basic_flags 2>&1 | tee -a ${logfilename}