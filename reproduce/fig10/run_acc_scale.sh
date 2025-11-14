if [ -z "$model" ]; then
    model="/home/dataset/Qwen3-8B"
fi

model_name=$(basename "$model")
logfilename="ppl_scale_${model_name}.log"

percent=0.015625
echo percent = $percent | tee -a $logfilename
srun --gres=gpu:H100:1 -w fuse2 -p Long python3 ../../eval.py --model $model --a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --batch-size 32 --eval-perplexity --activation-threshold $percent --weight-threshold $percent 2>&1 | tee -a $logfilename

percent=0.03125
echo percent = $percent | tee -a $logfilename
srun --gres=gpu:H100:1 -w fuse2 -p Long python3 ../../eval.py --model $model --a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --batch-size 32 --eval-perplexity --activation-threshold $percent --weight-threshold $percent 2>&1 | tee -a $logfilename
percent=0.046875
echo percent = $percent | tee -a $logfilename
srun --gres=gpu:H100:1 -w fuse2 -p Long python3 ../../eval.py --model $model --a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --batch-size 32 --eval-perplexity --activation-threshold $percent --weight-threshold $percent 2>&1 | tee -a $logfilename

percent=0.0625
echo percent = $percent | tee -a $logfilename
srun --gres=gpu:H100:1 -w fuse2 -p Long python3 ../../eval.py --model $model --a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --batch-size 32 --eval-perplexity --activation-threshold $percent --weight-threshold $percent 2>&1 | tee -a $logfilename
percent=0.078125
echo percent = $percent | tee -a $logfilename
srun --gres=gpu:H100:1 -w fuse2 -p Long python3 ../../eval.py --model $model --a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --batch-size 32 --eval-perplexity --activation-threshold $percent --weight-threshold $percent 2>&1 | tee -a $logfilename

percent=0.09375
echo percent = $percent | tee -a $logfilename
srun --gres=gpu:H100:1 -w fuse2 -p Long python3 ../../eval.py --model $model --a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --batch-size 32 --eval-perplexity --activation-threshold $percent --weight-threshold $percent 2>&1 | tee -a $logfilename
percent=0.125
echo percent = $percent | tee -a $logfilename
srun --gres=gpu:H100:1 -w fuse2 -p Long python3 ../../eval.py --model $model --a-bits 4 --w-bits 4 --rotate hadamard --w-clip --smooth-quant --smooth-quant-alpha 0.5 --smooth-quant-dataset ../../val.jsonl --mixed-precision bitweaver --threshold-policy percentage --batch-size 32 --eval-perplexity --activation-threshold $percent --weight-threshold $percent 2>&1 | tee -a $logfilename
