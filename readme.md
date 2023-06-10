`data/data.xlsx`的格式：
| prompt | response |
|---|---|
| 指令1 | 回复1|
| 指令2 | 回复2 |
| ... | ... |
| 指令n | 回复n |


<br>

单机多卡微调：
```bash
CUDA_VISIBLE_DEVICES=0,1 && WANDB_DISABLED=true python3 main.py \
    --do_train \
    --train_file data/train.json \
    --validation_file data/val.json \
    --prompt_column prompt \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --output_dir output/chatglm-6b-int4-pt \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --quantization_bit 4
```

单机多卡评估：
```bash
CUDA_VISIBLE_DEVICES=0,1 && WANDB_DISABLED=true python3 main.py \
    --do_predict \
    --validation_file data/val.json \
    --test_file data/test.json \
    --overwrite_cache \
    --prompt_column prompt \
    --response_column response \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --ptuning_checkpoint output/chatglm-6b-int4-pt/checkpoint-500 \
    --output_dir output/chatglm-6b-int4-pt \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len 128 \
    --quantization_bit 4
```

<br>

单机单卡微调：
```bash
CUDA_VISIBLE_DEVICES=0 && WANDB_DISABLED=true python3 main.py \
    --do_train \
    --train_file data/train.json \
    --validation_file data/val.json \
    --prompt_column prompt \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --output_dir output/chatglm-6b-int4-pt \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --quantization_bit 4
```


单机单卡评估：
```bash
CUDA_VISIBLE_DEVICES=0 && WANDB_DISABLED=true python3 main.py \
    --do_predict \
    --validation_file data/val.json \
    --test_file data/test.json \
    --overwrite_cache \
    --prompt_column prompt \
    --response_column response \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --ptuning_checkpoint output/chatglm-6b-int4-pt/checkpoint-500 \
    --output_dir output/chatglm-6b-int4-pt \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len 128 \
    --quantization_bit 4
```
