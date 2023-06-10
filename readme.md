# ChatGLM 微调详细攻略


## 1 克隆仓库
```
git clone https://github.com/tiansztiansz/chatglm-int4-pt.git
```

<br>

## 2 进入项目目录
```
cd chatglm-int4-pt/src
```

<br>

## 3 安装依赖
```
pip install datasets rouge_chinese transformers==4.27.1 cpm_kernels sentencepiece
```

<br>


## 4 整理语料
将所有待微调的数据按以下格式整理好，并将语料人工核对，这样才能保证微调后的模型更好

`data/data.xlsx`
| prompt | response |
|---|---|
| 指令1 | 回复1|
| 指令2 | 回复2 |
| ... | ... |
| 指令n | 回复n |


<br>


## 5 拆分语料
将语料拆分为训练集、测试集、验证集：
```
python3 get_data.py
```

<br>


## 6 微调
### 6.1 多卡微调
```
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

<br>


### 6.2 单卡微调
```
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

<br>


## 7 分类模型评估
to do