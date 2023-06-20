<!-- # ChatGLM 微调详细攻略 -->

<h1 align="center">ChatGLM 微调和评估攻略</h1>

<p align="center">
  <a href="https://space.bilibili.com/28606893?spm_id_from=333.1007.0.0">
    bilibili
  </a>&nbsp; &nbsp; 
  <a href="https://github.com/tiansztiansz">
    github
  </a>&nbsp; &nbsp;
  <a href="https://www.kaggle.com/tiansztianszs">
    kaggle
  </a>&nbsp; &nbsp;
  <a href="https://huggingface.co/tiansz">
    huggingface
  </a>
</p>

<br>

### 1 克隆仓库
请注意，以下所有命令在ubuntu20.04运行成功，其他系统请自行探索
```bash
git clone https://github.com/tiansztiansz/chatglm-int4-pt.git
```

<br>

### 2 进入项目目录
```bash
cd chatglm-int4-pt
```

<br>

### 3 安装依赖
```bash
pip install datasets rouge_chinese transformers==4.27.1 cpm_kernels sentencepiece
```

<br>


### 4 整理语料
将所有待微调的数据按以下格式整理好，并将语料人工核对，这样才能保证微调后的模型更好

`data/data.xlsx`
| *prompt* | *response* |
|---|---|
| 指令1 | 回复1|
| 指令2 | 回复2 |
| ... | ... |
| 指令n | 回复n |


<br>


### 5 拆分语料
将语料拆分为训练集、验证集、测试集：
```bash
python3 get_data.py --data_path "data/data.xlsx"
```
在`data`文件夹下将生成： `train.json`、`val.json`、`test.json`

<br>


### 6 微调

- 若输入和输出的语句较长，可以增大 `max_source_length` 和 `max_target_length` ，但注意是要2的倍数。
- 若微调效果不好，则可以适当增大 `max_steps` 和 `save_steps`
- 对于多卡微调，该脚本似乎已经自动分配到多卡
- 若想微调其它版本的模型，可以将 `THUDM/chatglm-6b-int4` 更改为 `THUDM/chatglm-6b-int8` 或者 `THUDM/chatglm-6b`
```bash
WANDB_DISABLED=true python3 main.py \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --max_source_length 64 \
    --max_target_length 64 \
    --max_steps 3000 \
    --logging_steps 100 \
    --save_steps 3000 \
    --do_train \
    --train_file data/train.json \
    --validation_file data/val.json \
    --prompt_column prompt \
    --response_column response \
    --overwrite_cache \
    --output_dir output \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --quantization_bit 4
```

<br>



### 7 模型评估
分类模型评估：
- 若想评估其它版本的模型，可以将 `THUDM/chatglm-6b-int4` 更改为 `THUDM/chatglm-6b-int8` 或者 `THUDM/chatglm-6b`
- 若想评估其他步长的模型，请将 `3000` 改为对应的步长
```bash
python3 evaluate.py --model_path "THUDM/chatglm-6b-int4" \
                    --save_steps "3000"
```

<br>

### 8 微调后的模型推理
```python
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch


def load_pt_model(model_path="THUDM/chatglm-6b-int4", save_steps="3000"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Fine-tuning 后的表现测试
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, pre_seq_len=128
    )
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)

    # 此处使用你的 ptuning 工作目录
    prefix_state_dict = torch.load(f"output/checkpoint-{save_steps}/pytorch_model.bin")
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        new_prefix_state_dict[k[len("transformer.prefix_encoder.") :]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_pt_model(model_path="THUDM/chatglm-6b-int4", save_steps="3000")

    prompt = "你提问的第一句话"
    response, history = model.chat(tokenizer, prompt, history=[])
    print(response)

    prompt = "你提问的第二句话"
    response, history = model.chat(tokenizer, prompt, history=history)
    print(response)    
```
