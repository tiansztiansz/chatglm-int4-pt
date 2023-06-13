from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import json


def load_pt_model(model_path = "THUDM/chatglm-6b-int4"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Fine-tuning 后的表现测试
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, pre_seq_len=128
    )
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)

    # 此处使用你的 ptuning 工作目录
    prefix_state_dict = torch.load(
        "output/checkpoint-500/pytorch_model.bin"
    )
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        new_prefix_state_dict[k[len("transformer.prefix_encoder.") :]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()

    return model, tokenizer


def cls_accuracy(model, tokenizer):
    true_num = 0
    total_num = 0

    with open("data/test.json", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            total_num = total_num + 1
            line_dict = json.loads(line)
            prompt = line_dict["prompt"]
            true_response = line_dict["response"]
            predict_response, history = model.chat(tokenizer, prompt, history=[])
            if predict_response == true_response:
                true_num = true_num + 1

    accuracy = true_num / total_num
    print("在分类任务中模型准确率为：{}".format(accuracy))


if __name__ == "__main__":
    model, tokenizer = load_pt_model(model_path = "THUDM/chatglm-6b-int4")
    cls_accuracy(model, tokenizer)
