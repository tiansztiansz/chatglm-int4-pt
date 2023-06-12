import pandas as pd
import json
import random
import argparse

parser = argparse.ArgumentParser(description="此脚本的手册")
parser.add_argument("--data_path", type=str, default="data/data.xlsx")
args = parser.parse_args()


def get_all_data(data_path=args.data_path):
    # 1、读入xlsx文件，只能训练100w条数据，但应该够用了；
    df = pd.read_excel(data_path)
    prompts = df["prompt"].to_list()
    responses = df["response"].to_list()

    # 2、将全部提示词和回复都变成字符串，防止数字格式错误；
    prompts = [str(i) for i in prompts]
    responses = [str(i) for i in responses]

    # 3、将提示词和回复转换为 {"prompt": prompt, "response": response} 格式；
    content_list = []
    for response, prompt in zip(responses, prompts):
        content = {"prompt": prompt, "response": response}
        content = json.dumps(content, ensure_ascii=False)
        content_list.append(content)

    # 4、设置训练集、测试集、验证集的比例；
    total_lines = len(content_list)
    train_lines = int(total_lines * 0.9)
    test_lines = int(total_lines * 0.005)
    val_lines = int(total_lines * 0.095)

    # 5、获取分片之后的数据列表；
    train_list = []
    test_list = []
    val_list = []
    random.shuffle(content_list)
    for i, line in enumerate(content_list):
        if i < train_lines:
            train_list.append(line)
        elif i < train_lines + test_lines:
            test_list.append(line)
        elif i < train_lines + test_lines + val_lines:
            val_list.append(line)
        else:
            break

    with open("data/train.json", "w", encoding="UTF-8") as f:
        f.writelines(line.strip() + "\n" for line in train_list)
    with open("data/test.json", "w", encoding="UTF-8") as f:
        f.writelines(line.strip() + "\n" for line in test_list)
    with open("data/val.json", "w", encoding="UTF-8") as f:
        f.writelines(line.strip() + "\n" for line in val_list)


if __name__ == "__main__":
    get_all_data()
