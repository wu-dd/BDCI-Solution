import os
import json
import argparse
from tqdm import tqdm
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# 验证训练集中的golden_docs_pids中的引用标识是否都出现在了response里
def read_jsonl_file_train(tokenizer, jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for each_data in tqdm(data, desc="Processing"):
        for dialogue in each_data['turns']:
            golden_docs_pids=dialogue["golden_docs_pids"]
            id = each_data['conv_id'] + '_' + str(dialogue["turn_id"])
            for pid in golden_docs_pids:
                if "[{}]".format(pid) not in dialogue["response"]:
                    print("{}没有引用{}".format(id,pid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl_file", type=str, default="./data/CORAL/train/train_conversation.json")
    parser.add_argument("--dev_jsonl_file", type=str, default="./data/a_test_conversation.json")
    parser.add_argument("--save_path", type=str, default="./data_processed")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default='./model/Qwen/Qwen2-0.5B')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    read_jsonl_file_train(tokenizer, args.train_jsonl_file)
    print("Complete json data done!")

