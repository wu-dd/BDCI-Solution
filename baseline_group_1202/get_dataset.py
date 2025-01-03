import os
import json
import argparse
from tqdm import tqdm
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer



def preprocess(tokenizer, example, max_seq_length):
    prompt = example["prompt"]
    response = example["response"]
    a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=max_seq_length)
    b_ids = tokenizer.encode(text=response, add_special_tokens=False, truncation=True, max_length=max_seq_length)
    input_ids = a_ids + b_ids + [tokenizer.eos_token_id]

    return {"input_ids": input_ids, "seq_len": len(a_ids)}


def read_jsonl(path, max_seq_length, tokenizer, skip_overlength=False):
    def load_data(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    train_data = load_data(path)
    for data in tqdm(train_data):
        feature = preprocess(tokenizer, data, max_seq_length)
        if skip_overlength and len(feature["input_ids"]) > max_seq_length:
            continue
        yield feature


def tokenize_dataset(train_json_path, save_path, max_seq_length, tokenizer, skip_overlength=False):
    os.makedirs(save_path, exist_ok=True)
    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(train_json_path, max_seq_length, tokenizer, skip_overlength),
        cache_dir=False
    )
    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./data_processed")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default='./model/Qwen/Qwen2.5-7B-Instruct')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")

    tokenize_dataset('./data_processed/train.json', args.save_path, args.max_seq_length, tokenizer, args.skip_overlength)