import os
import json
import argparse
from tqdm import tqdm
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

def read_jsonl_file_train(tokenizer, model, jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    records = []
    for each_data in tqdm(data, desc="Processing"):
        instruction = "Based on the provided dialogue and reference documents, answer the current question <{}> while considering the context from the conversation history. If the question lacks complete details or contains omissions, incorporate relevant context from the conversation history to formulate a response. If the reference documents are insufficient or irrelevant, use your own knowledge to provide a concise answer. Avoid stating that the question cannot be answered. The conversation is as follows."
        id = each_data['conv_id'] + '_' + str(len(each_data['turns']))
        question = each_data['turns'][-1]['question']
        rewrite_question = each_data['turns'][-1]['golden_rewrite']
        response = each_data['turns'][-1]['response']

        context1,context2 = [],[]
        context1.append({"role": "system", "content": instruction.format(question)})
        context2.append({"role": "system", "content": instruction.format(rewrite_question)})
        for dialogue in each_data['turns'][:-1]:
            context1.append({"role": "user", "content": dialogue['question']})
            context1.append({"role": "assistant", "content": dialogue['response']})
            context2.append({"role": "user", "content": dialogue['golden_rewrite']})
            context2.append({"role": "assistant", "content": dialogue['response']})
        context1.append({"role": "user", "content": question})
        context2.append({"role": "user", "content": rewrite_question})
        prompt1 = tokenizer.apply_chat_template(
            context1,
            tokenize=False,
            add_generation_prompt=True)
        prompt2 = tokenizer.apply_chat_template(
            context2,
            tokenize=False,
            add_generation_prompt=True)
        records.append({'prompt': prompt1, 'response': response, 'id': id})
        records.append({'prompt': prompt2, 'response': response, 'id': id})
    return records


def read_jsonl_file_test(tokenizer, jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    records = []
    for each_data in tqdm(data, desc="Processing"):
        instruction = "Based on the provided dialogue and reference documents, answer the current question <{}> while considering the context from the conversation history. If the question lacks complete details or contains omissions, incorporate relevant context from the conversation history to formulate a response. If the reference documents are insufficient or irrelevant, use your own knowledge to provide a concise answer. Avoid stating that the question cannot be answered. The conversation is as follows."
        id = each_data['conv_id'] + '_' + str(len(each_data['turns']))
        question = each_data['turns'][-1]['question']
        response= None

        context = []
        context.append({"role": "system", "content": instruction.format(question)})
        for dialogue in each_data['turns'][:-1]:
            context.append({"role": "user", "content": dialogue['question']})
            context.append({"role": "assistant", "content": dialogue['response']})
        context.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True)
        records.append({'prompt': prompt, 'response': response, 'id': id})
    return records


def read_jsonl_file_evaluation(tokenizer, jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    records = []
    for each_data in tqdm(data, desc="Processing"):
        instruction = "Based on the provided dialogue and reference documents, answer the current question <{}> while considering the context from the conversation history. If the question lacks complete details or contains omissions, incorporate relevant context from the conversation history to formulate a response. If the reference documents are insufficient or irrelevant, use your own knowledge to provide a concise answer. Avoid stating that the question cannot be answered. The conversation is as follows."
        id = each_data['conv_id'] + '_' + str(len(each_data['turns']))
        question = each_data['turns'][-2]['question']
        response = each_data['turns'][-2]['response']

        context = []
        context.append({"role": "system", "content": instruction.format(question)})
        for dialogue in each_data['turns'][:-2]:
            context.append({"role": "user", "content": dialogue['question']})
            context.append({"role": "assistant", "content": dialogue['response']})
        context.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True)
        records.append({'prompt': prompt, 'response': response, 'id': id})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl_file", type=str, default="./data/CORAL/train/train_conversation.json")
    parser.add_argument("--dev_jsonl_file", type=str, default="./data/a_test_conversation.json")
    parser.add_argument("--save_path", type=str, default="./data_processed")
    parser.add_argument("--model_path", type=str, default='./model/Qwen/Qwen2-0.5B')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
    train_jsonl_content = read_jsonl_file_train(tokenizer, model, args.train_jsonl_file)
    with open(os.path.join(args.save_path, 'train.json'), 'w', encoding='utf-8') as file:
        json.dump(train_jsonl_content, file, ensure_ascii=False, indent=4)
    test_jsonl_content = read_jsonl_file_test(tokenizer, args.dev_jsonl_file)
    with open(os.path.join(args.save_path, 'test.json'), 'w', encoding='utf-8') as file:
        json.dump(test_jsonl_content, file, ensure_ascii=False, indent=4)
    eval_jsonl_content = read_jsonl_file_evaluation(tokenizer, args.dev_jsonl_file)
    with open(os.path.join(args.save_path, 'eval.json'), 'w', encoding='utf-8') as file:
        json.dump(eval_jsonl_content, file, ensure_ascii=False, indent=4)

    print("Complete json data done!")



