import os
import json
import argparse
from tqdm import tqdm
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer


def abbreviate_response(tokenizer, model, history, question, response):
    instruction = "Based on the conversation history <{}> and the current question <{}>, summarize the assistant’s response to make it concise and directly relevant to the question, and output only the summarized content. The summary should:\n\n1. Focus solely on the parts of the response that address the question.\n2. Retain references to cited document IDs.\n3. Exclude unnecessary details or overly long explanations.\n4. Ensure factual accuracy and maintain coherence.\n\nAssistant’s response: <{}>"
    prompt = instruction.format(history, question, response)
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=200, min_new_tokens=100,
                                   pad_token_id=tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    abbreviated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return abbreviated_response

def abbreviate_document(tokenizer, model, history, text):
    instruction = "Based on the conversation history <{}>, generate a concise and relevant summary of the provided reference document. Output only the summarized content. The summary should:\n\n1. Focus on details that directly address the question or align with the conversation context.\n2. Exclude unrelated or redundant information.\n3. Be factually accurate, clear, and coherent.\n\nReference document content: <{}>"

    prompt=instruction.format(history, text)
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, min_new_tokens=50,
                                   pad_token_id=tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    abbreviated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return abbreviated_response

def read_jsonl_file_train(tokenizer, model, jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    records = []
    for each_data in tqdm(data[:2], desc="Processing"):
        id = each_data['conv_id'] + '_' + str(len(each_data['turns']))
        question = each_data['turns'][-1]['question']
        response = each_data['turns'][-1]['response']

        # obtain conversation history for pre-processing and abbreviated conversation dialogue
        context=[]
        # context_instruction="You are an AI assistant that acts as a comprehensive and reliable source of information, similar to Wikipedia.\nYour role is to provide accurate, concise, and well-structured responses to questions based on the provided context or your general knowledge."
        # context.append({"role": "system", "content": context_instruction})
        dialogue_messages=[]
        golden_docs_pids = []
        golden_docs_text = []
        for dialogue in each_data['turns'][:-1]:
            context.append({"role": "user", "content": dialogue['question']})
            response=abbreviate_response(tokenizer,model,tokenizer.apply_chat_template(context,tokenize=False,add_generation_prompt=False),dialogue['question'],dialogue['response'])
            context.append({"role": "assistant", "content": dialogue['response']})
            dialogue_messages.append({"role": "user", "content": dialogue['question']})
            dialogue_messages.append({"role": "assistant", "content": response})
            golden_docs_pids = golden_docs_pids + dialogue['golden_docs_pids']
            golden_docs_text = golden_docs_text + dialogue['golden_docs_text']
        # obtain referenced documentation and abbreviate them
        document_messages=[]
        if golden_docs_pids and golden_docs_text:
            for pid, text in zip(golden_docs_pids, golden_docs_text):
                abbreviated_text = abbreviate_document(tokenizer, model, context,text)
                document_messages.append("Document ID: {}\n{}".format(pid,abbreviated_text))
        # obtain prompt
        instruction = "Based on the provided dialogue and reference documents, answer the current question <{}> while considering the context from the conversation history. Use the reference documents (IDs and text) as the primary source of information. If the question lacks complete details or contains omissions, incorporate relevant context from the conversation history to formulate a response. If the reference documents are insufficient or irrelevant, use your own knowledge to provide a concise answer. Avoid stating that the question cannot be answered. The conversation is <{}>."
        prompt_message=[]
        prompt_message.append({"role": "system", "content": instruction.format(question,tokenizer.apply_chat_template(dialogue_messages,tokenize=False,add_generation_prompt=False))})
        prompt_message.append({"role": "system", "content": "The following reference documents are provided:"})
        for each_document in document_messages:
            prompt_message.append({"role": "system", "content": f"Document ID: {each_document}"})
        prompt = tokenizer.apply_chat_template(
            prompt_message,
            tokenize=False,
            add_generation_prompt=False)
        records.append({'prompt': prompt, 'response': response, 'id': id})
    return records


def read_jsonl_file_test(tokenizer, jsonl_file):
    instruction = "Based on the provided dialogue and reference documents, answer the current question <{}> while considering the context from the conversation history. Use the reference documents (IDs and text) as the primary source of information. If the question lacks complete details or contains omissions, incorporate relevant context from the conversation history to formulate a response. If the reference documents are insufficient or irrelevant, use your own knowledge to provide a concise answer. Avoid stating that the question cannot be answered. The conversation is <{}>."
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    records = []
    for each_data in tqdm(data, desc="Processing"):
        id = each_data['conv_id'] + '_' + str(len(each_data['turns']))
        question = each_data['turns'][-1]['question']
        if 'response' in each_data['turns'][-1]:
            response = each_data['turns'][-1]['response']
        else:
            response = None

        dialogue_messages = []
        for dialogue in each_data['turns'][:-1]:
            dialogue_messages.append({"role": "user", "content": dialogue['question']})
            dialogue_messages.append({"role": "assistant", "content": dialogue['response']})
        messages = []
        messages.append({"role": "system", "content": instruction.format(question,tokenizer.apply_chat_template(dialogue_messages,tokenize=False,add_generation_prompt=False))})
        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        records.append({'prompt': prompt, 'response': response, 'id': id})
    return records


def read_jsonl_file_evaluation(tokenizer, jsonl_file):
    instruction = "Based on the provided dialogue and reference documents, answer the current question <{}> while considering the context from the conversation history. Use the reference documents (IDs and text) as the primary source of information. If the question lacks complete details or contains omissions, incorporate relevant context from the conversation history to formulate a response. If the reference documents are insufficient or irrelevant, use your own knowledge to provide a concise answer. Avoid stating that the question cannot be answered. The conversation is <{}>."
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    records = []
    for each_data in tqdm(data, desc="Processing"):
        id = each_data['conv_id'] + '_' + str(len(each_data['turns']))
        question = each_data['turns'][-2]['question']
        if 'response' in each_data['turns'][-2]:
            response = each_data['turns'][-2]['response']
        else:
            response = None
        dialogue_messages = []
        for dialogue in each_data['turns'][:-1]:
            dialogue_messages.append({"role": "user", "content": dialogue['question']})
            dialogue_messages.append({"role": "assistant", "content": dialogue['response']})
        messages = []
        messages.append({"role": "system", "content": instruction.format(question, tokenizer.apply_chat_template(dialogue_messages, tokenize=False, add_generation_prompt=False))})
        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        records.append({'prompt': prompt, 'response': response, 'id': id})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl_file", type=str, default="./data/CORAL/train/train_conversation.json")
    parser.add_argument("--dev_jsonl_file", type=str, default="./data/a_test_conversation.json")
    parser.add_argument("--save_path", type=str, default="./data_processed")
    parser.add_argument("--model_path", type=str, default='./model/Qwen/Qwen2.5-7B-Instruct')
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



