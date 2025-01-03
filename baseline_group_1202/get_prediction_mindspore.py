import os
import json
from tqdm import tqdm
import mindspore as ms
from mindspore import context
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM


def predict(test_json_path,tokenizer,model,result_path):
    with open(test_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        results=[]
        for each_data in tqdm(data, desc="Processing"):
            id=each_data['id']
            model_inputs = tokenizer(each_data['prompt'],return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs,max_new_tokens=args.max_new_tokens,pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append({"id":id,"response":response})

        with open(result_path, 'w', encoding='utf-8') as f:
            for item in results:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="./exp/LORA-QWen-8-240-1e-3-500")
    parser.add_argument("--result_path", type=str, default="result.jsonl")
    parser.add_argument("--model_path", type=str, default='./model/Qwen/Qwen2-0.5B')
    parser.add_argument("--test_json_path", type=str, default='./data_processed/test.json')
    parser.add_argument("--max_new_tokens",type=int,default=128)
    args = parser.parse_args()
    
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")  # 或者 'CPU'、'Ascend'
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.set_train(False)
    print("start!")
    predict(args.test_json_path, tokenizer, model)
    print("Done!")
