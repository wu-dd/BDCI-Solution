import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from rouge import Rouge

def predict(test_json_path,tokenizer,model,result_path):
    print(test_json_path)
    rouge = Rouge()  # 初始化 Rouge 计算器
    with open(test_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        predictions = []
        references = []
        results = []
        for each_data in tqdm(data, desc="Processing"):
            id=each_data['id']
            model_inputs = tokenizer(each_data['prompt'],return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs,max_new_tokens=args.max_new_tokens,pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response_target = each_data['response']
            # print(each_data['prompt'])
            # print("**Predict**:{}".format(response))
            # print("**Targett**:{}".format(response_target))
            # 保存预测与目标响应
            # predictions.append(response)
            # references.append(response_target)

            # 计算单例的 Rouge-L 分数
            if len(response)==0:
                response="I done't know."
            # print(id)
            # print(response)
            scores = rouge.get_scores(response, response_target, avg=True)
            results.append({
                "id": id,
                # "predict": response,
                # "target": response_target,
                "rouge-l": scores["rouge-l"]
            })
            # results.append({"id":id,"response":response})

        with open(result_path, 'w', encoding='utf-8') as f:
            # for item in results:
            #     json_line = json.dumps(item, ensure_ascii=False)
            #     f.write(json_line + '\n')
            avg_rouge_l_r = sum([res["rouge-l"]["r"] for res in results]) / len(results)
            avg_rouge_l_p = sum([res["rouge-l"]["p"] for res in results]) / len(results)
            avg_rouge_l_f = sum([res["rouge-l"]["f"] for res in results]) / len(results)
            f.write("Average ROUGE-L Score:{}\n".format(avg_rouge_l_r))
            f.write("Average ROUGE-L Score:{}\n".format(avg_rouge_l_p))
            f.write("Average ROUGE-L Score:{}\n".format(avg_rouge_l_f))

    # 保存每个样本的预测结果和 Rouge-L 分数
    # with open(result_path, 'w', encoding='utf-8') as result_file:
    #     json.dump(results, result_file, ensure_ascii=False, indent=4)

    # 计算数据集上的平均 Rouge-L 分数
    # avg_rouge_l = sum([res["rouge-l"] for res in results]) / len(results)
    # print("Average ROUGE-L Score:", avg_rouge_l)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="./exp/LORA-QWen-8-240-1e-3-500")
    parser.add_argument("--result_path", type=str, default="test.txt")
    parser.add_argument("--model_path", type=str, default='./model/Qwen/Qwen2-0.5B')
    parser.add_argument("--test_json_path", type=str, default='./data_processed/eval.json')
    parser.add_argument("--max_new_tokens",type=int,default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype="auto",device_map="auto")
    model = PeftModel.from_pretrained(model, args.lora_path)
    merged_model = model.merge_and_unload()
    model.eval()
    predict(args.test_json_path,tokenizer,model,args.result_path)
    print("Done!")