TRAIN_JSONL_FILE="./data/CORAL/train/train_conversation.json"
DEV_JSONL_FILE="./data/a_test_conversation.json"
MODEL_PATH="./model/Qwen/Qwen2-7B-Instruct"
DATASET_PATH="./data_processed"

# 用于生成dataset
MSL=512

# LORA模型参数
RANK=16
ALPHA=32
DROPOUT=0.1

# 模型参数
LR=5e-4
EPOCH=1000
BS=4
GAS=16


# 标记
NAME="1201-1"

# 输出长度token
MNT=256

# get features
# python get_feature_baseline.py \
#  --train_jsonl_file $TRAIN_JSONL_FILE \
#  --dev_jsonl_file $DEV_JSONL_FILE \
#  --save_path $DATASET_PATH \
#  --model_path $MODEL_PATH

# get dataset
# python get_dataset.py \
#  --save_path $DATASET_PATH/dataset-$MSL \
#  --model_path $MODEL_PATH \
#  --max_seq_length $MSL



# finetuning model
python train.py \
 --model_path  $MODEL_PATH \
 --json_path "$DATASET_PATH/train.json" \
 --dataset_path $DATASET_PATH/dataset-$MSL \
 --lora_rank $RANK \
 --lora_alpha $ALPHA \
 --lora_dropout $DROPOUT \
 --per_device_train_batch_size $BS \
 --gradient_accumulation_steps $GAS \
 --max_steps $EPOCH \
 --save_steps 1000 \
 --save_total_limit 2 \
 --learning_rate $LR \
 --fp16 \
 --remove_unused_columns false \
 --logging_steps 50 \
 --output_dir ./exp/$NAME-LORA-QWen-$MSL-$RANK-$ALPHA-$DROPOUT-$LR-$BS-$GAS-$EPOCH

# evaluation model
python get_evaluate.py \
 --lora_path ./exp/$NAME-LORA-QWen-$MSL-$RANK-$ALPHA-$DROPOUT-$LR-$BS-$GAS-$EPOCH \
 --result_path "Evaluation-result-LORA-QWen-$MSL-$RANK-$ALPHA-$DROPOUT-$LR-$BS-$GAS-$EPOCH-$MNT.txt" \
 --model_path $MODEL_PATH \
 --test_json_path "$DATASET_PATH/eval.json" \
 --max_new_tokens $MNT


# # # # predict
python get_prediction.py \
 --lora_path ./exp/$NAME-LORA-QWen-$MSL-$RANK-$ALPHA-$DROPOUT-$LR-$BS-$GAS-$EPOCH \
 --result_path "Test-result-LORA-QWen-$MSL-$RANK-$ALPHA-$DROPOUT-$LR-$EPOCH-$MNT.jsonl" \
 --model_path $MODEL_PATH \
 --test_json_path "$DATASET_PATH/test.json" \
 --max_new_tokens $MNT

 # 最近3个
 # 对于512的，bs=4，GAS=16
 # 对于1024的，bs=2，GAS=16
 # 对于2048的，bs=2，GAS=16