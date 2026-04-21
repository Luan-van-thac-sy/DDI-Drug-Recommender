lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="resources/biomistral-7b"   # BioMistral-7B model path
your_data_path="data/mimic3/handled/"  # 填入数据集所在的文件夹路径
your_checkpopint_path="saved"  # 填入用来存储模型的路径
MAX_STEPS=3000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="0105"
MAX_SOURCE_LENGTH=1024

# Testing Command
deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm_cls.py \
    --do_predict \
    --test_file $your_data_path/test_0105.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column drug_code \
    --model_name_or_path $model_name_or_path \
    --peft_path $your_checkpopint_path/lora-$date/checkpoint-$MAX_STEPS \
    --output_dir results/$date \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_eval_batch_size 4