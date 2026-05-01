# Stage 1b: Predict on test set (Teacher - BioMistral-7B) on T4 15GB
# This writes: results/teacher_lora_ddi/test_predictions.json

CUDA_VISIBLE_DEVICES=0 python main_llm_cls.py \
    --do_predict \
    --test_file data/mimic3/handled/test_leader.json \
    --cache_dir data/mimic3/handled/ \
    --prompt_column input \
    --response_column drug_code \
    --overwrite_cache \
    --model_name_or_path resources/biomistral-7b \
    --peft_path saved/lora-ddi/checkpoint-3000 \
    --output_dir results/teacher_lora_ddi \
    --overwrite_output_dir \
    --max_source_length 2048 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --bf16