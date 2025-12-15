# here put the import lib
import os
# comment out for colab
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from llm.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, HfArgumentParser, Seq2SeqTrainingArguments
from transformers import AutoModel, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from llm.llama import LlamaForMedRec
from llm.trainer_seq2seq import MedRecTrainer
from llm.lora_cls import PeftModelForCLS
from llm.arguments import DataTrainingArguments, ModelArguments
from llm.data_processor.llama import llama_train_cls, llama_eval_cls
from llm.data_processor.collator import LongestSequenceCollator
from generators.data import Voc, EHRTokenizer
from evaluate import evaluate_jsonlines
import time


# save model for PeftModel
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


def train():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device_map = "auto"

    # load diag, proc, med word2id tokenizer
    # Extract dataset name (mimic3 or mimic4) from cache_dir or train_file
    if model_args.cache_dir:
        # Extract dataset name from cache_dir (e.g., "data/mimic4/handled/" -> "mimic4")
        dataset_name = model_args.cache_dir.split('/')[1] if '/' in model_args.cache_dir else "mimic3"
    elif data_args.train_file:
        # Extract dataset name from train_file (e.g., "data/mimic4/handled/train_0104.json" -> "mimic4")
        dataset_name = data_args.train_file.split('/')[1] if '/' in data_args.train_file else "mimic3"
    else:
        dataset_name = "mimic3"  # default fallback

    voc_dir = f"data/{dataset_name}/handled/voc_final.pkl"
    ehr_tokenizer = EHRTokenizer(voc_dir)

    ## Load Model ##
    model = LlamaForMedRec.from_pretrained(
        model_args.model_name_or_path,
        med_voc=len(ehr_tokenizer.med_voc.word2idx),
        # TODO: add device map and torch dtype for colab
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if model_args.peft_path is not None:    # for test model
        # Resume_training
        if training_args.resume_from_checkpoint is not None:
            model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=True)
        else:
            model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=False)
    else:   # for train model
        # Load Lora Config
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.trainable.split(","),
            lora_dropout=model_args.lora_dropout,
            task_type="SEQ_CLS",
        )

        model = PeftModelForCLS(model, peft_config)  # LoRA wrapped llama

    if training_args.do_train:
        for name, param in model.named_parameters():    # activate the CLS head parameters
            if "cls_head" in name:
                param.requires_grad = True
    model.print_trainable_parameters()

    ## Load Tokenizer ##
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"  # define the padding direction

    ## Load Dataset ##
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir
    )
    print("raw_datasets: ", raw_datasets)

    if training_args.do_train:
        target_dataset = raw_datasets["train"]
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        target_dataset = raw_datasets["eval"]
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        target_dataset = raw_datasets["test"]
        column_names = raw_datasets["test"].column_names

    # Use appropriate preprocessing function based on mode
    if training_args.do_train:
        preprocess_func = llama_train_cls(data_args, model_args, tokenizer, ehr_tokenizer)
    else:
        preprocess_func = llama_eval_cls(data_args, model_args, tokenizer, ehr_tokenizer)
    data_collator = LongestSequenceCollator(tokenizer)

    with training_args.main_process_first(desc="Dataset map pre-processing"):
        target_dataset = target_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on prediction dataset",
        )
    target_dataset.set_format("torch")

    ## Set Trainer ##
    trainer = MedRecTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None), # substitute the original model saver
    )

    ## Train Model
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()

    ## Evaluation ##
    results = {}

    if training_args.do_predict:
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        start_time = time.time()
        with torch.no_grad():
            predict_results = trainer.predict(
                target_dataset,
                metric_key_prefix="predict",
                # max_tokens=512,
                # max_new_tokens=data_args.max_target_length,
                # do_sample=True,
                # top_p=0.7,
                # temperature=0.95,
                # repetition_penalty=1.1
            )
        end_time = time.time()

        if trainer.is_world_process_zero():
            predictions = predict_results.predictions
            assert len(predictions) == len(list_test_samples)
            hidden_states = predict_results.label_ids

            output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for idx, p in enumerate(predictions):
                    samp = list_test_samples[idx]
                    #samp["target"] = ehr_tokenizer.med_voc.idx2word[p]
                    samp["hidden_states"] = hidden_states[idx].astype(float).tolist()
                    samp["target"] = p.astype(float).tolist()
                    res = json.dumps(samp, ensure_ascii=False)
                    writer.write(f"{res}\n")

            ja, prauc, avg_p, avg_r, avg_f1, drug_code_results = evaluate_jsonlines(output_prediction_file, ehr_tokenizer)   # output the MedRec metrics

            # Save drug codes to files (both JSON and CSV)
            drug_codes_json_file = os.path.join(training_args.output_dir, "drug_codes_comparison.json")
            drug_codes_csv_file = os.path.join(training_args.output_dir, "drug_codes_comparison.csv")

            # Save JSON (keep for compatibility)
            with open(drug_codes_json_file, "w", encoding="utf-8") as f:
                json.dump(drug_code_results, f, indent=2, ensure_ascii=False)

            # Save CSV with 3 columns (semicolon separator for better readability)
            import csv
            with open(drug_codes_csv_file, "w", encoding="utf-8", newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['subject_id', 'true_drug_codes', 'pred_drug_codes'])

                # Write data rows
                for i in range(len(drug_code_results['subject_ids'])):
                    subject_id = drug_code_results['subject_ids'][i]
                    true_codes = '; '.join(drug_code_results['true_drug_codes'][i])  # Join with semicolon + space
                    pred_codes = '; '.join(drug_code_results['pred_drug_codes'][i])  # Join with semicolon + space
                    writer.writerow([subject_id, true_codes, pred_codes])

            print(f"\n✓ Drug codes saved to:")
            print(f"   - JSON: {drug_codes_json_file}")
            print(f"   - CSV:  {drug_codes_csv_file}")
            print(f"   - Total samples: {len(drug_code_results['subject_ids'])}")
            print(f"   - Average true drugs per patient: {sum(len(codes) for codes in drug_code_results['true_drug_codes']) / len(drug_code_results['true_drug_codes']):.2f}")
            print(f"   - Average predicted drugs per patient: {sum(len(codes) for codes in drug_code_results['pred_drug_codes']) / len(drug_code_results['pred_drug_codes']):.2f}")

            # Store in results dict
            results = {
                'jaccard': ja,
                'prauc': prauc,
                'precision': avg_p,
                'recall': avg_r,
                'f1': avg_f1,
                'drug_codes': drug_code_results
            }

    return results


if __name__ == "__main__":

    train()


