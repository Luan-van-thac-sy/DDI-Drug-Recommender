"""
Benchmark inference time: Teacher (BioMistral-7B) vs Student (LEADER).
Measures latency, throughput, memory usage, and model size.

Usage:
    python benchmark_inference.py \
        --teacher_path saved/lora-ddi/checkpoint-6000 \
        --student_path saved/mimic3/leader/distill-ddi/pytorch_model.bin \
        --data_dir data/mimic3/handled/ \
        --llm_path resources/biomistral-7b
"""
import torch
import numpy as np
import pickle
import json
import argparse
import time
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from generators.distill_generator import DistillEHRDataset
from generators.data import EHRTokenizer
from models.LEADER import LEADER
from utils.config import BertConfig
from utils.utils import read_jsonlines
from llm.biomistral import MistralForMedRec
from llm.lora_cls import PeftModelForCLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="saved/lora-ddi/checkpoint-6000")
    parser.add_argument("--student_path", type=str, default="saved/mimic3/leader/distill-ddi/pytorch_model.bin")
    parser.add_argument("--data_dir", type=str, default="data/mimic3/handled/")
    parser.add_argument("--llm_path", type=str, default="resources/biomistral-7b")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def benchmark_teacher(args, device, test_data):
    """Benchmark Teacher (BioMistral-7B + LoRA) inference."""
    print("\n" + "=" * 60)
    print("BENCHMARKING TEACHER MODEL (BioMistral-7B + LoRA)")
    print("=" * 60)

    tokenizer = EHRTokenizer(f"{args.data_dir}/voc_final.pkl")
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    llm_tokenizer.pad_token = llm_tokenizer.unk_token
    llm_tokenizer.padding_side = "right"

    # Load teacher
    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    teacher = MistralForMedRec.from_pretrained(
        args.llm_path,
        med_voc=len(tokenizer.med_voc.word2idx),
    ).half().to(device)
    teacher = PeftModelForCLS.from_pretrained(teacher, args.teacher_path, is_trainable=False)
    teacher.eval()
    load_time = time.time() - load_start

    total_params, trainable_params = count_parameters(teacher)
    model_memory = get_gpu_memory_mb()

    print(f"  Model loaded in {load_time:.2f}s")
    print(f"  Total parameters: {total_params:,}")
    print(f"  GPU memory (model): {model_memory:.0f} MB")

    # Prepare inputs
    max_source_length = 1056
    latencies = []
    num_samples = min(args.num_samples, len(test_data))

    # Warmup
    print(f"  Warming up ({args.warmup} iterations)...")
    for i in range(min(args.warmup, num_samples)):
        prompt = test_data[i]["input"]
        tokens = llm_tokenizer.encode(text=prompt, add_special_tokens=False)[:max_source_length - 1]
        input_ids = tokens + [llm_tokenizer.eos_token_id]
        while len(input_ids) < max_source_length:
            input_ids += [llm_tokenizer.pad_token_id]
        input_tensor = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            _ = teacher(input_ids=input_tensor, labels=None)

    # Benchmark
    print(f"  Benchmarking ({num_samples} samples)...")
    torch.cuda.synchronize()
    for i in tqdm(range(num_samples), desc="Teacher inference"):
        prompt = test_data[i]["input"]
        tokens = llm_tokenizer.encode(text=prompt, add_special_tokens=False)[:max_source_length - 1]
        input_ids = tokens + [llm_tokenizer.eos_token_id]
        while len(input_ids) < max_source_length:
            input_ids += [llm_tokenizer.pad_token_id]
        input_tensor = torch.tensor([input_ids]).to(device)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = teacher(input_ids=input_tensor, labels=None)
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

    peak_memory = get_gpu_memory_mb()

    results = {
        "model": "Teacher (BioMistral-7B + LoRA)",
        "total_params": total_params,
        "load_time_s": load_time,
        "model_memory_mb": model_memory,
        "peak_memory_mb": peak_memory,
        "num_samples": num_samples,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "median_latency_ms": np.median(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "throughput_samples_per_sec": 1.0 / np.mean(latencies),
    }

    # Free memory
    del teacher
    torch.cuda.empty_cache()

    return results


def benchmark_student(args, device, test_data):
    """Benchmark Student (LEADER compact model) inference."""
    print("\n" + "=" * 60)
    print("BENCHMARKING STUDENT MODEL (LEADER)")
    print("=" * 60)

    tokenizer = EHRTokenizer(f"{args.data_dir}/voc_final.pkl")
    profile_tokenizer = json.load(open(f"{args.data_dir}/profile_dict.json"))
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    llm_tokenizer.pad_token = llm_tokenizer.unk_token

    model_args = argparse.Namespace(
        hidden_size=64, distill=True, d_loss='mse', alpha=0.1,
        align=True, align_weight=0.005, profile=True, prompt_num=1,
        num_trm_layers=1, ddi=True, ml_weight=0.05,
        mdc=False, mdc_weight=0.03, max_seq_length=100,
        max_record_num=10, max_source_length=1056,
        temperature=10, therhold=0.3, graph=False
    )

    test_dataset = DistillEHRDataset(test_data, tokenizer, profile_tokenizer, llm_tokenizer, model_args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load student
    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    config.hidden_size = 64
    model = LEADER(config, model_args, tokenizer, device, profile_tokenizer)
    ddi_adj = pickle.load(open(f"{args.data_dir}/full/ddi_A_final.pkl", "rb"))
    model.register_buffer('ddi_adj', torch.FloatTensor(ddi_adj))
    model.load_state_dict(torch.load(args.student_path, map_location=device))
    model.to(device)
    model.eval()
    load_time = time.time() - load_start

    total_params, trainable_params = count_parameters(model)
    model_memory = get_gpu_memory_mb()

    print(f"  Model loaded in {load_time:.2f}s")
    print(f"  Total parameters: {total_params:,}")
    print(f"  GPU memory (model): {model_memory:.0f} MB")

    # Benchmark
    latencies = []
    num_samples = min(args.num_samples, len(test_data))
    sample_iter = iter(test_loader)

    # Warmup
    print(f"  Warming up ({args.warmup} iterations)...")
    for i in range(min(args.warmup, num_samples)):
        batch = next(sample_iter)
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            _ = model(batch[0], batch[1], batch[2], batch[3], batch[4],
                      profile=batch[6], multi_label=batch[5])

    # Reset iterator
    sample_iter = iter(test_loader)
    print(f"  Benchmarking ({num_samples} samples)...")
    torch.cuda.synchronize()
    for i, batch in enumerate(tqdm(test_loader, desc="Student inference", total=num_samples)):
        if i >= num_samples:
            break
        batch = tuple(t.to(device) for t in batch)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model(batch[0], batch[1], batch[2], batch[3], batch[4],
                           profile=batch[6], multi_label=batch[5])
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

    peak_memory = get_gpu_memory_mb()

    results = {
        "model": "Student (LEADER)",
        "total_params": total_params,
        "load_time_s": load_time,
        "model_memory_mb": model_memory,
        "peak_memory_mb": peak_memory,
        "num_samples": num_samples,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "median_latency_ms": np.median(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "throughput_samples_per_sec": 1.0 / np.mean(latencies),
    }

    del model
    torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_data = read_jsonlines(f"{args.data_dir}/test_leader.json")
    print(f"Test data: {len(test_data)} samples")
    print(f"Benchmarking with {args.num_samples} samples")

    # Benchmark student first (less memory)
    student_results = benchmark_student(args, device, test_data)

    # Benchmark teacher
    teacher_results = benchmark_teacher(args, device, test_data)

    # Print comparison
    print("\n" + "=" * 70)
    print("INFERENCE BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Teacher (7B)':<20} | {'Student (1.2M)':<20} | {'Speedup':<10}")
    print("-" * 85)

    t = teacher_results
    s = student_results

    speedup_latency = t["mean_latency_ms"] / s["mean_latency_ms"]
    speedup_throughput = s["throughput_samples_per_sec"] / t["throughput_samples_per_sec"]
    param_ratio = t["total_params"] / s["total_params"]
    memory_ratio = t["peak_memory_mb"] / max(s["peak_memory_mb"], 1)

    print(f"{'Parameters':<30} | {t['total_params']:>15,} | {s['total_params']:>15,} | {param_ratio:.0f}x smaller")
    print(f"{'Model load time (s)':<30} | {t['load_time_s']:>15.2f} | {s['load_time_s']:>15.2f} | {t['load_time_s']/max(s['load_time_s'],0.01):.1f}x faster")
    print(f"{'GPU memory (MB)':<30} | {t['peak_memory_mb']:>15.0f} | {s['peak_memory_mb']:>15.0f} | {memory_ratio:.1f}x less")
    print(f"{'Mean latency (ms)':<30} | {t['mean_latency_ms']:>15.1f} | {s['mean_latency_ms']:>15.1f} | {speedup_latency:.1f}x faster")
    print(f"{'Median latency (ms)':<30} | {t['median_latency_ms']:>15.1f} | {s['median_latency_ms']:>15.1f} | ")
    print(f"{'P95 latency (ms)':<30} | {t['p95_latency_ms']:>15.1f} | {s['p95_latency_ms']:>15.1f} | ")
    print(f"{'Throughput (samples/sec)':<30} | {t['throughput_samples_per_sec']:>15.1f} | {s['throughput_samples_per_sec']:>15.1f} | {speedup_throughput:.1f}x higher")

    # Save results
    os.makedirs("results/benchmark", exist_ok=True)
    results = {"teacher": teacher_results, "student": student_results,
               "speedup_latency": speedup_latency, "speedup_throughput": speedup_throughput,
               "param_ratio": param_ratio, "memory_ratio": memory_ratio}
    with open("results/benchmark/inference_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/benchmark/inference_benchmark.json")


if __name__ == "__main__":
    main()
