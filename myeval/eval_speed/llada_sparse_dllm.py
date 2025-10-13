import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM
from opencompass.models.sparse_dllm.llada_generate import generate
import json
import argparse
from pathlib import Path

def load_questions(data_path):
    """Load questions directly from file"""
    with open(data_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def batch_measure_tps(model, tokenizer, questions, batch_size=1, block_length = 32, steps = 256, gen_length=256, apply_chat_template=False):
    """
    Batch measurement function for TPS and peak memory
    Args:
        model: loaded model
        tokenizer: corresponding tokenizer
        questions: list of pre-processed questions
        batch_size: batch size
        block_length: block length for generation
        steps: generation steps
        gen_length: generation length
    Returns:
        results: dictionary containing all measurement results
    """
    total_tokens = 0
    total_time = 0
    peak_memory = 0
    max_length = 4096  # Truncation length
    results = {
        'batch_info': [],
        'total_tokens': 0,
        'total_time': 0,
        'peak_memory_mb': 0,
        'tps': 0,
        'throughput': 0
    }
    
    # Warmup (first run might be slower)
    print("Running warmup...")
    warmup_questions = questions[:batch_size]
    if batch_size == 1 and isinstance(warmup_questions, list):
        warmup_questions = warmup_questions[0]
    
    if apply_chat_template:
        messages = [{"role": "user", "content": warmup_questions}] 
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs_ids = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=max_length)['input_ids']
    else:
        inputs_ids = tokenizer(warmup_questions, return_tensors="pt", truncation=True, max_length=max_length)['input_ids']
    
    with torch.no_grad():
        _ = generate(model, inputs_ids, steps = steps, gen_length = gen_length, block_length = block_length, temperature=0.,
            cfg_scale=0., remasking='low_confidence', mask_id=126336)
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Process questions in batches
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        if batch_size == 1 and isinstance(batch_questions, list):
            batch_questions = batch_questions[0]
        
        if apply_chat_template:
            messages = [{"role": "user", "content": batch_questions}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs_ids = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=max_length)['input_ids']
        else:
            inputs_ids = tokenizer(batch_questions, return_tensors="pt", truncation=True, max_length=max_length)['input_ids']
        
        input_lengths = inputs_ids.shape[1]
        
        # Measurement
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            outputs = generate(model, inputs_ids, steps = steps, gen_length = gen_length, block_length = block_length,
                    temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336)
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate peak memory
        current_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
        peak_memory += current_peak
        
        # Calculate generated token count
        generated_tokens = gen_length
        
        batch_time = end_time - start_time
        total_tokens += generated_tokens
        total_time += batch_time
        
        # Store batch info
        results['batch_info'].append({
            'batch_num': i//batch_size + 1,
            'input_length': input_lengths,
            'generated_tokens': generated_tokens,
            'time_seconds': batch_time,
            'instant_tps': (generated_tokens/batch_time),
            'current_peak_memory_mb': current_peak
        })
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Calculate final results
    results.update({
        'total_tokens': total_tokens,
        'total_time': total_time,
        'peak_memory_mb': peak_memory / len(questions),
        'tps': (total_tokens / total_time),
        'throughput': (len(questions)/total_time)
    })
    
    return results

def save_results(results, output_path):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Measure model TPS and memory usage")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--data_type", type=str, required=True, help="Type of dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for measurement")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for maxpool")
    parser.add_argument("--keep_ratio", type=float, default=0.5, help="Keep ratio")
    parser.add_argument("--block_length", type=int, default=32, help="Block length")
    parser.add_argument("--steps", type=int, default=256, help="Steps")
    parser.add_argument("--gen_length", type=int, default=256, help="Generate length")
    parser.add_argument("--apply_chat_template", type=bool, default=False, help="Apply chat template")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    
    args = parser.parse_args()
    # Create output directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_filename = f"{args.model_type}_ours_{args.data_type}.json"
    output_path = Path(args.output_dir) / output_filename
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.kernel_size = args.kernel_size
    config.keep_ratio = args.keep_ratio
    config.block_len = args.block_length
    model = LLaDAModelLM.from_pretrained(args.model_path, config=config, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    model.eval()
    
    # 2. Load dataset
    questions = load_questions(args.data_path)
    
    # 3. Run measurement
    results = batch_measure_tps(
        model=model, 
        tokenizer=tokenizer, 
        questions=questions, 
        batch_size=args.batch_size,
        block_length=args.block_length,
        steps=args.steps,
        gen_length=args.gen_length,
        apply_chat_template=args.apply_chat_template
    )
    
    # Add metadata to results
    results['metadata'] = {
        'model_path': args.model_path,
        'model_type': args.model_type,
        'data_path': args.data_path,
        'data_type': args.data_type,
        'method': 'ours',
        'kernel_size': args.kernel_size,
        'keep_ratio': args.keep_ratio,
        'batch_size': args.batch_size,
        'block_length': args.block_length,
        'steps': args.steps,
        'gen_length': args.gen_length,
        'apply_chat_template': args.apply_chat_template,
        'num_questions': len(questions),
        'timestamp': timestamp
    }
    
    # 4. Save results
    save_results(results, output_path)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
