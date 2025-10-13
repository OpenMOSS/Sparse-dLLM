import time
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from pathlib import Path
import argparse
from opencompass.models.sparse_dllm.dream.modeling_dream import DreamModel

def load_questions(data_path):
    """Load questions directly from file"""
    with open(data_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def batch_measure_tps(model, tokenizer, questions, batch_size=1, steps=256, max_out_len=256, apply_chat_template=False):
    """
    Batch measurement function for TPS and peak memory for Dream model
    Args:
        model: loaded model
        tokenizer: corresponding tokenizer
        questions: list of pre-processed questions
        batch_size: batch size
        steps: diffusion steps
        max_out_len: maximum output length
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
        attention_mask = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=max_length)['attention_mask']
    else:
        messages = tokenizer.bos_token + warmup_questions
        inputs_ids = tokenizer(messages, return_tensors="pt", truncation=True, max_length=max_length)['input_ids']
        attention_mask = tokenizer(messages, return_tensors="pt", truncation=True, max_length=max_length)['attention_mask']
    
    with torch.no_grad():
        _ = model.diffusion_generate(
            inputs_ids.to(model.device),
            attention_mask = attention_mask.to(model.device),
            return_dict_in_generate = True,
            max_new_tokens=max_out_len,
            steps=steps,
            temperature=0.2,
            top_p=0.95,
            alg='entropy'
        )

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
            attention_mask = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=max_length)['attention_mask']
        else:
            messages = tokenizer.bos_token + batch_questions
            inputs_ids = tokenizer(messages, return_tensors="pt", truncation=True, max_length=max_length)['input_ids']
            attention_mask = tokenizer(messages, return_tensors="pt", truncation=True, max_length=max_length)['attention_mask']
        
        input_lengths = inputs_ids.shape[1]
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        outputs = model.diffusion_generate(
            inputs_ids.to(model.device),
            attention_mask = attention_mask.to(model.device),
            return_dict_in_generate = True,
            max_new_tokens=max_out_len,
            steps=steps,
            temperature=0.2,
            top_p=0.95,
            alg='entropy'
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate peak memory
        current_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
        peak_memory += current_peak
        
        # Calculate generated token count
        generated_tokens = max_out_len
        
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
    parser = argparse.ArgumentParser(description="Measure Dream model TPS and memory usage")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--data_type", type=str, required=True, help="Type of dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for measurement")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for maxpool")
    parser.add_argument("--keep_ratio", type=float, default=0.5, help="Keep ratio")
    parser.add_argument("--block_length", type=int, default=32, help="Block length")
    parser.add_argument("--steps", type=int, default=256, help="Diffusion steps")
    parser.add_argument("--max_out_len", type=int, default=256, help="Maximum output length")
    parser.add_argument("--apply_chat_template", type=bool, default=False, help="Apply chat template")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Generate output filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"{args.model_type}_ours_{args.data_type}.json"
    output_path = Path(args.output_dir) / output_filename
    
    # 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.kernel_size = args.kernel_size
    config.keep_ratio = args.keep_ratio
    config.block_len = args.block_length
    model = DreamModel.from_pretrained(
        args.model_path, 
        config=config, 
        device_map='auto', 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).cuda()
    model.eval()
    
    # 2. Load dataset
    questions = load_questions(args.data_path)
    
    # 3. Run measurement
    results = batch_measure_tps(
        model=model, 
        tokenizer=tokenizer, 
        questions=questions, 
        batch_size=args.batch_size,
        steps=args.steps,
        max_out_len=args.max_out_len,
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
        'block_length': args.block_length,
        'batch_size': args.batch_size,
        'steps': args.steps,
        'max_out_len': args.max_out_len,
        'apply_chat_template': args.apply_chat_template,
        'num_questions': len(questions),
        'timestamp': timestamp
    }
    
    # 4. Save results
    save_results(results, output_path)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()