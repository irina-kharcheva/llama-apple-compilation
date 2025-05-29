import time
import argparse
import random
import csv
import numpy as np
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaConfig
from coremltools.models import MLModel
from export import MODEL_ID
from generation_utils import load_coreml_model_and_tokenizer, get_next_token_coreml
import torch.profiler
from torch.mps.profiler import profile as mps_profile

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# Ensure deterministic behavior for cuDNN, if used (though less relevant for CPU inference)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROMPTS_FILENAME = "prompts.csv"

def generate_text_original(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int
) -> Tuple[str, float]:
    """Generates text using the original Hugging Face model and measures inference time."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    if model.device.type == "mps":
        input_ids = input_ids.to(model.device)

    start_time = time.perf_counter()
    generated_ids_list = input_ids.cpu().tolist()[0] # Keep generated_ids as a Python list for append

    if model.device.type == "mps":
        print("Starting MPS profiler for original model generation...")
        with mps_profile(mode='interval', wait_until_completed=True):
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    current_input_tensor = torch.tensor([generated_ids_list], dtype=torch.long).to(model.device)
                    outputs = model(current_input_tensor)
                    logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                if next_token_id == tokenizer.eos_token_id:
                    break
                generated_ids_list.append(next_token_id)
        print("Stopped MPS profiler for original model generation.")
    else: # Original path for non-MPS devices
        for _ in range(max_new_tokens):
            with torch.no_grad():
                current_input_tensor = torch.tensor([generated_ids_list], dtype=torch.long).to(model.device)
                outputs = model(current_input_tensor)
                logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            if next_token_id == tokenizer.eos_token_id:
                break
            generated_ids_list.append(next_token_id)
            
    end_time = time.perf_counter()
    
    generated_text = tokenizer.decode(generated_ids_list)
    generation_time = end_time - start_time
    tokens_generated = len(generated_ids_list) - input_ids.shape[1]
    time_per_token = generation_time / tokens_generated if tokens_generated > 0 else 0.0
    
    return generated_text, time_per_token

def generate_text_coreml(
    model: MLModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int
) -> Tuple[str, float]:
    """Generates text using the Core ML model and measures inference time."""
    prompt_tokens_np = tokenizer(prompt, return_tensors="np").input_ids
    
    start_time = time.perf_counter()
    
    extend_tokens = []
    kv_cache_state = model.make_state() # Initialize KV cache for CoreML
    
    token_generator = get_next_token_coreml(
        model,
        prompt_tokens=prompt_tokens_np.astype(np.int32),
        kv_cache_state=kv_cache_state
    )
    
    generated_token_count = 0
    for i, (token, updated_kv_cache_state) in enumerate(token_generator):
        kv_cache_state = updated_kv_cache_state
        if token == tokenizer.eos_token_id or i >= max_new_tokens:
            break
        extend_tokens.append(token)
        generated_token_count +=1
        
    end_time = time.perf_counter()
    
    full_text = tokenizer.decode(prompt_tokens_np[0].tolist() + extend_tokens)
    generation_time = end_time - start_time
    time_per_token = generation_time / generated_token_count if generated_token_count > 0 else 0.0
    
    return full_text, time_per_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare original and CoreML model outputs and speed.")
    parser.add_argument("coreml_model_path", type=str, help="Path to the CoreML model package (.mlpackage)")
    parser.add_argument("--prompt", type=str, default="Translate to German: My name is Llama.", help="Prompt for text generation (used if --use_prompts_from_file is not set).")
    parser.add_argument("--use_prompts_from_file", action="store_true", help=f"If set, read prompts from {PROMPTS_FILENAME} (second column), ignoring --prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate for each prompt.")
    parser.add_argument("--torch_device", type=str, default='cpu', choices=['cpu', 'mps'], help="Device for the original model: 'cpu' or 'mps'. Defaults to 'mps' if available, else 'cpu'.")
    
    args = parser.parse_args()

    prompts_to_run = []
    if args.use_prompts_from_file:
        print(f"Reading prompts from {PROMPTS_FILENAME} (second column)...")
        try:
            with open(PROMPTS_FILENAME, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) > 1:
                        prompts_to_run.append(row[1])
            if not prompts_to_run:
                print(f"Warning: No prompts found in the second column of {PROMPTS_FILENAME} or file is empty.")
        except FileNotFoundError:
            print(f"Error: {PROMPTS_FILENAME} not found. Please create it or do not use --use_prompts_from_file.")
            exit()
        except Exception as e:
            print(f"Error reading {PROMPTS_FILENAME}: {e}")
            exit()
        if prompts_to_run:
             print(f"Loaded {len(prompts_to_run)} prompts from {PROMPTS_FILENAME}.")
    else:
        prompts_to_run.append(args.prompt)
    
    if not prompts_to_run: 
        print("Error: No prompts to process. Please provide a prompt via --prompt or ensure prompts.csv is valid and contains data in the second column when using --use_prompts_from_file.")
        exit()

    print(f"Loading original model: {MODEL_ID}...")
    original_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    config_dict, _ = LlamaConfig.get_config_dict(MODEL_ID, token=True)
    config_dict["rope_scaling"] = {"type": "linear", "factor": 2.0}
    model_config = LlamaConfig.from_dict(config_dict)
    original_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=model_config, token=True)
    original_model.eval()

    torch_device_str = args.torch_device
    if torch_device_str is None: # Default logic if not specified
        if torch.backends.mps.is_available():
            torch_device_str = "mps"
        else:
            torch_device_str = "cpu"
    
    if torch_device_str == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS specified but not available. Falling back to CPU for original model.")
        torch_device_str = "cpu"
    elif torch_device_str == "mps":
        print("Original model will run on MPS device.")
    else:
        print("Original model will run on CPU device.")

    torch_device = torch.device(torch_device_str)
    original_model.to(torch_device)
    
    print("Original model loaded.")

    print(f"Loading CoreML model from: {args.coreml_model_path}...")
    coreml_model, coreml_tokenizer = load_coreml_model_and_tokenizer(args.coreml_model_path)
    print("CoreML model loaded.")

    all_original_times_per_token = []
    all_coreml_times_per_token = []

    for i, current_prompt in enumerate(prompts_to_run):
        print(f"\n--- Prompt {i+1}/{len(prompts_to_run)}: \"{current_prompt}\" ---")
        
        print(f"--- Generating with Original Model ({MODEL_ID}) ---")
        profiler_activities = [torch.profiler.ProfilerActivity.CPU]
        # Note: ProfilerActivity.MPS does not exist. Use XCode to profile MPS.

        with torch.profiler.profile(
            activities=profiler_activities,
            profile_memory=True,
            record_shapes=True,
            with_stack=True # Added for more detailed stack traces
        ) as prof_original:
            original_text, original_time_per_token = generate_text_original(
                original_model, original_tokenizer, current_prompt, args.max_new_tokens
            )
        
        print(f"Output: {original_text}")
        print(f"Time per token: {original_time_per_token:.4f} seconds")
        if original_time_per_token > 0: # Only add valid times for averaging
             all_original_times_per_token.append(original_time_per_token)

        print("\n--- Original Model Profiler Results ---")
        print(prof_original.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        trace_filename_original = f"compare_original_model_trace_prompt_{i+1}.json"
        prof_original.export_chrome_trace(trace_filename_original)
        print(f"Exported Chrome trace for original model to {trace_filename_original}")

        print(f"\n--- Generating with CoreML Model ---")
        coreml_text, coreml_time_per_token = generate_text_coreml(
            coreml_model, coreml_tokenizer, current_prompt, args.max_new_tokens
        )
        print(f"Output: {coreml_text}")
        print(f"Time per token: {coreml_time_per_token:.4f} seconds")
        if coreml_time_per_token > 0: # Only add valid times for averaging
            all_coreml_times_per_token.append(coreml_time_per_token)

        print("\n--- Individual Comparison Summary ---")
        print(f"Prompt: {current_prompt}")
        print(f"Original Model Output: {original_text}")
        print(f"CoreML Model Output:   {coreml_text}")
        output_match = "Yes" if original_text == coreml_text else "No"
        print(f"Outputs Match: {output_match}")
        print(f"Original Model - Time per token: {original_time_per_token:.4f} s")
        print(f"CoreML Model   - Time per token: {coreml_time_per_token:.4f} s")

    print("\n--- Overall Average Performance ---")
    if all_original_times_per_token:
        avg_original_time = sum(all_original_times_per_token) / len(all_original_times_per_token)
        print(f"Average Original Model - Time per token: {avg_original_time:.4f} s (over {len(all_original_times_per_token)} prompts)")
    else:
        avg_original_time = 0
        print("Original model did not successfully generate tokens for any prompt or time was zero.")

    if all_coreml_times_per_token:
        avg_coreml_time = sum(all_coreml_times_per_token) / len(all_coreml_times_per_token)
        print(f"Average CoreML Model   - Time per token: {avg_coreml_time:.4f} s (over {len(all_coreml_times_per_token)} prompts)")
    else:
        avg_coreml_time = 0
        print("CoreML model did not successfully generate tokens for any prompt or time was zero.")

    if avg_coreml_time > 0 and avg_original_time > 0:
        avg_speed_diff = avg_original_time / avg_coreml_time
        print(f"On average, CoreML model is {avg_speed_diff:.2f}x {'faster' if avg_speed_diff > 1 else 'slower'} than the original model.")
    else:
        print("Could not compute average speed comparison (one or both models had no successful generations with measurable time).") 
