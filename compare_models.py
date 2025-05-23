import time
import argparse
import random
import numpy as np
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaConfig
from coremltools.models import MLModel
from export import MODEL_ID # Using the same base model ID
from generation_utils import load_coreml_model_and_tokenizer, get_next_token_coreml
# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# Ensure deterministic behavior for cuDNN, if used (though less relevant for CPU inference)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def generate_text_original(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int
) -> Tuple[str, float]:
    """Generates text using the original Hugging Face model and measures inference time."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    
    start_time = time.perf_counter()
    # Generate tokens
    # Use a simple generation loop for fair comparison with the CoreML version
    generated_ids = input_ids.tolist()[0]
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated_ids], dtype=torch.long))
            logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        
        if next_token_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token_id)
        
    end_time = time.perf_counter()
    
    generated_text = tokenizer.decode(generated_ids)
    generation_time = end_time - start_time
    tokens_generated = len(generated_ids) - input_ids.shape[1]
    time_per_token = generation_time / tokens_generated if tokens_generated > 0 else 0
    
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
    
    for i, (token, updated_kv_cache_state) in enumerate(token_generator):
        kv_cache_state = updated_kv_cache_state
        if token == tokenizer.eos_token_id or i >= max_new_tokens:
            break
        extend_tokens.append(token)
        
    end_time = time.perf_counter()
    
    full_text = tokenizer.decode(prompt_tokens_np[0].tolist() + extend_tokens)
    generation_time = end_time - start_time
    time_per_token = generation_time / len(extend_tokens) if extend_tokens else 0
    
    return full_text, time_per_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare original and CoreML model outputs and speed.")
    parser.add_argument("coreml_model_path", type=str, help="Path to the CoreML model package (.mlpackage)")
    parser.add_argument("--prompt", type=str, default="Translate to German: My name is Llama.", help="Prompt for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()

    # Load original Hugging Face model and tokenizer
    print(f"Loading original model: {MODEL_ID}...")
    original_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    config_dict, _ = LlamaConfig.get_config_dict(MODEL_ID, token=True)
    config_dict["rope_scaling"] = {"type": "linear", "factor": 2.0}
    model_config = LlamaConfig.from_dict(config_dict)
    original_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=model_config, token=True)
    original_model.eval() # Set to evaluation mode
    print("Original model loaded.")

    # Load Core ML model and tokenizer (tokenizer should be the same)
    print(f"Loading CoreML model from: {args.coreml_model_path}...")
    coreml_model, coreml_tokenizer = load_coreml_model_and_tokenizer(args.coreml_model_path)
    print("CoreML model loaded.")

    print(f"\n--- Generating with Original Model ({MODEL_ID}) ---")
    original_text, original_time_per_token = generate_text_original(
        original_model, original_tokenizer, args.prompt, args.max_new_tokens
    )
    print(f"Output: {original_text}")
    print(f"Time per token: {original_time_per_token:.4f} seconds")

    print(f"\n--- Generating with CoreML Model ---")
    coreml_text, coreml_time_per_token = generate_text_coreml(
        coreml_model, coreml_tokenizer, args.prompt, args.max_new_tokens
    )
    print(f"Output: {coreml_text}")
    print(f"Time per token: {coreml_time_per_token:.4f} seconds")

    print("\n--- Comparison Summary ---")
    print(f"Prompt: {args.prompt}")
    print(f"Original Model Output: {original_text}")
    print(f"CoreML Model Output:   {coreml_text}")
    
    output_match = "Yes" if original_text == coreml_text else "No"
    print(f"Outputs Match: {output_match}")

    print(f"Original Model - Time per token: {original_time_per_token:.4f} s")
    print(f"CoreML Model   - Time per token: {coreml_time_per_token:.4f} s")

    if coreml_time_per_token > 0 and original_time_per_token > 0:
        speed_diff = original_time_per_token / coreml_time_per_token
        print(f"CoreML model is {speed_diff:.2f}x {'faster' if speed_diff > 1 else 'slower'} than the original model.")
    elif coreml_time_per_token == 0 and original_time_per_token > 0:
        print("CoreML model generated tokens instantly (or too fast to measure for the given prompt/tokens).")
    elif original_time_per_token == 0 and coreml_time_per_token > 0:
         print("Original model generated tokens instantly (or too fast to measure for the given prompt/tokens).")
    else:
        print("Could not compare speed (one or both models did not generate tokens or took no measurable time).") 
