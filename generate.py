import argparse
import random
from typing import List

import numpy as np
import torch
from coremltools.models import MLModel
from transformers import AutoTokenizer

from generation_utils import load_coreml_model_and_tokenizer, get_next_token_coreml

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def generate(
    model: MLModel,
    prompt: str,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> str:
    prompt_tokens: np.ndarray = tokenizer(prompt, return_tensors="np").input_ids
    extend_tokens: List[int] = []
    
    # Initialize KV cache state for Core ML model
    kv_cache_state = model.make_state()

    token_generator = get_next_token_coreml(
        model,
        prompt_tokens=prompt_tokens.astype(np.int32),
        kv_cache_state=kv_cache_state
    )

    for i, (token, updated_kv_cache_state) in enumerate(token_generator):
        kv_cache_state = updated_kv_cache_state # Persist the updated state
        if token == tokenizer.eos_token_id or i >= max_new_tokens: # Fix: ensure max_new_tokens is respected
            break
        extend_tokens.append(token)
    return tokenizer.decode(prompt_tokens[0].tolist() + extend_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    # Use the renamed loading function
    model, tokenizer = load_coreml_model_and_tokenizer(args.model_path)
    extend_text: str = generate(
        model,
        prompt=args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    print(extend_text)
