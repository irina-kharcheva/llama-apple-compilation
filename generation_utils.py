import argparse
import random
from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
from coremltools.models import MLModel
from transformers import AutoTokenizer

from export import METADATA_TOKENIZER

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def load_coreml_model_and_tokenizer(model_path: str) -> Tuple[MLModel, AutoTokenizer]:
    """Load a Core ML model and corresponding tokenizer."""
    model: MLModel = MLModel(model_path)
    description = model.get_spec().description
    if METADATA_TOKENIZER not in description.metadata.userDefined:
        raise ValueError("Model metadata does not contain tokenizer path.")
    tokenizer_path: str = description.metadata.userDefined[METADATA_TOKENIZER]
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def get_next_token_coreml(model: MLModel, prompt_tokens: np.ndarray, kv_cache_state) -> Generator[Tuple[int, Dict[str, np.ndarray]], None, None]:
    """Generate a sequence of tokens with naive greedy decoding for a CoreML model."""

    def sample(logits: np.ndarray) -> int:
        """Perform greedy decoding on the logits array to get the next token."""
        return int(np.argmax(logits[0][-1], axis=-1))

    def inference_coreml(model: MLModel, input_ids: np.ndarray, num_past_tokens: int, current_kv_cache_state) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Perform inference with the given CoreML model and input data."""
        causal_mask: np.ndarray = np.triu(
            np.full(
                (1, 1, input_ids.shape[-1], num_past_tokens + input_ids.shape[-1]),
                fill_value=-np.inf if num_past_tokens == 0 else 0,
            ),
            k=1,
        ).astype(np.float16)
        outputs: Dict[str, np.ndarray] = model.predict(
            data={"inputIds": input_ids, "causalMask": causal_mask},
            state=current_kv_cache_state,
        )
        # The predict call updates the state in-place, so we return it.
        return outputs["logits"], current_kv_cache_state

    logits, kv_cache_state = inference_coreml(model, input_ids=prompt_tokens, num_past_tokens=0, current_kv_cache_state=kv_cache_state)
    token: int = sample(logits=logits)
    num_past_tokens: int = prompt_tokens.shape[-1]

    while True:
        yield token, kv_cache_state
        logits, kv_cache_state = inference_coreml(
            model,
            input_ids=np.array([[token]], dtype=np.int32),
            num_past_tokens=num_past_tokens,
            current_kv_cache_state=kv_cache_state
        )
        token: int = sample(logits=logits)
        num_past_tokens += 1
