import time
import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_models():
    logger.info("Loading models...")
    # Load original PyTorch model
    model_name = os.getenv('MODEL_NAME')
    logger.debug(f"Model name: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
    logger.debug(f"Tokenizer loaded: {type(tokenizer)}")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        logger.debug("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.debug(f"Pad token ID: {tokenizer.pad_token_id}")
    
    pytorch_model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
    pytorch_model.eval()
    logger.debug(f"PyTorch model loaded: {type(pytorch_model)}")
    
    # Load compiled CoreML model
    coreml_model = ct.models.MLModel(os.getenv('MODEL_OUTPUT_NAME'))
    logger.debug(f"CoreML model loaded: {type(coreml_model)}")
    
    return pytorch_model, coreml_model, tokenizer

def pad_sequence(input_ids, attention_mask, tokenizer, max_length=7, initial_prompt_length=4):
    logger.debug(f"Padding sequence. Current shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
    logger.debug(f"Input device: {input_ids.device}, dtype: {input_ids.dtype}")
    
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        logger.debug("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.debug(f"Using pad token ID: {tokenizer.pad_token_id}")
    
    # For generation, we want to keep the initial prompt and the last tokens
    if input_ids.shape[1] > max_length:
        logger.debug(f"Keeping initial {initial_prompt_length} tokens and last {max_length - initial_prompt_length} tokens")
        # Keep initial prompt
        initial_tokens = input_ids[:, :initial_prompt_length]
        initial_mask = attention_mask[:, :initial_prompt_length]
        
        # Keep last tokens
        last_tokens = input_ids[:, -(max_length - initial_prompt_length):]
        last_mask = attention_mask[:, -(max_length - initial_prompt_length):]
        
        # Combine them
        input_ids = torch.cat([initial_tokens, last_tokens], dim=1)
        attention_mask = torch.cat([initial_mask, last_mask], dim=1)
        
        logger.debug(f"Initial tokens: {tokenizer.decode(initial_tokens[0])}")
        logger.debug(f"Last tokens: {tokenizer.decode(last_tokens[0])}")
    elif input_ids.shape[1] < max_length:
        padding_length = max_length - input_ids.shape[1]
        logger.debug(f"Padding length: {padding_length}")
        
        try:
            # Create padding tensors with correct shape
            padding_ids = torch.full(
                size=(1, padding_length),
                fill_value=tokenizer.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            logger.debug(f"Created padding_ids tensor: shape={padding_ids.shape}, dtype={padding_ids.dtype}, device={padding_ids.device}")
            
            padding_mask = torch.zeros(
                size=(1, padding_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            logger.debug(f"Created padding_mask tensor: shape={padding_mask.shape}, dtype={padding_mask.dtype}, device={padding_mask.device}")
            
            input_ids = torch.cat([input_ids, padding_ids], dim=1)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
            logger.debug(f"After padding - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            
        except Exception as e:
            logger.error(f"Error in padding: {str(e)}")
            logger.error(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            logger.error(f"Padding length: {padding_length}")
            logger.error(f"Device info - input_ids: {input_ids.device}, attention_mask: {attention_mask.device}")
            logger.error(f"Pad token ID: {tokenizer.pad_token_id}")
            raise
    
    return input_ids, attention_mask

def get_next_token_pytorch(model, input_ids, attention_mask):
    logger.debug(f"PyTorch generation - input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
    logger.debug(f"Generated next token: {next_token.item()}")
    return next_token

def get_next_token_coreml(model, input_ids, attention_mask, tokenizer):
    logger.debug(f"CoreML generation - input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    logger.debug(f"Current sequence: {tokenizer.decode(input_ids[0])}")
    
    # Pad sequence to fixed length
    input_ids, attention_mask = pad_sequence(input_ids, attention_mask, tokenizer)
    logger.debug(f"After padding - shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    logger.debug(f"Padded sequence: {tokenizer.decode(input_ids[0])}")
    
    # Convert inputs to CoreML format
    coreml_inputs = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32)
    }
    logger.debug(f"CoreML inputs shapes: input_ids={coreml_inputs['input_ids'].shape}, attention_mask={coreml_inputs['attention_mask'].shape}")
    
    # Get logits from CoreML model
    outputs = model.predict(coreml_inputs)
    logger.debug(f"CoreML outputs keys: {outputs.keys()}")
    
    # Log raw output values
    logits = torch.tensor(outputs["logits"])
    logger.debug(f"Logits shape: {logits.shape}")
    logger.debug(f"Logits min: {logits.min().item()}, max: {logits.max().item()}, mean: {logits.mean().item()}")
    
    # Mask special tokens
    special_token_ids = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id
    ]
    special_token_ids = [id for id in special_token_ids if id is not None]
    
    if special_token_ids:
        logits[0, special_token_ids] = float('-inf')
    
    # Get top 5 tokens for debugging
    top_values, top_indices = torch.topk(logits[0], 5)
    
    # Debug token decoding
    logger.debug("Top 5 tokens details:")
    for idx, (value, token_id) in enumerate(zip(top_values, top_indices)):
        token_id = token_id.item()
        try:
            token_text = tokenizer.decode([token_id])
            logger.debug(f"Token {idx+1}: id={token_id}, value={value:.4f}, text='{token_text}', bytes={[ord(c) for c in token_text]}")
        except Exception as e:
            logger.debug(f"Token {idx+1}: id={token_id}, value={value:.4f}, decode error: {str(e)}")
    
    next_token = torch.argmax(logits, dim=-1)
    selected_token_id = next_token.item()
    try:
        selected_token_text = tokenizer.decode([selected_token_id])
        logger.debug(f"Selected token: id={selected_token_id}, text='{selected_token_text}', bytes={[ord(c) for c in selected_token_text]}")
    except Exception as e:
        logger.debug(f"Selected token: id={selected_token_id}, decode error: {str(e)}")
    
    return next_token

def clean_text(text):
    # Remove unreadable characters and normalize whitespace
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def measure_token_speed_pytorch(model, tokenizer, prompt, num_tokens=50):
    logger.info(f"Measuring PyTorch speed for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")
    logger.debug(f"Tokenized input shapes: input_ids={inputs['input_ids'].shape}, attention_mask={inputs['attention_mask'].shape}")
    
    total_time = 0
    generated_tokens = []
    
    for i in range(num_tokens):
        logger.debug(f"Generating token {i+1}/{num_tokens}")
        start_time = time.time()
        next_token = get_next_token_pytorch(model, inputs["input_ids"], inputs["attention_mask"])
        token_time = time.time() - start_time
        total_time += token_time
        
        # Update inputs for next token
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.unsqueeze(-1)], dim=-1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        # Get the new token
        new_token = next_token[0].item()
        generated_tokens.append(new_token)
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    generated_text = clean_text(generated_text)
    avg_time_per_token = total_time / num_tokens
    logger.info(f"PyTorch generation completed. Average time per token: {avg_time_per_token*1000:.2f}ms")
    return generated_text, avg_time_per_token

def measure_token_speed_coreml(model, tokenizer, prompt, num_tokens=50):
    logger.info(f"Measuring CoreML speed for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")
    logger.debug(f"Tokenized input shapes: input_ids={inputs['input_ids'].shape}, attention_mask={inputs['attention_mask'].shape}")
    logger.debug(f"Input tokens: {tokenizer.decode(inputs['input_ids'][0])}")
    
    total_time = 0
    generated_tokens = []
    
    for i in range(num_tokens):
        logger.debug(f"Generating token {i+1}/{num_tokens}")
        start_time = time.time()
        next_token = get_next_token_coreml(model, inputs["input_ids"], inputs["attention_mask"], tokenizer)
        token_time = time.time() - start_time
        total_time += token_time
        
        # Update inputs for next token
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.unsqueeze(-1)], dim=-1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        # Get the new token
        new_token = next_token[0].item()
        generated_tokens.append(new_token)
        try:
            token_text = tokenizer.decode([new_token])
            logger.debug(f"Token {i+1} generated: id={new_token}, text='{token_text}', bytes={[ord(c) for c in token_text]}")
        except Exception as e:
            logger.debug(f"Token {i+1} generated: id={new_token}, decode error: {str(e)}")
        
        # Log current sequence
        current_sequence = tokenizer.decode(inputs["input_ids"][0])
        logger.debug(f"Current sequence: {current_sequence}")
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    avg_time_per_token = total_time / num_tokens
    logger.info(f"CoreML generation completed. Average time per token: {avg_time_per_token*1000:.2f}ms")
    logger.info(f"Final generated text: {generated_text}")
    return generated_text, avg_time_per_token

def compare_models():
    logger.info("Starting model comparison")
    # Load models
    pytorch_model, coreml_model, tokenizer = load_models()
    
    # Test prompts
    test_prompts = [
        "Once upon a time",
        # "The future of artificial intelligence",
        # "Write a short poem about",
    ]
    
    print("Comparing token generation speed between PyTorch and CoreML models:\n")
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")

        try:
            # Generate with PyTorch model
            pytorch_text, pytorch_time = measure_token_speed_pytorch(pytorch_model, tokenizer, prompt)
            print(f"\nPyTorch output (avg {pytorch_time*1000:.2f}ms per token):")
            print(pytorch_text)
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            continue
            
        try:
            # Generate with CoreML model
            coreml_text, coreml_time = measure_token_speed_coreml(coreml_model, tokenizer, prompt)
            print(f"\nCoreML output (avg {coreml_time*1000:.2f}ms per token):")
            print(coreml_text)
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            continue



if __name__ == "__main__":
    compare_models() 
