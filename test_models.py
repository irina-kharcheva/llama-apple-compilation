import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import coremltools as ct
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_original_model():
    model_name = os.getenv('MODEL_NAME')
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
    
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=50)
    end_time = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    time_per_token = (end_time - start_time) / len(outputs[0])
    
    return generated_text, time_per_token

def test_compiled_model():
    model = ct.models.MLModel(os.getenv('MODEL_OUTPUT_NAME'))
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'), token=os.getenv('HF_TOKEN'))
    
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    outputs = model.predict({
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    })
    end_time = time.time()
    
    generated_text = tokenizer.decode(outputs["output_ids"][0], skip_special_tokens=True)
    time_per_token = (end_time - start_time) / len(outputs["output_ids"][0])
    
    return generated_text, time_per_token

if __name__ == "__main__":
    print("Testing original model...")
    orig_text, orig_time = test_original_model()
    print(f"Original model output: {orig_text}")
    print(f"Time per token: {orig_time:.4f} seconds")
    
    print("\nTesting compiled model...")
    comp_text, comp_time = test_compiled_model()
    print(f"Compiled model output: {comp_text}")
    print(f"Time per token: {comp_time:.4f} seconds")
