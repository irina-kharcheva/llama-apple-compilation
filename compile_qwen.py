import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def compile_qwen_model():
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create example input
    example_input = tokenizer("Hello, how are you?", return_tensors="pt")
    
    # Define input shape
    input_shape = {
        "input_ids": ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512))),
        "attention_mask": ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512)))
    }
    
    # Convert model to CoreML
    mlmodel = ct.convert(
        model,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_shape["input_ids"]),
            ct.TensorType(name="attention_mask", shape=input_shape["attention_mask"])
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
        quantization_config=ct.QuantizationConfig(
            mode="linear",
            bit_width=4,
            calibration_data=example_input
        )
    )
    
    # Save the model
    mlmodel.save("Qwen1_5B.mlpackage")

if __name__ == "__main__":
    compile_qwen_model() 