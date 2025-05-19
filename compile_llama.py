import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def compile_llama_model():
    # Load model and tokenizer with token
    model_name = os.getenv('MODEL_NAME')
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
    
    # Set model to evaluation mode
    model.eval()
    
    # Create example input
    example_input = tokenizer("Hello, how are you?", return_tensors="pt")
    input_ids = example_input["input_ids"]
    attention_mask = example_input["attention_mask"]
    
    # Define a wrapper class for the model
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits
    
    # Create and trace the wrapped model
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    try:
        # Try to trace the model
        traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))
        
        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="input_ids", shape=input_ids.shape),
                ct.TensorType(name="attention_mask", shape=attention_mask.shape)
            ],
            source="pytorch",
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS16
        )
        
        # Save the model
        output_name = os.getenv('MODEL_OUTPUT_NAME')
        mlmodel.save(output_name)
        print(f"Model successfully compiled and saved as {output_name}")
        
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        raise

if __name__ == "__main__":
    compile_llama_model()
