# Llama-3.2-1B Apple Compilation

This project compiles the Llama-3.2-1B model for Apple devices using Core ML and compare results.


### Prerequisites

- Python 3.8–3.11
- macOS with Xcode installed
- Apple device for testing
- Access to Llama-3.2-1B model (requires accepting Meta's license agreement)
- Hugging Face account and access token

### Project Structure

- `compile_llama.py` - Script for model compilation
- `test_models.py` - Script for testing and comparing models
- `requirements.txt` - Project dependencies
- `.env` - Environment variables (not in git)
- `.env.example` - Example environment variables file
- `Compiled_Llama3_2_1B.mlpackage` - Compiled model (generated after running compile_llama.py)

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the following content:
```
HF_TOKEN=your_huggingface_token
MODEL_NAME=meta-llama/Llama-3.2-1B
MODEL_OUTPUT_NAME=llama_3_2_1b.mlpackage
```

Replace `your_huggingface_token` with your actual Hugging Face token with “Access to gated models” and “Access to gated public repositories” permissions.

## Usage

1. Compile the model:
```bash
python compile_llama.py
```

2. Test the model:
```bash
python test_models.py
```

### Notes

- The compilation process includes optimization for Apple devices
- The compiled model is saved as a CoreML package
- Testing script compares both original and compiled model performance
- Make sure to keep your Hugging Face token secure and never commit it to git
