# Llama-3.2-1B Apple Compilation

This project compiles the Llama-3.2-1B model for Apple devices using Core ML.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the following content:
```
HF_TOKEN=your_huggingface_token
MODEL_NAME=meta-llama/Llama-3.2-1B
MODEL_OUTPUT_NAME=llama_3_2_1b.mlpackage
```

Replace `your_huggingface_token` with your actual Hugging Face token.

## Usage

1. Compile the model:
```bash
python compile_llama.py
```

2. Test the model:
```bash
python test_models.py
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Core ML Tools
- python-dotenv

## Notes

- The model is compiled for iOS 16 and later
- Make sure you have access to the Llama-3.2-1B model on Hugging Face
- The compiled model will be saved as `llama_3_2_1b.mlpackage`

This project provides tools for compiling the Llama-3.2-1B model for Apple devices using CoreML.

### Prerequisites

- Python 3.8+
- macOS with Xcode installed
- Apple device for testing (iPhone/iPad/Mac)
- Access to Llama-3.2-1B model (requires accepting Meta's license agreement)
- Hugging Face account and access token

### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Replace `your_token_here` in `.env` with your actual Hugging Face token
   - You can get your token from https://huggingface.co/settings/tokens

### Usage

1. Compile the model:
```bash
python compile_llama.py
```

2. Test and compare models:
```bash
python test_models.py
```

### Project Structure

- `compile_llama.py` - Script for model compilation
- `test_models.py` - Script for testing and comparing models
- `requirements.txt` - Project dependencies
- `.env` - Environment variables (not in git)
- `.env.example` - Example environment variables file
- `Llama3_2_1B.mlpackage` - Compiled model (generated after running compile_llama.py)

### Notes

- The compilation process includes optimization for Apple devices
- The compiled model is saved as a CoreML package
- Testing script compares both original and compiled model performance
- Make sure to keep your Hugging Face token secure and never commit it to git
