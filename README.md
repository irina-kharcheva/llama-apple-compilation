# Llama-3.2-1B Apple Compilation

This project compiles the Llama-3.2-1B model for Apple devices using Core ML and compare results.


### Prerequisites

- Python 3.8–3.11
- macOS with Xcode installed
- Apple device for testing
- Access to Llama-3.2-1B model (requires accepting Meta's license agreement)
- Hugging Face account and access token

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Login to Hugging Face:
```bash
huggingface-cli login
```

Notice that your actual Hugging Face token should be with “Read access to contents of all public gated repos you can access” permissions.

## Usage

1. Compile the model:
```bash
python export.py
```

2. Generate text:
```bash
python3 generate.py ./StatefulLlama1BInstruct.mlpackage --prompt "Once upon a time"
```

3. Compare models:
```bash
python test_models.py
```

### Notes

- The compilation process includes optimization for Apple devices
- The compiled model is saved as a CoreML package
- Testing script compares both original and compiled model performance
- Make sure to keep your Hugging Face token secure and never commit it to git
