**LLaMA 2 GPU Inference with llama-cpp-python**

This guide demonstrates how to set up and run the LLaMA 2 large language model (LLM) on GPU using llama-cpp-python, leveraging CUDA acceleration for faster inference.
The workflow includes model download from Hugging Face, environment setup, prompt creation, and text generation.

**Features**

GPU-accelerated inference with GGML_CUDA

Quantized model support (.gguf format)

Direct integration with Hugging Face Hub

Simple prompt-response generation

Customizable inference parameters (threads, GPU layers, temperature, etc.)

**Requirements**

Before you begin, ensure you have:

NVIDIA GPU with CUDA support

Python ≥ 3.10

CUDA Toolkit properly configured (nvidia-smi should show your GPU)

A Hugging Face account (for model download access)

**Setup**
1. Check GPU
!nvidia-smi


This confirms your GPU driver and CUDA setup.

2. **Install Dependencies**
# Enable CUDA support and install llama-cpp-python
!CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose

# Install supporting libraries
!pip install huggingface_hub
!pip install numpy


If you encounter compatibility issues with numpy, use:

pip install "numpy<2.3.0"
!pip uninstall numpy -y
!pip install numpy==1.26.4

3. Model Configuration

Specify your model path and quantized model file:

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
model_basename = "llama-2-13b-chat.Q5_0.gguf"


The .gguf file format is optimized for performance and memory efficiency.

4. **Import Required Libraries**
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

5. **Download the Model**

Download the model directly from Hugging Face Hub:

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
model_path

6. **Load the Model (GPU)**
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,       
    n_batch=512,        
    n_gpu_layers=32     
)


**Tip**: Increase n_gpu_layers if you have a GPU with large VRAM (e.g., 24GB). Lower it for smaller GPUs.

7. **Create a Prompt Template**
prompt = "Write a linear regression code"

prompt_template = f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''

8. **Generate a Response**
response = lcpp_llm(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=150,
    echo=True
)


Display the raw response:

print(response)


Or extract the generated text:

print(response["choices"][0]["text"])

 **Example Output**
Q: Write a linear regression code

A: Sure! Here’s an example using Python and scikit-learn:

import numpy as np
from sklearn.linear_model import LinearRegression



⚡ Performance Tips

Use CUDA-enabled build (GGML_CUDA=on) for maximum performance.

Keep batch size (n_batch) below n_ctx to avoid GPU overflow.

Use quantized models (Q4, Q5, Q6, etc.) for faster inference.

Disable echoing (echo=False) to return only generated text.

  **Environment Summary**
Component	Example Value
GPU Model	NVIDIA RTX 4090
Python Version	3.11
LLaMA Model	Llama-2-13B-chat.Q5_0.gguf
Library	llama-cpp-python
Acceleration	CUDA (GGML_CUDA=on)


This project follows the original LLaMA 2 License
 and MIT License
 for accompanying scripts.

