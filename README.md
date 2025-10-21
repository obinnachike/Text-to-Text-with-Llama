<h1 align="center">🦙 LLaMA 2 GPU Inference</h1> <p align="center">Fast Local Inference with <b>llama-cpp-python</b> + CUDA Acceleration</p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python"> <img src="https://img.shields.io/badge/llama--cpp--python-GPU-green?logo=nvidia"> <img src="https://img.shields.io/badge/HuggingFace-Enabled-orange?logo=huggingface"> <img src="https://img.shields.io/badge/CUDA-Acceleration-yellow?logo=nvidia"> <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg"> </p>

 Run Meta’s LLaMA 2 models locally — fast and GPU-optimized — using llama-cpp-python.
This project demonstrates GPU-based inference, quantized model loading, and prompt-based text generation powered by CUDA and GGUF models from Hugging Face.

 Key Features

 GPU-accelerated inference with GGML_CUDA

 Quantized .gguf model support for performance and efficiency

 Direct integration with Hugging Face Hub

 Simple prompt template + structured responses

 Customizable inference parameters (threads, GPU layers, temperature, etc.)

 Installation & Setup
 Verify GPU
!nvidia-smi


Make sure your GPU and CUDA drivers are correctly installed.

 Install Dependencies
# Install with CUDA enabled
!CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose

# Additional packages
!pip install huggingface_hub
!pip install numpy


 Compatibility Tip:
If you face numpy errors, reinstall a stable version:

pip install "numpy<2.3.0"
!pip uninstall numpy -y
!pip install numpy==1.26.4

 Model Setup
 Specify Model
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
model_basename = "llama-2-13b-chat.Q5_0.gguf"


The .gguf file is a quantized format — faster and more memory-efficient for GPUs.

 Import Required Libraries
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

 Download Model
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
print(model_path)

 Load Model on GPU
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,        # CPU threads
    n_batch=512,        # Tune based on VRAM
    n_gpu_layers=32     # Adjust depending on GPU memory
)


 Tuning Tip:
Increase n_gpu_layers if you have high VRAM (e.g., RTX 4090).
For smaller GPUs (e.g., 8GB), reduce this value.

 Prompt & Response
 Create Prompt Template
prompt = "Write a linear regression code"

prompt_template = f'''SYSTEM: You are a helpful, respectful and honest assistant.
USER: {prompt}

ASSISTANT:
'''

 Generate Response
response = lcpp_llm(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=150,
    echo=True
)


Retrieve and print output:

print(response["choices"][0]["text"])

<details> <summary> Example Output</summary>
Q: Write a linear regression code

A: Sure! Here’s a simple implementation in Python:


</details>
 Performance Guide
Parameter	Description	Suggested Value
n_threads	CPU threads for offloading	2–4
n_batch	Batch size (LLM context chunks)	256–512
n_gpu_layers	Layers loaded to GPU	16–64
temperature	Output randomness	0.5–0.8
top_p	Nucleus sampling	0.9–0.95

 Use quantized models (Q4, Q5, Q6) for faster inference and lower VRAM use.

 Example Environment
Component	Example Value
GPU Model	NVIDIA RTX 4090
Python Version	3.11
LLaMA Model	llama-2-13b-chat.Q5_0.gguf
Framework	llama-cpp-python
Acceleration	CUDA (GGML_CUDA=on)

 License

.
LLaMA 2 models follow the Meta LLaMA 2 License
.



Obinna Chike
💻 AI/ML Researcher & Developer


<p align="center"> <b>🦙 LLaMA 2 — Run Large Language Models Locally, Fast, and GPU-Optimized.</b> </p>
