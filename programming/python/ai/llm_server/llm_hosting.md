# LLM Hosting

## Common LLM Model Weight Formats

The storage and distribution of LLM parameters rely on several specialized file formats, each optimizing for different constraints such as security, loading speed, and hardware execution.

### 1. Safetensors (`.safetensors`)

Developed by Hugging Face, Safetensors has become the de facto standard for model distribution.
* **Security**: Enforces strict data parsing without relying on Python's `pickle`, eliminating arbitrary code execution vulnerabilities.
* **Performance**: Utilizes zero-copy memory mapping, enabling fast, direct loading into memory (RAM/VRAM) while bypassing CPU serialization overhead.
* **Frameworks**: Hugging Face Transformers, vLLM, Text Generation Inference (TGI), DeepSpeed.

### 2. PyTorch Pickle (`.bin`, `.pt`, `.pth`)

The legacy serialization format indigenous to PyTorch.
* **Vulnerability**: Deserialization relies on `pickle`, structurally allowing malicious arbitrary code execution.
* **Status**: Largely deprecated for public weight distribution, though remains functional for internal, trusted operations.
* **Frameworks**: PyTorch (native), Hugging Face Transformers (legacy support).

### 3. GGUF (GPT-Generated Unified Format)

An optimized binary format architected primarily for `llama.cpp` and edge-device inference, superseding the legacy GGML format.
* **Architecture**: Encapsulates tensors, model architecture metadata, and tokenization rules within a singular, extensible file wrapper.
* **Application**: Tailored for CPU-bound environments and unified memory architectures (e.g., Apple Silicon), naturally supporting robust integer quantization architectures (e.g., INT4, INT8) and partial GPU offloading.
* **Frameworks**: llama.cpp, Ollama, LM Studio, GPT4All.

|Feature|llama.cpp|Ollama|LM Studio|
|:---|:---|:---|:---|
|Core Abstraction|Bare-metal inference engine|Orchestration & API layer|All-in-one desktop application|
|Primary Interface|Command Line (CLI) & C API|CLI & REST API|Graphical User Interface (GUI)|
|Ease of Setup|Difficult (often requires compilation & manual flags)|Medium (1-click install, "Docker-like" commands)|Easy (standard desktop installer)|

### 4. ONNX (Open Neural Network Exchange)

An interoperability standard maintained by the Linux Foundation.
* **Purpose**: Facilitates model translation and execution across distinct frameworks and hardware backends (e.g., PyTorch to TensorRT).
* **Usage**: Frequently utilized for specific inference accelerators via the ONNX Runtime, though less prevalent for raw LLM weight distribution compared to Safetensors or GGUF.
* **Frameworks**: ONNX Runtime, TensorRT, OpenVINO.