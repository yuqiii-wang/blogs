# Ollama and llama.cpp: Architecture and Inference

Ollama is a lightweight, extensible framework designed to run Large Language Models (LLMs) locally on **GPU-limited edge devices**. Under the hood, Ollama leverages **llama.cpp**, a highly optimized C/C++ inference engine. This foundation allows Ollama to execute models efficiently across various hardware configurations, utilizing quantized GGUF (GPT-Generated Unified Format) files.

## 1. Partial Model Loading (GPU Offloading)

Modern LLMs based on the Transformer architecture consist of $N$ sequential transformer blocks (layers). When a system lacks sufficient Video RAM (VRAM) to hold the entire model, `llama.cpp` utilizes **partial model loading** (GPU offloading).

Let $N$ be the total number of layers. The model is partitioned such that:
*   $k$ layers are loaded into GPU VRAM.
*   $N - k$ layers remain in system RAM.

### Computation Flow

During the forward pass for a given input tensor $X_0$, the computation proceeds sequentially:
$$ X_{i} = f_i(X_{i-1}) \quad \text{for } i = 1, 2, \dots, N $$
where $f_i$ represents the $i$-th transformer layer.

1.  **GPU Execution:** For $i \le k$, operations (e.g., matrix multiplications for queries, keys, values, and feed-forward networks) execute on the GPU.
2.  **Data Transfer:** The intermediate activation tensor $X_k$ is copied from GPU VRAM to system DDR RAM. Memory bandwidth (PCIe) becomes the bottleneck here.
3.  **CPU Execution:** For $i > k$, computations continue on the CPU.

This hybrid architecture enables running models larger than available VRAM, trading off inference speed (CPU computation and PCIe transfer latency) for memory capacity.

### Multi-GPU Support

When multiple GPUs are available, `llama.cpp` aggregates their VRAM and distributes work via two strategies: **layer-wise distribution** and **tensor parallelism**. Ollama detects all CUDA/ROCm/Metal devices automatically and applies layer-wise distribution by default.

#### Layer-Wise Distribution (Pipeline Parallelism)

Each GPU owns a contiguous slice of transformer layers. With $G$ GPUs and $N$ total layers, GPU $g$ (zero-indexed) is assigned layers in the range:

$$\left[\left\lfloor g \cdot \frac{N}{G} \right\rfloor,\ \left\lfloor (g+1) \cdot \frac{N}{G} \right\rfloor - 1\right]$$

`llama.cpp` computes the assignment proportionally to each GPU's available VRAM so that memory is balanced rather than layers split evenly. The forward pass becomes a pipeline:

```
GPU 0: layers 0..k₀   →  activation X_{k₀}
         ↓  (PCIe / NVLink transfer)
GPU 1: layers k₀+1..k₁ →  activation X_{k₁}
         ↓
  ...
GPU G-1: layers k_{G-2}+1..N-1  →  logits
```

**Characteristics:**
- Only one activation tensor is in flight between adjacent GPUs at any moment — low inter-GPU bandwidth requirement.
- GPUs execute **sequentially**, so only one GPU is active at a time per token step. Effective utilization per GPU ≈ $\frac{1}{G}$ of wall-clock time.
- **Latency** scales roughly linearly with $G$ for a single request, but **throughput** scales well under many concurrent requests via pipeline filling.
- Default strategy in Ollama; controlled via `--tensor-split` CLI flag or `CUDA_VISIBLE_DEVICES` ordering.

#### Tensor Parallelism

Each transformer layer is split *across* all GPUs simultaneously. The key weight matrices — query/key/value projections $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ and feed-forward $W_1, W_2$ — are column- or row-partitioned:

$$W = \begin{bmatrix} W^{(0)} \mid W^{(1)} \mid \cdots \mid W^{(G-1)} \end{bmatrix}$$

Each GPU $g$ computes its shard $Y^{(g)} = X \cdot W^{(g)}$, then an **all-reduce** (sum) synchronizes the partial results:

$$Y = \sum_{g=0}^{G-1} Y^{(g)}$$

**Characteristics:**
- All GPUs are active **in parallel** for every layer → much higher GPU utilization per token step.
- Requires an all-reduce after every attention and FFN sub-layer — demands **high inter-GPU bandwidth** (NVLink >> PCIe).
- Reduces peak VRAM per GPU for large weight matrices by a factor of $G$.
- Latency for a single request improves with more GPUs (unlike pipeline parallelism).
- Enabled in llama.cpp via `--tensor-split` with equal weights and compiled with `GGML_CUDA_FORCE_MMQ` or `GGML_SYCL`; not yet exposed as a first-class Ollama option — requires direct `llama-server` invocation.

#### Strategy Comparison

| | Layer-wise | Tensor Parallel |
|---|---|---|
| GPU activity per step | Sequential (1 of G active) | All G active in parallel |
| Inter-GPU bandwidth need | Low | High (all-reduce per layer) |
| Single-request latency | Worse with more GPUs | Better with more GPUs |
| Throughput under load | Good (pipeline) | Good (parallel) |
| Best hardware | PCIe multi-GPU | NVLink / NVSwitch systems |
| Ollama default | Yes | Manual (`llama-server`) |

### Concurrency Support

Ollama handles multiple simultaneous requests through a **serial queue with KV-cache sharing**:

- **Request queue.** Incoming requests are queued per model. By default only **one request runs at a time** per loaded model instance (`OLLAMA_NUM_PARALLEL=1`). Setting `OLLAMA_NUM_PARALLEL > 1` enables true parallel decoding by batching tokens from multiple sequences together in a single `llama_decode` call — at the cost of higher VRAM usage (each active sequence maintains its own KV cache slot).

- **Same-prompt KV cache hit.** Because the KV cache is **process-scoped** (see §4), concurrent requests that share a common prefix (e.g., a fixed system prompt) all reuse the same cached KV entries after the first request warms the cache. This dramatically reduces prefill latency for subsequent requests.

- **Continuous batching.** When `OLLAMA_NUM_PARALLEL > 1`, llama.cpp uses **continuous batching**: new requests are inserted into the active batch as soon as a slot is free, rather than waiting for the entire batch to finish. This keeps GPU utilization high under mixed-length workloads.

- **Model multiplexing.** Multiple *different* models can be kept loaded simultaneously (`OLLAMA_MAX_LOADED_MODELS`, default 1 on CPU / 3 on GPU). Requests for different models are served from their respective in-memory instances without reload overhead, subject to total VRAM availability.

| Env variable | Default | Effect |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `1` | Max concurrent sequences per model |
| `OLLAMA_MAX_LOADED_MODELS` | `1` / `3` | Max simultaneously loaded models |
| `OLLAMA_KEEP_ALIVE` | `5m` | Idle timeout before model unload |

## 2. Memory Consumption Model

To deploy LLMs effectively, it is critical to calculate the memory bounds. The total memory consumption ($M_{total}$) comprises three primary components:

$$ M_{total} = M_{weights} + M_{KV} + M_{activations} $$

### 2.1 Weight Memory ($M_{weights}$)

The memory required for model parameters depends on the total parameter count $P$ and the quantization bit-width $b_w$ (e.g., 4 bits for Q4, 16 bits for FP16).

$$ M_{weights} = P \times \frac{b_w}{8} \text{ (bytes)} $$

*Example:* A 7B parameter model quantized to 4-bit (Q4_0):
$$ M_{weights} = 7 \times 10^9 \times \frac{4}{8} = 3.5 \text{ GB} $$

### 2.2 KV Cache Memory ($M_{KV}$)
Autoregressive generation requires caching the Key (K) and Value (V) tensors for all previous tokens to avoid recomputation.

Let:
*   $C_{max}$: Maximum context length (tokens)
*   $N$: Number of layers
*   $H$: Number of attention heads
*   $d$: Dimension per head
*   $b_{kv}$: Bit-width for KV cache (typically 16-bit FP16, $b_{kv} = 16$)

The KV cache for a single token uses:
$$ M_{token} = 2 \times N \times H \times d \times \frac{b_{kv}}{8} \text{ (bytes)} $$
*(The factor of $2$ accounts for both Key and Value matrices).*

The total KV cache memory for the entire context window is:
$$ M_{KV} = 2 \times N \times H \times d \times C_{max} \times \frac{b_{kv}}{8} \text{ (bytes)} $$

### 2.3 Activation Memory ($M_{activations}$)
During the forward pass, temporary tensors (activations) are allocated. For a batch size of $B$ and sequence length $S$, the peak activation memory scales roughly with $\mathcal{O}(B \times S \times d_{model})$. While smaller than weights and KV cache during text generation, it can spike massively during the pre-fill phase.

### Summary of VRAM Requirements for Partial Loading
When $k$ out of $N$ layers are offloaded to the GPU, the required VRAM ($V_{req}$) scales nearly linearly with the fraction of offloaded layers:

$$ V_{req} \approx \left( \frac{k}{N} \times M_{weights} \right) + \left( \frac{k}{N} \times M_{KV} \right) + M_{buffer} $$

where $M_{buffer}$ accounts for CUDA contexts and activation buffers (typically 0.5 GB - 1 GB).

## 3. Common APIs

Ollama exposes a REST API on `http://localhost:11434`. All request and response bodies are JSON. Streaming endpoints emit newline-delimited JSON (NDJSON) with `"done": false` on intermediate chunks and `"done": true` on the final chunk.

### 3.1 `GET /api/tags` — List Local Models

Returns all models currently pulled to the local host.

```python
import requests

models = requests.get("http://localhost:11434/api/tags").json()["models"]
for m in models:
    print(m["name"], m["size"])
```

### 3.2 `POST /api/generate` — Single-Turn Text Generation

Accepts a raw `prompt` string and returns completions token by token (streaming by default).

```python
import requests, json

resp = requests.post("http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": "Explain KV cache.", "stream": True},
    stream=True)

for line in resp.iter_lines():
    chunk = json.loads(line)
    print(chunk["response"], end="", flush=True)
    if chunk["done"]:
        break
```

Set `"stream": false` to receive a single JSON object with the complete response.

### 3.3 `POST /api/chat` — Multi-Turn Chat

Follows the OpenAI-style `messages` format with `role` and `content` fields. This is the idiomatic endpoint for conversational agents.

```python
import requests, json

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is GGUF?"}
]

resp = requests.post("http://localhost:11434/api/chat",
    json={"model": "llama3", "messages": messages, "stream": False})

reply = resp.json()["message"]["content"]
```

### 3.4 `POST /api/embed` — Embeddings

Generates dense vector representations for one or more texts. Use a dedicated embedding model for retrieval tasks.

```python
import requests

resp = requests.post("http://localhost:11434/api/embed",
    json={"model": "nomic-embed-text", "input": ["The sky is blue", "VRAM is limited"]})

vectors = resp.json()["embeddings"]  # list[list[float]], shape (2, 768)
```

`"input"` accepts a single string or a list of strings for batch encoding.

### 3.5 `GET /api/ps` — Inspect Running Models and Queue State

Returns all models currently loaded in memory, along with their active request counts and VRAM usage. This is the primary way to observe how many requests are in flight per model.

```python
import requests

data = requests.get("http://localhost:11434/api/ps").json()
for m in data["models"]:
    print(m["name"],
          "size_vram:", m["size_vram"],
          "expires_at:", m["expires_at"])
```

**Polling for queue depth.** Ollama does not expose a dedicated queue-length field, but you can infer load by polling `/api/ps` and counting loaded models, then correlating with response latency.

### 3.6 `GET metrics` Metrics

```python
import requests

# Raw Prometheus text exposition
metrics = requests.get("http://localhost:11434/metrics").text
# Look for: ollama_request_duration_seconds, ollama_pending_requests_total
for line in metrics.splitlines():
    if "pending" in line or "request" in line:
        print(line)
```

Key metrics:

| Metric | Meaning |
|---|---|
| `ollama_pending_requests_total` | Requests currently waiting in the queue |
| `ollama_active_requests_total` | Requests being processed right now |
| `ollama_request_duration_seconds` | Histogram of end-to-end latency |

---

## 4. KV Cache Reuse Across Requests

Ollama (via llama.cpp) supports **prompt caching**: the KV cache computed for a prompt prefix is retained in memory between requests so that a subsequent request beginning with the same prefix can skip re-computing those tokens.

**How it works.** Let a conversation consist of turns $T_1, T_2, \dots$ where each turn appends tokens to the context. On the second turn the already-computed KV entries for positions $1, \dots, \ell$ (the shared prefix of length $\ell$) are read directly from cache; only the new tokens at positions $\ell+1, \dots$ are computed:

$$\text{prefill cost}_{T_2} = (L_{T_2} - \ell) \times C_{layer} \quad \ll \quad L_{T_2} \times C_{layer}$$

where $L_{T_i}$ is the total context length at turn $i$ and $C_{layer}$ is the per-token, per-layer compute cost.

**Conditions for cache hit:**
- The new prompt must **start with the exact same token sequence** as the cached prefix.
- The same model and context parameters (`num_ctx`, quantization) must be in use.
- The model must still be loaded in memory (Ollama unloads models after a configurable idle timeout, default 5 minutes; set `OLLAMA_KEEP_ALIVE` to control this).

**Cross-request scope.** Cache reuse is **process-scoped**, not session-scoped. Any client that sends the same prefix to the same running Ollama process benefits from the cache — there is no per-session isolation. This makes system-prompt caching particularly effective: a fixed system prompt shared across many requests will almost always be a cache hit after the first request.

**Limitation.** The cache is not persisted to disk; a model reload discards all cached KV entries. For very long documents, the entire prefix must fit within the configured `num_ctx` window to be cacheable.

## 5. Vision Language Models (VLMs) and ViT Support

Ollama natively supports **Vision Language Models (VLMs)** that incorporate Vision Transformers (ViT).
### How it Works

A VLM in Ollama typically consists of three components packaged together in the GGUF file:
1. **Vision Encoder (ViT)**: Often based on CLIP (Contrastive Language-Image Pre-training). It slices the input image into patches and extracts visual embeddings.
2. **Cross-Modal Projector**: A neural network layer (usually a linear projection or MLP) that maps the visual embeddings from the ViT's latent space into the LLM's text embedding space.
3. **Large Language Model (LLM)**: A standard autoregressive LLM (like Llama 3 or Mistral) that processes the combined sequence of visual and text tokens.

When an image is provided in the API request, Ollama offloads the image processing to the ViT (which is also GPU-accelerated), generates the sequence of visual tokens, and prepends/inserts them into the text prompt before passing the entire context to the LLM.

### API Usage

You can pass images to models that support vision (e.g., `llava`) using the `/api/generate` or `/api/chat` endpoints by including base64-encoded image data in the `images` array.

```python
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "model": "llava",
    "prompt": "Describe this image in detail.",
    "images": [encode_image("sample.jpg")],
    "stream": False
}

response = requests.post("http://localhost:11434/api/generate", json=payload)
print(response.json()["response"])
```

