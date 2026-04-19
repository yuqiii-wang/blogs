# LLM Inference Practices

## Structure

* Encoder-Only

BERT (Bidirectional Encoder Representations from Transformers)

* Prefix-Decoder (Non-Causal Decoder)

T5 (Text-To-Text Transfer Transformer)

* Causal Decoder

GPT (Generative Pre-trained Transformer)

* Encoder-Decoder


## Prefill and Decode

In LLM inference there are two stages: prefill and decode, corresponding to pure prompt process until the 1st token is generated (prefill) and the ensued tokens are then streamed to output (decode).

### Prefill

The primary goal of prefill is to create an internal representation of the prompt's meaning and context by processing all the tokens from prompts.

It features:

* Parallel Processing: The model can work on all input tokens at once, making this phase highly parallelizable and efficient at utilizing GPU resources.
* Compute-Bound: Because it involves a large number of simultaneous calculations, the prefill phase is typically limited by the raw computational power of the hardware.

Given $Q,K\in\mathbb{R}^{n\times d_k}$, the attention score matrix is $S\in\mathbb{R}^{n\times n}$

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

This matrix multiplication can be easily parallelized.

#### Masking in Prefill

In autoregressive prefill, a causal mask $M$ is applied using the Hadamard product (element-wise multiplication, $\odot$) to prevent tokens from attending to future tokens. The mask is typically a lower-triangular matrix of 1s (allowed) and 0s (masked):

$$
S_{masked} = S + M = 
\begin{bmatrix}
s_{11} & s_{12} & \cdots & s_{1n} \\
s_{21} & s_{22} & \cdots & s_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
s_{n1} & s_{n2} & \cdots & s_{nn}
\end{bmatrix} +
\begin{bmatrix}
0 & -\infty & \cdots & -\infty \\
0 & 0 & \cdots & -\infty \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
=
\begin{bmatrix}
s_{11} & 0 & \cdots & 0 \\
s_{21} & s_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
s_{n1} & s_{n2} & \cdots & s_{nn}
\end{bmatrix}
$$

#### Example Parallel Computation in Prefill

Let prompt be "I want to buy", tokenized and embedded to $[\mathbf{t}_1, \mathbf{t}_2, \mathbf{t}_3, \mathbf{t}_4]$.
Let attention score be $a_{j,i}$ where $j$ is the target token position and $i$ is the position being attended to.
The softmax applies to the unnormalized attention scores $s_{j,i} = \frac{Q_j K_i^T}{\sqrt{d_k}}$:

$$
a_{j,i} = \text{softmax}(s_{j,i}) = \frac{\exp(s_{j,i})}{\sum_{k=1}^j \exp(s_{j,k})}
$$

This prompt in prefill is computed as

$$
\def\arraystretch{1.5}
\begin{matrix}
\mathbf{h}_1 & \mathbf{h}_2 & \mathbf{h}_3 & \mathbf{h}_4 \\
\uparrow & \uparrow & \uparrow & \uparrow \\
a_{1,1}V_1 & \sum_{j=1}^2 a_{2,j}V_j & \sum_{j=1}^3 a_{3,j}V_j & \sum_{j=1}^4 a_{4,j}V_j \\
\uparrow & \uparrow & \uparrow & \uparrow \\
\frac{Q_1 K_1^T}{\sqrt{d_k}} & \frac{Q_2 [K_1, K_2]^T}{\sqrt{d_k}} & \frac{Q_3 [K_1 \cdots K_3]^T}{\sqrt{d_k}} & \frac{Q_4 [K_1 \cdots K_4]^T}{\sqrt{d_k}} \\
\uparrow & \uparrow & \uparrow & \uparrow \\
(Q_1,K_1,V_1) & (Q_2,K_2,V_2) & (Q_3,K_3,V_3) & (Q_4,K_4,V_4) \\
\uparrow & \uparrow & \uparrow & \uparrow \\
\mathbf{t}_1 (\text{"I"}) & \mathbf{t}_2 (\text{"want"}) & \mathbf{t}_3 (\text{"to"}) & \mathbf{t}_4 (\text{"buy"})
\end{matrix}
$$

In prefill stage, there is no need of redirecting $\mathbf{h}_i$ to input as in prompt the token $\mathbf{t}_i$ is the actual input.
For prompt tokens are known then shall by masking they can be computed in parallel.

In summary, $\mathbf{h}_4$, which will be sent to prediction in inference, is computed by

$$
\mathbf{h}_4=\sum^4_{i=1}a_{4,i} V_i,\qquad a_{4,i}=\text{softmax}\Big(\frac{Q_4K_i^{\top}}{\sqrt{d_k}}\Big)
$$

This computation need input of below matrices:

* $Q_4$
* $K_1, K_2, K_3, K_4$
* $V_1, V_2, V_3, V_4$

### Decode

Once the prefill is complete and the KV cache is populated, the LLM switches to the decode phase. This is the autoregressive part of the process, where the model generates the output one token at a time.

* Sequential Operation: Each new token depends on the previously generated one, making this an inherently serial process that cannot be parallelized in the same way as the prefill phase.
* Memory-Bound: The speed of the decode phase is often limited by how quickly the model can access the growing KV cache from memory, rather than by raw computation. This is because with each new token generated, the KV cache expands, increasing the memory demands.
* Word-by-Word Generation: This is the phase where you see the response appearing token by token, creating the characteristic streaming output of many LLM applications.

Decode cannot be parallelized for each token output is conditioned on preceding already output tokens.

$$
P(X) = P(\mathtt{x}_1) \cdot P(\mathtt{x}_2 | \mathtt{x}_1) \cdot P(\mathtt{x}_3 | \mathtt{x}_1, \mathtt{x}_2) \cdots P(\mathtt{x}_T | \mathtt{x}_1, ..., \mathtt{x}_{T-1})
$$

Cache is updated via concatenation.

$$
S_1 = \frac{q_1 \cdot K_{prompt}^T}{\sqrt{d_k}}
$$

Having output token $\mathtt{x}_1$; there is ensued cache update: the model now computes the key and value vectors, $\mathtt{k}_1$ and $\mathtt{v}_1$.
For the new token $\mathtt{x}_1$. It appends them to the cache.

$$
\begin{align*}
S_2 &= \frac{\mathtt{q}_2 \cdot K_{cache}^{\top}}{\sqrt{d_k}} = \frac{\mathtt{q}_2 \cdot \text{concat}(K_{prompt}, \mathtt{k}_1)^{\top}}{\sqrt{d_k}} \\\\
S_3 &= \frac{\mathtt{q}_3 \cdot K_{cache}^{\top}}{\sqrt{d_k}} = \frac{\mathtt{q}_3 \cdot \text{concat}(K_{prompt}, \mathtt{k}_1, \mathtt{k}_2)^{\top}}{\sqrt{d_k}} \\\\
&... \\\\
S_T &= \frac{\mathtt{q}_T \cdot K_{cache}^{\top}}{\sqrt{d_k}} = \frac{\mathtt{q}_T \cdot \text{concat}(K_{prompt}, \mathtt{k}_1, \mathtt{k}_2, ..., \mathtt{k}_T)^{\top}}{\sqrt{d_k}} 
\end{align*}
$$

### Where Cache Plays A Role in LLM Inference

* Without Cache: To generate the $100$-th token, the model would need to process the prompt $+99$ previous tokens. This is incredibly redundant and computationally expensive, with costs that grow quadratically with the sequence length.
* With Cache: To generate the $100$-th token, the model only processes the $99$-th token and uses the cache to instantly access the context from the prompt and the first 98 tokens. This changes the problem from re-calculating everything to performing one small calculation and one append operation.


## LLM Memory Consumption During Inference

There are

* Model Parameters
* Key Value Caches
* Temporary Computation Results

### Model Parameters

Take BERT-base as an example.
In total, there are $108{,}369{,}656$ parameters.
By FP16, the model memory consumption is $108{,}369{,}656 \times 2\text{ bytes} = 216.7\text{ MB}$.

#### Embedding Layers

* Token Embeddings $30{,}000 \times 768 = 23{,}040{,}000$
* Position Embeddings: $512 \times 768 = 393{,}216$ for a maximum sequence length (commonly 512)
* Token Type Embeddings: $2 \times 768 = 1{,}536$ for 2 token types (for sentence A and sentence B)

#### Transformer Layer Components

For each of $12$ transformer layers, there are

* Query, Key, and Value weights: $3 \times 768 \times 768 = 1{,}769{,}472$
* Attention Output Linear Projection: $768 \times 768 = 589{,}824$
* Feed-Forward: $768 \times 3072 + 3072 \times 768 = 4{,}718{,}592$

### Key Value Caches

The key $K$ and value $V$ of the $\text{Attention}(Q,K,V)$ are stored for previous tokens for next token prediction.

For example, assumed model has already processed $128$ tokens, base on which to predict the $129$-th token.

For ONE layer of BERT-base, there is

$$
K\in\mathbb{R}^{\text{numHeads}\times\text{seqLen}\times\text{headDim}}=\mathbb{R}^{12\times 128\times 64} \\
V\in\mathbb{R}^{\text{numHeads}\times\text{seqLen}\times\text{headDim}}=\mathbb{R}^{12\times 128\times 64}
$$

These caches are maintained per layer. Thus, there are $12$ independent pairs of caches (one pair per layer).

For 4k context length with FP16, there is KV cache $2 \times 12 \times 12 \times 4096 \times 64 \times 2\text{ bytes} = 144\text{ MB}$.

### Temporary Intermediate Computation Results

Temporary computation results are used only on the current layer, and on the next layer the intermediate values are re-computed.

Again take BERT-base as example, for each head $h=1,2,...,12$ in ONE layer, given $128$ already predicted tokens, there is

* Raw Attention Score

$$
S_{h} = \frac{Q_h K_h^{\top}}{\sqrt{64}} \in \mathbb{R}^{128}
$$

* Attention Score Softmax Normalization

$$
a_{h,i} = \frac{\exp(S_{h,i})}{\sum_{i=1}^{128}\exp(S_{h,i})},
\qquad a_{h} \in \mathbb{R}^{128}
$$

* Weighted Sum over Values

$$
O_h = \sum^{128}_{i=1}a_{h,i}V_{h,i} \in \mathbb{R}^{64}
$$

* Output Concatenation for all $12$ heads

$$
O = \text{Concat}(O_1, O_2, ..., O_{12}) \in \mathbb{R}^{12\times 64}
$$

* Compute the new $K$ and $V$ for the $129$-th token for all $12$ heads, and add them to KV Caches

$$
K_{i=129} \in\mathbb{R}^{12\times 64} \\
V_{i=129} \in\mathbb{R}^{12\times 64}
$$

## The Problem of Token Repetition

The probability of a token $t_{i+1}$ being selected takes into consideration of all previous tokens $P(t_{i+1}|t_1,t_2,...,t_i)$.

* Contextual Bias: If the model has seen similar patterns during training (e.g., repetitive phrases like "hello hello hello"), it may overestimate the probability of repeating certain tokens.
* Overconfidence in Token Selection: The model might select the most probable token repeatedly, leading to a loop.

### The Root Cause: Attention Sink

Revisit the attention formula. At step $n$, the model generates a query $\mathbf{q}_{n}$. It compares this against all previous keys $\mathbf{k}_i$ for $1 \le i < n$.

$$
\text{score}(n,i) = \frac{\mathbf{q}_{n}^{\top}\mathbf{k}_i}{\sqrt{d}}
$$

Assumed RoPE as token embedding, then let $\mathbf{q}_{n}=R_{n}\mathbf{q}_1$ and $\mathbf{k}_i=R_{i}\mathbf{k}_1$ so that their position info is represented via rotation matrices $R_{n}$ and $R_{i}$, there is

$$
\max \text{score}(\mathbf{q}_{n}, \mathbf{k}_i) = (R_{n} \mathbf{q}_1)^{\top} (R_{i} \mathbf{k}_1) = \mathbf{q}_1^{\top} R_{n}^{\top} R_{i} \mathbf{k}_1 = \mathbf{q}_1^{\top} R_{n-i} \mathbf{k}_1
$$

$R_{n-i}$ represents the distance between the query token $\mathbf{q}_{n}$ vs history key token $\mathbf{k}_i$.
The subscript $\space_{1}$ represents token sequence position if both query and key tokens are aligned to the same $1$-st position.
Further decompose $R_{n-i}$ can see that within LLM embedding context length the smaller the $|n-i|$, the higher the $\text{score}(n,i)$.

The above explanation shows one proof that recent tokens have much higher weights influencing next token output.

Then, consider value matrix $V$ and residual mixing, where $\mathbf{v}^{(l)}_{i-1}$ is used to encode next layer token $\mathbf{v}^{(l+1)}_{i}$ with one position left-shift.


#### Example: Periodic Token Repetition in CSV Manipulation

CSV data is inherently periodic: **commas appear at fixed intervals** (e.g., every $N$ positions separating fields). When a model operates on CSV, these periodic tokens create a **striped attention pattern** that amplifies collapse.

**Example CSV sequence:**
```
name, age, city, name, age, city, ...
  ^     ^    ^     ^    ^    ^
  |     |    |     |    |    |
  0     4    8    12   16   20  (approximate positions with period=4)
```

**How RoPE Position Embeddings Interact with Periodicity:**

Recall from the RoPE formula, the attention score between two tokens depends on their **relative distance** via rotation:

$$
\text{score}(\mathbf{q}_m, \mathbf{k}_n) = \mathbf{q}_1^{\top} R_{n-m} \mathbf{k}_1
$$

If token "comma" appears at positions $\{0, 4, 8, 12, ...\}$ (period $p=4$), then:
- Position 0 ["," at index 0] attends to position 4 ["," at index 4]: relative distance $= 4$
- Position 4 ["," at index 4] attends to position 8 ["," at index 8]: relative distance $= 4$
- Position 8 ["," at index 8] attends to position 12 ["," at index 12]: relative distance $= 4$

Since the relative **rotation angle** $R_4$ is identical for all these pairs, the attention scores form **periodic bands** in the attention matrix.

**Why Identical Relative Distance = Identical Rotation = Identical Attention Scores:**

The RoPE rotation matrix $R_d$ depends only on the **relative distance** $d$ between two positions, not their absolute positions. This is the key insight. For any two positions separated by distance $d = n - m$:

$$
R_{n-m} = \begin{bmatrix} \cos(d\theta_1) & -\sin(d\theta_1) & 0 & \cdots \\ \sin(d\theta_1) & \cos(d\theta_1) & 0 & \cdots \\ 0 & 0 & \cos(d\theta_2) & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{bmatrix}
$$

In CSV with period $p = 4$:
- Pair (position 0, position 4): distance = 4, rotation = $R_4$
- Pair (position 4, position 8): distance = 4, rotation = $R_4$
- Pair (position 8, position 12): distance = 4, rotation = $R_4$

Since all comma-to-comma pairs have **identical distance**, they compute the **identical rotation matrix** $R_4$. Thus, the dot product $\mathbf{q}_1^{\top} R_4 \mathbf{k}_1$ yields the **same numerical value** for all these token pairs. This is why the attention matrix develops **striped columns**: every row computes attention to the same distant-relative positions, producing identical high values at periodic intervals.

Here's a $12 \times 12$ expansion to make the pattern obvious:

$$
S = \frac{QK^T}{\sqrt{d_k}} = \begin{bmatrix}
a_{\mathbf{high}} & a_{0,1} & a_{0,2} & a_{0,3} & a_{\mathbf{high}} & a_{0,5} & a_{0,6} & a_{0,7} & a_{\mathbf{high}} & a_{0,9} & a_{0,10} & a_{0,11} \\
a_{\mathbf{high}} & a_{1,1} & a_{1,2} & a_{1,3} & a_{\mathbf{high}} & a_{1,5} & a_{1,6} & a_{1,7} & a_{\mathbf{high}} & a_{1,9} & a_{1,10} & a_{1,11} \\
a_{\mathbf{high}} & a_{2,1} & a_{2,2} & a_{2,3} & a_{\mathbf{high}} & a_{2,5} & a_{2,6} & a_{2,7} & a_{\mathbf{high}} & a_{2,9} & a_{2,10} & a_{2,11} \\
a_{\mathbf{high}} & a_{3,1} & a_{3,2} & a_{3,3} & a_{\mathbf{high}} & a_{3,5} & a_{3,6} & a_{3,7} & a_{\mathbf{high}} & a_{3,9} & a_{3,10} & a_{3,11} \\
a_{\mathbf{high}} & a_{4,1} & a_{4,2} & a_{4,3} & a_{\mathbf{high}} & a_{4,5} & a_{4,6} & a_{4,7} & a_{\mathbf{high}} & a_{4,9} & a_{4,10} & a_{4,11} \\
a_{\mathbf{high}} & a_{5,1} & a_{5,2} & a_{5,3} & a_{\mathbf{high}} & a_{5,5} & a_{5,6} & a_{5,7} & a_{\mathbf{high}} & a_{5,9} & a_{5,10} & a_{5,11} \\
a_{\mathbf{high}} & a_{6,1} & a_{6,2} & a_{6,3} & a_{\mathbf{high}} & a_{6,5} & a_{6,6} & a_{6,7} & a_{\mathbf{high}} & a_{6,9} & a_{6,10} & a_{6,11} \\
a_{\mathbf{high}} & a_{7,1} & a_{7,2} & a_{7,3} & a_{\mathbf{high}} & a_{7,5} & a_{7,6} & a_{7,7} & a_{\mathbf{high}} & a_{7,9} & a_{7,10} & a_{7,11} \\
a_{\mathbf{high}} & a_{8,1} & a_{8,2} & a_{8,3} & a_{\mathbf{high}} & a_{8,5} & a_{8,6} & a_{8,7} & a_{\mathbf{high}} & a_{8,9} & a_{8,10} & a_{8,11} \\
a_{\mathbf{high}} & a_{9,1} & a_{9,2} & a_{9,3} & a_{\mathbf{high}} & a_{9,5} & a_{9,6} & a_{9,7} & a_{\mathbf{high}} & a_{9,9} & a_{9,10} & a_{9,11} \\
a_{\mathbf{high}} & a_{10,1} & a_{10,2} & a_{10,3} & a_{\mathbf{high}} & a_{10,5} & a_{10,6} & a_{10,7} & a_{\mathbf{high}} & a_{10,9} & a_{10,10} & a_{10,11} \\
a_{\mathbf{high}} & a_{11,1} & a_{11,2} & a_{11,3} & a_{\mathbf{high}} & a_{11,5} & a_{11,6} & a_{11,7} & a_{\mathbf{high}} & a_{11,9} & a_{11,10} & a_{11,11}
\end{bmatrix}
$$

In this $12 \times 12$ matrix (for a period-4 repeating token), the **bolded entries** form **perfect vertical stripes** at columns $\{0, 4, 8\}$—exactly where the periodic comma tokens appear. Every row has identical patterns of high attention values, creating a rigid striped structure that dominates the entire attention matrix.

**The Collapse Cascade:**

Once the model generates a "comma" token at position $n$:

1. **Layer $\ell_0$ (First Attention Layer)**: The comma's embedding receives high self-attention because all commas at previous positions had similar embeddings. The attention weight converges to the comma's own embedding $\mathbf{v}_{\text{comma}}$.

2. **Context Vector Dominance**: The context vector $\mathbf{c}_n$ becomes dominated by the comma's embedding:
$$
\mathbf{c}_n = \sum_{i=0}^{n} \alpha_i \mathbf{v}_i \approx \alpha_{\text{comma}} \mathbf{v}_{\text{comma}}, \quad \alpha_{\text{comma}} \to 1
$$

**Why This Dominance Occurs:**

The context vector is a **weighted sum** of all value embeddings from previous tokens:

$$
\mathbf{c}_n = \sum_{i=0}^{n} \alpha_i \mathbf{v}_i
$$

where $\alpha_i = \text{softmax}(S_i)$ are the attention weights summing to 1: $\sum_{i=0}^{n} \alpha_i = 1$.

Due to the striped attention matrix pattern, **comma positions receive extremely high attention scores** (e.g., $S_{\text{comma}} \approx 50$) while **non-comma positions receive low scores** (e.g., $S_{\text{other}} \approx -5$). When passed through softmax:

$$
\alpha_{\text{comma}} = \frac{\exp(50)}{\exp(50) + \sum_{i \neq \text{comma}} \exp(-5)} \approx \frac{\exp(50)}{\exp(50)} \approx 0.9999...
$$

Meanwhile, all non-comma weights collectively receive:

$$
\sum_{i \neq \text{comma}} \alpha_i \approx 1 - 0.9999 \approx 0.0001
$$

Thus the context vector becomes:

$$
\mathbf{c}_n \approx 0.9999 \cdot \mathbf{v}_{\text{comma}} + 0.0001 \cdot (\text{negligible contributions from other tokens}) \approx \mathbf{v}_{\text{comma}}
$$

**The Information Bottleneck:**

Only the comma's embedding information flows forward. All other semantic content from surrounding tokens (like "name", "age", "city") is effectively **erased** from the context. The next layer receives an input that is almost purely "comma-like", losing all diversity and contextual nuance.

3. **Next Token Bias**: The next token logit is computed as:
$$
z_{n+1} = \text{Linear}(\mathbf{c}_n) \approx \text{Linear}(\mathbf{v}_{\text{comma}})
$$

Since $\mathbf{v}_{\text{comma}}$ is the **only large contributor**, the model computes logits primarily from the comma embedding. Because the comma was trained to predict the next token after commas in CSV, the logit for "comma" becomes high.

4. **Positive Feedback Loop (All Layers)**: In layer $\ell_1$, the model once again attends strongly to its own comma output (from $\ell_0$), triggering the same collapse. This repeats through all $L$ layers, creating a **runaway positive feedback loop** where each layer amplifies the comma probability.

5. **Softmax Saturation**: By the output layer, the softmax becomes degenerate: $\text{softmax}([..., 50, -20, -15, ...]) \approx [1, 0, 0, ...]$, where only the comma token has non-zero probability.

**Why CSV is Particularly Vulnerable:**

CSV sequences have **maximum periodicity**: the same tokens (comma, newline) reappear in **exact rhythmic patterns**, which is far more likely to trigger attention collapse than natural language. In prose, tokens rarely repeat at such regular intervals, so the feedback loop is normally disrupted by semantic diversity.

**Mathematical Insight:**

The **eigenvalue structure** of the recurring attention pattern matters. If $S$ (the attention matrix) has a dominant eigenvalue $\lambda_1$ corresponding to the periodic structure, then repeated matrix multiplications in subsequent layers cause the eigenvector magnification:

$$
S^L \mathbf{v}_{\text{comma}} \approx \lambda_1^L \mathbf{v}_{\text{comma}}
$$

For large $L$ (deep networks) and $\lambda_1 > 1$ (positive feedback), this grows **exponentially**, guaranteeing collapse.


### Non-Training Mitigation Solutions

Such options are helpful in mitigating the repeated token generation.

```py
from transformers import GPT2LMHeadModel

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")

outputs = model.generate(input_ids, 
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=2.0)
```

#### Agentic Action (BEST)

Let AI agent write a local tool, e.g., python pandas script, to read csv file and to csv manipulation by this tool.

#### Temperature Scaling

Temperature scaling is a common method for controlling randomness in predictions.
Given temperature $T$, for token prediction by softmax, there is

$$
t_i = \frac{\exp(\frac{\text{logit}_i}{T})}{\sum_{j=1}^n\exp(\frac{\text{logit}_j}{T})}
$$

* High Temperature $T > 1$: Increases randomness by flattening the distribution. The logits are scaled down, causing the difference between the probabilities of different tokens to become smaller. The results are more diverse.
* Low Temperature $T < 1$: Increases determinism by sharpening the distribution. The logits are amplified, causing higher-probability tokens to become more dominant. This results in more predictable, conservative outputs.
* Temperature $T = 1$: The distribution remains unchanged, as it represents the default probability scale from the model.
* Temperature $T = 0$: $\frac{\text{logit}_i}{T}$ becomes extremely large for the highest logit, and the other logits become negligible. The model will produce the same output every time for a given input, as it always selects the most probable token.

#### Penalty for Repetition

Introduce a hyperparameter $\lambda$ that controls the strength of the penalty to adjust logit.

For next token $t_{i+1}$ prediction, let $\hat{t}_{i+1}$ be the $\text{logit}_{i+1}$ supposed corresponding prediction by softmax.
After having adjusted as per $\hat{\text{logit}}_{i+1}$, this new logit might predict a new token different from the old one.

$$
\hat{\text{logit}}_{i+1} = \text{logit}_{i+1}(\hat{t}_{i+1}) - \lambda \cdot 1(\hat{t}_{i+1} = t_{1}, \hat{t}_{i+1} = t_{2}, \ldots, \hat{t}_{i+1} = t_{i})
$$

where

* $1(.)$ is the indicator function that checks if the token has already appeared in the sequence.

#### Top-k and Top-p Sampling

Top-k sampling restricts the selection of the next token to the top $k$ tokens with the highest probabilities.

$$
P_{\text{top-k}}(t_{i+1}|t_1,t_2,...,t_i)=\begin{cases}
    \frac{\exp(\frac{\text{logit}_i}{T})}{\sum_{j=1}^n\exp(\frac{\text{logit}_j}{T})} & \text{if } t_{i+1} \in \text{top-k} \\\\
    0 & \text{otherwise}
\end{cases}
$$

Top-p sampling restricts the selection by a defined cut-off threshold.

$$
P_{\text{top-p}}(t_{i+1}|t_1,t_2,...,t_i) > p_{\text{threshold}}
$$
