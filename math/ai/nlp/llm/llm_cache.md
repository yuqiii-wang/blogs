# Cache in LLM

In agent development, most of prompts are pre-defined as instructions leaving placeholders to user for input.
This phenomenon gives rise to caching that reusable prompts' vector representation/attention calculation results can be cached.

## Typical Cache in Prefill and Decode

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

## Prefix Cache (Explained in vLLM Implementation)

1. Hashing the Prefix:

When a new request arrives, vLLM doesn't look at the raw text. Instead, it takes the sequence of input token IDs and computes a hash for them. This hash uniquely identifies the prefix.

2. vLLM maintains a central dictionary-like data structure (a hash map) that acts as the prefix cache.
    * Key: The hash of a token sequence (the prefix).
    * Value: A list of the physical block numbers that hold the KV cache for this prefix.

3. The Logic Flow: Cache Hit vs. Cache Miss
    * Cache Hit: The model only needs to run the prefill phase on the new, non-shared part of the prompt.
    * Cache Miss: It proceeds with the normal prefill operation for this prefix. As it computes the KV cache for the prefix, it stores the results in newly allocated physical blocks.

## Prompt Cache by Prompt Markup Language (PML)

Reference: https://arxiv.org/pdf/2311.04934

This proposal suggests using *Prompt Markup Language* (PML) to write prompt.
Inside PML user can attach/detach various modules and populate only required fields.

For example, a prompt is written as an xml schema served to LLM to help plan a vacation trip.
Very likely most of the prompt tokens are kept unchanged for xml structure is fixed, leaving only some placeholders for update.

<div style="display: flex; justify-content: center;">
      <img src="imgs/prompt_cache_pml.png" width="70%" height="30%" alt="prompt_cache_pml" />
</div>
</br>

### Problem and Motivation

By the chained probability output from LLM that each next token is dependent on previous tokens, token positions matter.
This gives a problem that existing prompt must be identical at same position with the same value, even in the middle of prompt a token is replaced, the whole following tokens' cache is invalidated. 

However, the empirical study from the research found that

> LLMs can operate on attention states with discontinuous position IDs.
> As long as the relative position of tokens is preserved, output quality is not affected. 

This finding enables the idea that when PML xml schema is created, user inputs are set up as placeholders as well as empty token paddings to maintain the relative token positional distance.

### Attention Masking Effect

Placeholders in prompt can be viewed as attention masking in formula.

A mask matrix $M$ is added to attention score matrix $QK^T$.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

where $M_{ij}=-\infty$ is set to negative infinity.
This makes the corresponding attention score at the $ij$-th entry also infinity.

The subsequent softmax function will then assign a probability of 0 to the positions with -∞, effectively preventing the model from attending to those tokens.

* A causal mask is used in autoregressive models to prevent a position from attending to subsequent positions.
* A padding mask is used to make the model ignore padding tokens in a batch of sequences.

$$
\text{Causal Mask}\quad
M_{ij}=\begin{cases}
    0 & \text{if } j \le i \\\\
    -\infty & \text{if } j > i
\end{cases}, \qquad \text{Padding Mask}\quad
M_{j} =
\begin{cases}
    0 & \text{if token}_j \text{ is not padding} \\\\
    -\infty & \text{if token}_j \text{ is padding}
\end{cases}
$$

