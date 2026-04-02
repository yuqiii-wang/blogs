# Vectorization

In human linguistics, tokens can be considered the minimalist information unit


## From Token ID to Vector Representation

### One-Hot Representation

A token with ID $i$ in vocabulary of size $V$ is represented as a vector $\mathbf{e}_i \in \{0,1\}^V$ where:
$$\mathbf{e}_i[j] = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}$$

For example, if "restaurant" has ID 3 in a 5-word vocabulary: $\mathbf{e}_3 = [0, 0, 1, 0, 0]$.

**Sparse Matrix Example**: For the sentence "I like restaurants", with vocabulary $V=8$:

$$\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}$$

- Row 1: token "I" (ID 1) → one-hot with 1 at position 1
- Row 2: token "like" (ID 3) → one-hot with 1 at position 3
- Row 3: token "restaurants" (ID 6) → one-hot with 1 at position 6

The $3 \times 8$ matrix is sparse with only 3 non-zero entries; for realistic $V=5000$, each row would have 4999 zeros.

**Drawback**: High dimensionality, orthogonal but semantically uninformative (doesn't capture meaning).

### TF-IDF

For example, the word "restaurants" has the below attributes:

* isNoun: $\{0, 0, 0, 1, 0\}$ for $\{\text{isVerb}, \text{isAdjective}, \text{isPronoun}, \text{isNoun}, \text{isAdverb}\}$
* isPlural: $\{1\}$ for $\{\text{isPlural}\}$  
* synonyms: $\{ 5623, 1850, 2639 \}$ (vocabulary index) for $\{ \text{hotel}, \text{bar}, \text{club} \}$
* antonyms: $\emptyset$
* frequent occurrences under what topics: $\{ 1203, 5358, 1276 \}$ (vocabulary index) for $\{ \text{eating}, \text{outing}, \text{gathering} \}$
* Term frequency-inverse document frequency (TF-IDF): $\{ 0.016, 0.01, 0.0, 0.005 \}$ , formula:
  * $\text{TF-IDF}_j = \text{Term Frequency}_{i,j} \times \text{Inverse Document Frequency}_{i}$, where
  * $\text{Term Frequency}_{i,j} = \frac{\text{Term i frequency in document j}}{\text{Total no. of terms in document j}}$
  * $\text{Inverse Document Frequency}_{i} = \log \frac{\text{Total no. of documents}}{\text{No. of documents containing term i}}$

Given the four sentences/documents,

```txt
There are many popular restaurants nearby this church.
Some restaurants offer breakfasts as early as 6:00 am to provide for prayers.
"The price and taste are all good.", said one prayer who has been a frequent visitor to this church since 1998.
However, Covid-19 has forced some restaurants to shut down for lack of revenue during the pandemic, and many prayers are complained about it.
```

The TF-IDF per sentence/document is computed as below.

|No.|Token|Term count (Doc 1)|Term count (Doc 2)|Term count (Doc 3)|Term count (Doc 4)|Document count|IDF|TF $\times$ IDF (Doc 1)|TF $\times$ IDF (Doc 2)|TF $\times$ IDF (Doc 3)|TF $\times$ IDF (Doc 4)|
|-|-|-|-|-|-|-|-|-|-|-|-|
|1|many|0.125|0|0|0.04348|2|0.301|0.038|0|0|0.013|
|2|popular|0.125|0|0|0|1|0.602|0.075|0|0|0|
|3|restaurants|0.125|0.07692|0|0.04348|3|0.125|0.016|0.01|0|0.005|
|4|nearby|0.125|0|0|0|1|0.602|0.075|0|0|0|
|5|church|0.125|0|0.04762|0|2|0.301|0.038|0|0.014|0|
|6|offer|0|0.07692|0|0|1|0.602|0|0.046|0|0|

For compression, one popular approach is encoder/decoder, where dataset is fed to machine learning study.

For example, by placing "restaurants" and "bar" together in a text dataset that only describes food, likely the attribute "topic" might have little information hence neglected (set to zeros) in vector.

### BM25

BM25 improves TF-IDF with:

1. **TF Saturation**: TF score saturates instead of growing linearly. For a term $t$ in document $d$:
$$\text{BM25-TF}(t,d) = \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$
where $f(t,d)$ is term frequency, $|d|$ is document length, $\text{avgdl}$ is average document length, $k_1 \approx 1.5$ and $b \approx 0.75$ are tuning parameters.

2. **Document Length Normalization**: The factor $(1 - b + b \cdot \frac{|d|}{\text{avgdl}})$ normalizes for document length—longer documents don't automatically score higher just for having more words.

3. **Full BM25 Score**:
$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

**Tuning Parameter Effects**:
- **$k_1$ (typically 1.5)**: Controls TF saturation. Higher $k_1$ → term frequency matters more (slower saturation); lower $k_1$ → diminishing returns kick in faster. Adjust higher for keyword matching, lower for content-rich documents.
- **$b$ (typically 0.75)**: Controls length normalization intensity. $b=1.0$ → full normalization (penalizes long docs more); $b=0$ → no normalization (long docs get boosted). Higher $b$ favors uniform-length documents, lower $b$ rewards longer documents with more content.

**Advantage over TF-IDF**: Handles term frequency more realistically (diminishing returns) and adapts for document length, improving ranking quality in information retrieval.

## Embedding and Contextualization

* Embedding is learned via encoder to use a dense and semantic vector to represent a token.
* Contextualization takes embeddings from a sequence of surrounding embeddings to re-render a new embedding that is centered at the original token embedding with variations taking into consideration of the context semantics (e.g., by transformer attention).
