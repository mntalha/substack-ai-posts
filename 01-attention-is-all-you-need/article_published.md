# The Paper That Started It All: Attention Is All You Need

*A deep dive into the transformer architecture — the foundation behind every LLM, VLM and AI system you use today*

---

It's June 2017. Eight researchers at Google publish a 15-page paper about machine translation. At the time, recurrent neural networks (RNNs) and LSTMs dominated sequence modeling. The idea that you could throw all of that away and replace it with a single mechanism sounded reckless.

That paper introduced the Transformer. Every large language model you interact with today — GPT-4, Claude, Gemini, LLaMA, DeepSeek — is a direct descendant of the architecture described in those 15 pages.

Let's break it down. I will be having one-by-one arhitecutre deep down to understand what is really happening inside of the transformer. The actual architecture, the math, the tensor shapes, and why every design choice was made.

---

## The Problem: RNNs Can't Scale

Before the Transformer, the state-of-the-art for sequence tasks (translation, summarization, language modeling) was the encoder-decoder RNN, typically using LSTMs or GRUs.

The fundamental issue: **RNNs are sequential.** To process the 100th token, you must first process tokens 1 through 99. Each hidden state h_t depends on h_{t-1}. This creates a chain of dependencies that:

1. **Kills parallelism.** You can't use all your GPU cores. Most of them sit idle while you wait for the previous step to finish.
2. **Makes long-range dependencies hard.** By the time you reach the 100th token, information from the 1st token has passed through 99 transformations — the gradient either vanishes or explodes.
3. **Caps training speed.** You can't scale to billions of tokens because the sequential bottleneck is hardware-limited, not data-limited.

The Transformer solves all three. Not with a clever trick on top of RNNs — by eliminating recurrence entirely and replacing it with attention.

[IMAGE: 04-why-parallel.html — RNN vs Transformer parallelism comparison]

---

## The Big Idea: Attention as the Whole Model

The core insight is deceptively simple. Instead of processing tokens one at a time and passing hidden states forward, let every token look at every other token simultaneously.

An attention function maps a query and a set of key-value pairs to an output. The output is a weighted sum of the values, where the weight for each value comes from comparing the query to the corresponding key.

The paper proposes **Scaled Dot-Product Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Here's what each piece does:

- **Q (Queries):** What each token is looking for
- **K (Keys):** What each token advertises about itself
- **V (Values):** What each token actually gives when attended to
- **QK^T:** A matrix of "how much does token i care about token j" scores
- **÷ √d_k:** Scaling factor to keep gradients stable (without this, dot products grow too large and softmax saturates)
- **softmax:** Turn raw scores into a probability distribution
- **× V:** Weight values by these probabilities

The result: every token in the sequence gets a new representation that is a weighted mix of all other tokens' values. And because this is a matrix multiplication, **it can be computed in one GPU operation for the entire sequence.**

[IMAGE: 02-attention-mechanism.html — Scaled Dot-Product Attention detail]

---

## Why Scale by √d_k?

This is a small detail that matters a lot. When d_k is large (say 64), the dot products Q·K^T produce values with variance proportional to d_k. Large values push softmax into regions where the gradient is extremely small — the model stops learning.

Dividing by √d_k brings the variance back to ~1, keeping the softmax in a region where gradients flow well. The paper tested this empirically and found it critical for larger dimensions.

This is one of those details interviewers love asking about.

---

## Multi-Head Attention: Eight Perspectives at Once

A single attention head computes one set of relationships. But language is complex — one token might be grammatically important to another, semantically similar to a third, and positionally relevant to a fourth.

The paper's solution: run attention multiple times in parallel with different learned projections. This is **Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

The base model uses **h = 8 heads**. Since d_model = 512, each head operates on d_k = d_v = 512/8 = 64 dimensions.

The key insight: **the total computation is the same as single-head attention with full dimensionality.** You're not doing 8x the work — you're splitting the same work into 8 parallel perspectives, then combining them.

Each head can specialize. The paper's attention visualizations show that some heads learn syntactic relationships (subject-verb agreement), while others learn semantic ones (anaphora resolution, long-distance dependencies).

The projection matrices:
- $W_i^Q \in \mathbb{R}^{512 \times 64}$ — projects queries for head i
- $W_i^K \in \mathbb{R}^{512 \times 64}$ — projects keys for head i
- $W_i^V \in \mathbb{R}^{512 \times 64}$ — projects values for head i
- $W^O \in \mathbb{R}^{512 \times 512}$ — projects concatenated heads back to model dimension

---

## The Full Architecture

Now let's put it all together. The Transformer is an **encoder-decoder** model:

- The **encoder** reads the entire input sequence and produces a rich representation
- The **decoder** generates the output sequence one token at a time, attending both to the encoder's representation and to its own previously generated tokens

[IMAGE: 01-architecture.html — Full Transformer architecture with tensor shapes]

### Encoder (N = 6 identical layers)

Each encoder layer has two sub-layers:

1. **Multi-Head Self-Attention** — every input token attends to every other input token
2. **Position-wise Feed-Forward Network** — a two-layer MLP applied independently to each token

Both sub-layers have a **residual connection** and **layer normalization:**

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Why LayerNorm and not BatchNorm? In NLP, sequence lengths vary across examples in a batch. BatchNorm normalizes across the batch dimension — problematic when sequences have different lengths. LayerNorm normalizes across the feature dimension for each individual example, making it sequence-length-agnostic.

### Decoder (N = 6 identical layers)

Each decoder layer has **three** sub-layers:

1. **Masked Multi-Head Self-Attention** — decoder tokens attend only to previous decoder tokens (preventing the model from "seeing the future")
2. **Multi-Head Cross-Attention** — decoder tokens attend to encoder output (this is where the decoder looks at the input)
3. **Position-wise Feed-Forward Network** — same as encoder

The masking in the first sub-layer is implemented by setting attention scores to -∞ for all positions that correspond to future tokens. After softmax, these become 0. This ensures the prediction for position i depends only on known outputs at positions less than i.

### Three Types of Attention

The Transformer uses attention in three distinct ways:

| Type | Q comes from | K, V come from | Purpose |
|------|-------------|----------------|---------|
| Encoder Self-Attention | Encoder layer | Same encoder layer | Each input token attends to all input tokens |
| Masked Decoder Self-Attention | Decoder layer | Same decoder layer | Each output token attends to previous output tokens only |
| Cross-Attention | Decoder layer | Encoder output | Output tokens attend to input representation |

[IMAGE: 08-layer-flow.html — Encoder vs Decoder layer-by-layer flow with all shapes annotated]

### How Data Flows Through Each Layer

Understanding the architecture diagram is one thing. Understanding what happens to a tensor as it passes through a single layer is another. Let's trace the encoder path with precision.

#### Before the First Layer: Building x

The encoder's input is a sequence of token IDs — shape **(1, S)**. Two things happen before any layer sees it:

1. **Token embedding.** Each token ID indexes into a learned matrix $E \in \mathbb{R}^{V \times 512}$. Output: **(1, S, 512)**.
2. **Positional encoding addition.** A matrix $PE \in \mathbb{R}^{S \times 512}$ (computed from sinusoids, not learned) is **added element-wise** to the embeddings.

$$x = \text{Embedding}(\text{tokens}) + PE$$

This is a pure element-wise sum — no concatenation, no extra linear layer. Each of the 512 dimensions in the embedding gets a small sinusoidal offset based on its position. The result $x$ has shape **(1, S, 512)** and is now the input to layer 1.

Important: positional encoding happens **once**, before the first encoder layer. It is not reapplied between layers.

#### Inside One Encoder Layer: The Q, K, V Projections

Here's where confusion often arises. The Q, K, V projections are **completely separate learned linear layers** — they have nothing to do with positional encoding. Positional encoding is already baked into $x$ from the step above. The projections are standard matrix multiplications:

$$Q = xW^Q, \quad K = xW^K, \quad V = xW^V$$

Where $W^Q, W^K, W^V \in \mathbb{R}^{512 \times 512}$ are **three independent, learned weight matrices** with no bias in modern implementations. Each one is a fully connected linear layer (no activation function, no ReLU). They are trained via backpropagation like every other weight in the model.

Think of it this way:
- **Positional encoding** teaches the model *where* each token sits. It's a fixed signal added once.
- **W^Q** teaches the model *what to look for* — it learns to extract a "question" from each token's representation.
- **W^K** teaches the model *what to advertise* — it learns to extract a "label" that other tokens can match against.
- **W^V** teaches the model *what to share* — it learns to extract the actual content that gets passed when attention fires.

All three projections read from the **same** input $x$ (which already contains positional information). The output shapes are:

| Projection | Weight matrix | Input | Output |
|---|---|---|---|
| Q = x W^Q | (512 × 512) | (1, S, 512) | (1, S, 512) |
| K = x W^K | (512 × 512) | (1, S, 512) | (1, S, 512) |
| V = x W^V | (512 × 512) | (1, S, 512) | (1, S, 512) |

These (1, S, 512) tensors are then reshaped into 8 heads: (1, 8, S, 64) — splitting the 512 into 8 chunks of 64. Each head computes its own attention pattern independently.

#### Where W^O Comes In

After all 8 heads finish, each produces a (1, S, 64) output. These are concatenated back into (1, S, 512). But this concatenated tensor has a problem: **the 512 dimensions are siloed by head.** Dimensions 0–63 came from head 0, dimensions 64–127 from head 1, and so on. No head has seen what any other head found.

This is where $W^O \in \mathbb{R}^{512 \times 512}$ comes in — the **output projection matrix**. It's the fourth learned linear layer inside multi-head attention:

$$\text{MultiHeadOutput} = \text{Concat}(\text{head}_1, \dots, \text{head}_8) \cdot W^O$$

$W^O$ is the **only matrix that enables cross-head communication**. Without it, multi-head attention would be 8 completely independent attention operations that never share information. $W^O$ blends features across heads, letting the model combine "head 3 found the subject" with "head 5 found the verb tense" into a single coherent representation.

After $W^O$, the attention sub-layer output has shape (1, S, 512).

#### The Residual Connection and LayerNorm

The attention output is **not** used directly. Instead:

$$h = \text{LayerNorm}(x + \text{MultiHeadAttn}(x))$$

The original input $x$ is added back (residual connection), then layer-normalized. This gives the intermediate representation $h$ with shape (1, S, 512). The residual means the attention sub-layer only needs to learn the **correction** — what information to add — rather than reconstructing the entire representation from scratch.

#### The FFN Sub-layer

$h$ then goes through the position-wise feed-forward network:

$$\text{output} = \text{LayerNorm}(h + \text{FFN}(h))$$

Same pattern: FFN output is added to $h$ (residual), then normalized. Final output shape: **(1, S, 512)** — identical to the input. This output becomes the input to the next encoder layer.

#### Full Encoder Layer Summary (in order)

```
x (1, S, 512)                          ← input (already has positional encoding)
  │
  ├─→ Q = x·W_Q  (1, S, 512)          ← three separate learned projections
  ├─→ K = x·W_K  (1, S, 512)
  ├─→ V = x·W_V  (1, S, 512)
  │
  ├─→ reshape to 8 heads: (1, 8, S, 64)
  ├─→ attention scores: (1, 8, S, S)
  ├─→ weighted V: (1, 8, S, 64)
  ├─→ concat → (1, S, 512)
  ├─→ × W_O → (1, S, 512)             ← cross-head mixing
  │
  └─→ residual: x + attn_output
       └─→ LayerNorm → h (1, S, 512)
            │
            ├─→ FFN: 512 → 2048 → 512
            └─→ residual: h + ffn_output
                 └─→ LayerNorm → output (1, S, 512)
```

So one encoder layer has **4 weight matrices for attention** ($W^Q, W^K, W^V, W^O$, each 512×512 = ~1.05M params total) plus **2 weight matrices for the FFN** ($W_1, W_2$ = ~2.1M params). Total: ~3.15M parameters per encoder layer, × 6 layers = ~18.9M for the full encoder stack.

#### The Decoder Layer

The **decoder** layer is heavier. It has the same self-attention → residual → LayerNorm pattern, but after that comes cross-attention. Here's where the two halves of the model meet: the decoder's intermediate representation becomes Q, while the encoder's final output provides K and V. The decoder is literally asking "given what I've generated so far, what parts of the input should I pay attention to?" After another residual + LayerNorm, the FFN processes the result. Three sub-layers instead of two — that's why the decoder has 50% more parameters per layer than the encoder.

#### The Residual Stream

The residual stream is the backbone. Think of it as a highway running through all layers. Each sub-layer is an on-ramp that adds information. Without residual connections, a 6-layer encoder would need to learn a single massive function from input to output. With residuals, each layer learns a small refinement. This is what makes deep networks trainable — and it's why modern models can stack 80+ layers without the signal degrading.

---

## The Feed-Forward Network

Each layer in both encoder and decoder contains a fully connected feed-forward network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

This is applied **position-wise** — independently and identically to each token. Think of it as a per-token MLP.

The dimensions: input/output is d_model = 512, but the inner layer expands to d_ff = 2048. This 4x expansion-then-compression is a pattern you'll see repeated in every transformer model since.

Why expand 4x? The inner layer acts as a memory bank. The expansion gives the network more capacity to store and process patterns before compressing back down. The ReLU activation makes this a selective computation — only about half the neurons activate for any given input.

Parameter count for one FFN:
- W_1: 512 × 2048 = 1,048,576
- b_1: 2048
- W_2: 2048 × 512 = 1,048,576
- b_2: 512
- **Total: ~2.1M per FFN sub-layer**

---

## Positional Encoding: Teaching Order Without Recurrence

The Transformer has no recurrence and no convolution. If you shuffle the input tokens, the self-attention output is the same (it's a set operation). But word order matters — "dog bites man" is not the same as "man bites dog."

The solution: add **positional encodings** to the input embeddings. The paper uses sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where pos is the position in the sequence and i is the dimension index.

Why sinusoids instead of learned embeddings?

1. **Generalization to longer sequences.** Sinusoids are defined for any position, even ones never seen during training. Learned embeddings are fixed to the training length.
2. **Relative position information.** For any fixed offset k, PE_{pos+k} can be expressed as a linear function of PE_{pos}. This means the model can learn to attend by relative position.
3. **The paper tested both.** Learned and sinusoidal embeddings gave nearly identical results (Table 3, row E). They chose sinusoidal for the extrapolation benefit.

[IMAGE: 03-positional-encoding.html — Positional encoding heatmap visualization]

Each dimension of the positional encoding corresponds to a sinusoid with a different frequency. Low dimensions have short wavelengths (changing rapidly with position), high dimensions have long wavelengths (changing slowly). Together they create a unique fingerprint for each position.

---

## The Complete Tensor Shape Flow

This is where everything clicks. Let's trace an input through the entire encoder, step by step. Assume batch size B = 1, source sequence length S = 10, target sequence length T = 8.

[IMAGE: 05-tensor-shapes.html — Complete shape flow cheat sheet]

### Encoder Path

| Step | Operation | Output Shape |
|------|-----------|-------------|
| 1 | Input token IDs | (1, 10) |
| 2 | Token Embedding | (1, 10, 512) |
| 3 | + Positional Encoding | (1, 10, 512) |
| 4 | Dropout (p=0.1) | (1, 10, 512) |
| 5 | Q, K, V projections | 3 × (1, 10, 512) |
| 6 | Reshape for 8 heads | 3 × (1, 8, 10, 64) |
| 7 | QK^T / √64 | (1, 8, 10, 10) |
| 8 | Softmax | (1, 8, 10, 10) |
| 9 | × V | (1, 8, 10, 64) |
| 10 | Concat heads | (1, 10, 512) |
| 11 | Output projection W^O | (1, 10, 512) |
| 12 | Residual + LayerNorm | (1, 10, 512) |
| 13 | FFN: W_1 + ReLU | (1, 10, 2048) |
| 14 | FFN: W_2 | (1, 10, 512) |
| 15 | Residual + LayerNorm | (1, 10, 512) |
| 16 | Repeat steps 5-15 × 5 more layers | (1, 10, 512) |

The shape (1, 8, 10, 10) at step 7 is the attention score matrix — it tells you the relationship strength between every pair of tokens. This is the matrix you see in attention visualizations.

### Decoder Path

| Step | Operation | Output Shape |
|------|-----------|-------------|
| 1 | Output token IDs (shifted right) | (1, 8) |
| 2 | Token Embedding + PE | (1, 8, 512) |
| 3 | Masked Self-Attention QK^T | (1, 8, 8, 8) — upper triangle masked |
| 4 | After self-attention | (1, 8, 512) |
| 5 | Cross-Attention Q from decoder | (1, 8, 512) |
| 6 | Cross-Attention K, V from encoder | (1, 10, 512) |
| 7 | Cross-Attention QK^T | (1, 8, 8, 10) — decoder attends to encoder |
| 8 | After cross-attention | (1, 8, 512) |
| 9 | FFN | (1, 8, 512) |
| 10 | Repeat × 5 more layers | (1, 8, 512) |
| 11 | Linear projection to vocab | (1, 8, vocab_size) |
| 12 | Softmax | (1, 8, vocab_size) |

Notice the cross-attention shape at step 7: **(1, 8, 8, 10)**. The first 8 is heads. The second 8 is decoder positions (T=8). The 10 is encoder positions (S=10). This is how the decoder "looks at" the input — each decoder token has a distribution over all encoder tokens.

---

## Concrete Example: Tracing "The cat sat down"

Abstract tensor shapes are useful. Actual numbers are better. Let's trace the sentence "The cat sat down" through the encoder, step by step, with concrete dimensions and real-ish values.

[IMAGE: 07-concrete-example.html — "The cat sat down" traced through the Transformer with actual numbers]

### Step 1: Tokenization → Token IDs

The tokenizer maps each word to an integer ID from the vocabulary (vocab_size ≈ 37,000 for the original model):

| Position | Token | Token ID |
|----------|-------|----------|
| 0 | The | 1996 |
| 1 | cat | 2368 |
| 2 | sat | 3940 |
| 3 | down | 2091 |

Input tensor: **(1, 4)** — batch size 1, sequence length 4.

### Step 2: Token Embedding

Each token ID indexes into a learned embedding matrix of shape (37000, 512). Token 1996 → a 512-dimensional vector. Token 2368 → a different 512-dimensional vector.

Output: **(1, 4, 512)** — four tokens, each represented as a 512-dimensional vector. At this point, "cat" and "sat" have completely independent representations — no token knows about any other.

### Step 3: Add Positional Encoding

Each position gets a unique sinusoidal fingerprint added element-wise. Position 0 gets PE(0), position 1 gets PE(1), etc. The magnitude of the positional encoding is much smaller than the embedding — it nudges the vector rather than dominating it.

After addition: "The" at position 0 has a slightly different representation than "The" at position 3 would. The model can now distinguish word identity from word position.

Output: **(1, 4, 512)** — same shape, but now position-aware.

### Step 4: Self-Attention (One Head, Simplified)

Let's focus on what happens inside one of the 8 attention heads. The 512-d vector for each token is projected down to 64 dimensions:

- **Q projection:** Each token's 512-d vector × W_Q (512×64) → 64-d query. "cat" generates a query that means "I'm looking for subjects and modifiers."
- **K projection:** Each token × W_K → 64-d key. "sat" generates a key that means "I'm a verb."
- **V projection:** Each token × W_V → 64-d value. This is the actual content that will be shared.

Now the attention scores — **Q × K^T gives a 4×4 matrix:**

|  | The | cat | sat | down |
|------|------|------|------|------|
| **The** | 2.1 | 0.8 | 0.3 | 0.1 |
| **cat** | 1.2 | 1.9 | 3.8 | 0.5 |
| **sat** | 0.4 | 4.1 | 1.7 | 2.9 |
| **down** | 0.2 | 0.6 | 3.5 | 1.8 |

Divide by √64 = 8, then softmax each row:

|  | The | cat | sat | down |
|------|------|------|------|------|
| **The** | **0.48** | 0.27 | 0.15 | 0.10 |
| **cat** | 0.08 | 0.10 | **0.72** | 0.10 |
| **sat** | 0.03 | **0.55** | 0.12 | **0.30** |
| **down** | 0.03 | 0.05 | **0.68** | 0.24 |

Read row by row: "cat" puts 72% of its attention on "sat" (its verb). "sat" splits attention between "cat" (55%, its subject) and "down" (30%, its modifier). "down" attends mostly to "sat" (68%). This is the model learning syntax from data — no rules were programmed.

Each row is a probability distribution. Multiply by the value vectors: each token's new representation is a weighted combination of all tokens' values, with these weights. "cat" becomes mostly "sat's value + a bit of everything else."

### Step 5: After 8 Heads, Concatenate

All 8 heads compute their own version of this, each attending to different patterns. Head 1 might learn syntactic structure. Head 3 might learn that "down" modifies "sat." Head 6 might learn positional proximity. The 8 outputs (each 64-d) are concatenated back to 512-d, then projected through W_O.

### Step 6: Residual + LayerNorm + FFN

The attention output is added to the original input (residual), normalized, passed through the 512→2048→512 FFN, added to the residual again, and normalized. One encoder layer complete.

After 6 layers of this, the representation of "cat" is no longer just "cat" — it encodes "cat as the subject of sat, which is modified by down, preceded by The." Every token's final representation is context-dependent, shaped by all other tokens through 6 rounds of attention.

---

## Why This Can Train in Parallel

This is the killer advantage. Let me contrast with an RNN:

**RNN:** To compute h_100, you need h_99. To compute h_99, you need h_98. All the way back to h_1. That's 100 serial steps. On 8 GPUs with thousands of cores, only one core is doing useful work at any given moment.

**Transformer:** Attention(Q, K, V) is a matrix multiplication. QK^T computes ALL pairwise relationships between ALL tokens in ONE operation. GPU hardware is specifically designed to execute matrix multiplications with massive parallelism — every core works simultaneously.

For a sequence of length S:
- **RNN:** O(S) sequential operations
- **Transformer:** O(1) sequential operations (one matrix multiply)

The tradeoff: the attention matrix is S × S, so memory is O(S²). For S = 10,000, that's 100 million entries per head. This is why long-context models needed FlashAttention (a later paper) to be practical. But for the sequence lengths used in 2017 translation tasks, the parallelism gain far outweighed the memory cost.

During training, the Transformer uses **teacher forcing** — the decoder receives the entire correct output sequence at once (shifted right by one position), masked so position i can't see positions > i. This means even the decoder is fully parallelizable during training. Only during inference does the decoder generate one token at a time.

---

## The KV Cache: Why Inference Is a Different Problem

Training and inference look nothing alike. During training, you process the entire sequence in one shot — both encoder and decoder see all tokens at once (masked appropriately). During inference, the decoder generates one token at a time. Token 1, then token 2, then token 3. Each new token requires running attention over all previous tokens.

Naively, generating the Nth token means recomputing Q, K, V for all N-1 previous tokens. That's O(N²) total work to generate N tokens. For a 4096-token response, that's catastrophic.

The solution: **the KV cache.** Cache the K and V projections from all previous tokens. When generating token N:

1. Compute Q, K, V only for the new token N → shapes (1, 1, d_model)
2. Append K_N and V_N to the cached K and V → cached shapes grow to (1, N, d_model)
3. Compute attention: Q_N × K_cached^T → (1, 1, N) attention scores
4. Multiply by V_cached → (1, 1, d_model) output

Instead of recomputing everything, you do O(N) work per token — just the dot product of the new query against all cached keys. Total work for N tokens: O(N²) without cache vs O(N²) with cache... wait, same total? Yes, but the constant factor is dramatically smaller because you skip the Q, K, V projections for all previous tokens. In practice, KV caching gives a **10-20× speedup** for generation.

The cost: **memory.** For each layer, you store K and V tensors of shape (batch, seq_len, d_model). For LLaMA 2 70B with 80 layers, d_model = 8192, and a 4096-token sequence:

$$\text{KV cache} = 2 \times 80 \times 4096 \times 8192 \times 2 \text{ bytes} = 10.7 \text{ GB per sequence}$$

That's 10.7 GB of GPU memory just for the cache of one sequence. This is why GQA (Grouped-Query Attention) matters so much — with 8 KV heads instead of 64, the cache shrinks to ~1.3 GB. It's also why batch sizes during inference are memory-bound, not compute-bound.

The KV cache is the single most important implementation detail for production LLM serving. Every inference framework (vLLM, TensorRT-LLM, TGI) is fundamentally a KV cache management system.

---

## Tokenization: From Text to Numbers

We've been treating token IDs as a given. But how does "The cat sat down" become [1996, 2368, 3940, 2091]? This is the tokenizer's job, and it has more design decisions than you'd expect.

The original Transformer uses **Byte-Pair Encoding (BPE)** on a shared source-target vocabulary of ~37,000 tokens. BPE works by:

1. Start with individual characters as tokens
2. Count all adjacent pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until you reach the desired vocabulary size

For example: "low", "lower", "lowest" might produce tokens ["low", "er", "est"] rather than three separate words. Common words like "the" become single tokens. Rare words get split into subwords — "transformative" might become ["transform", "ative"].

Why subword tokenization instead of word-level?
- **Word-level:** Vocabulary explodes (English alone has 170,000+ words). Rare words get mapped to \<UNK\>. No morphological sharing between "run" and "running."
- **Character-level:** Vocabulary is tiny (~256), but sequences become 4-5× longer. Attention is O(S²), so this quadruples the compute.
- **BPE:** A principled middle ground. Common words are single tokens (efficient). Rare words decompose into meaningful subwords (no \<UNK\>). Vocabulary size is tunable.

Modern models have evolved the tokenizer:
- **GPT-2/3/4:** Byte-level BPE, ~50,000 tokens. Operates on raw bytes, no unknown tokens possible.
- **LLaMA 2:** SentencePiece BPE, 32,000 tokens. Treats text as a sequence of Unicode characters.
- **LLaMA 3:** Tiktoken-style BPE, **128,256 tokens.** The 4× larger vocabulary means more words are single tokens — "transformer" is one token, not two. This makes sequences shorter, which reduces both attention compute and KV cache size. The tradeoff: the embedding matrix is 4× larger (128K × 4096 = 524M parameters vs 32K × 4096 = 131M).

The tokenizer is the first and last thing in the pipeline. It determines what the model "sees" and how efficiently it processes text. A bad tokenizer can silently destroy model performance on specific languages or domains.

---

## Training Setup

The base model hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| d_model | 512 |
| N (layers) | 6 |
| h (heads) | 8 |
| d_k = d_v | 64 |
| d_ff | 2048 |
| Dropout | 0.1 |
| Label Smoothing | 0.1 |
| Warmup Steps | 4,000 |
| Total Steps | 100,000 |
| Hardware | 8 × P100 GPUs |
| Training Time | 12 hours (base), 3.5 days (big) |

### The Learning Rate Schedule

The paper uses a warmup-then-decay schedule:

$$lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

This increases linearly for the first 4,000 steps, then decays proportionally to the inverse square root of the step number.

Why warmup? Adam optimizer's moment estimates (the running averages of gradients) are initialized at zero. In the first few steps, these estimates are wildly inaccurate. A low learning rate during warmup lets the moments stabilize before the model takes larger steps.

### Label Smoothing

Instead of training the model to output probability 1.0 for the correct token and 0.0 for everything else, they use ε_ls = 0.1 — the correct token gets probability 0.9, and 0.1 is distributed among all other tokens.

This actually **hurts perplexity** (the model becomes less confident) but **improves BLEU score** (translation quality). The intuition: a model that puts all its probability mass on one answer is brittle. Label smoothing forces the model to hedge slightly, which makes it more robust at generation time.

[IMAGE: 06-training-results.html — Training details and BLEU results]

---

## Results

| Model | EN-DE BLEU | EN-FR BLEU | Training Cost |
|-------|-----------|-----------|---------------|
| GNMT + RL Ensemble | 26.30 | 41.16 | High |
| ConvS2S Ensemble | 26.36 | 41.29 | High |
| **Transformer (base)** | **27.3** | **38.1** | **3.3 × 10^18 FLOPs** |
| **Transformer (big)** | **28.4** | **41.8** | **2.3 × 10^19 FLOPs** |

The base model **beats all previous ensembles** (combinations of multiple models) while being a single model trained in 12 hours on 8 GPUs. The big model sets a new state of the art by over 2 BLEU points on English-German.

The model also generalized to English constituency parsing without task-specific tuning — matching or exceeding dedicated parsers — showing the Transformer wasn't just good at translation.

---

## Parameter Count

For the base model (~65M parameters):

| Component | Count | Parameters |
|-----------|-------|-----------|
| Encoder Embedding | 1 | vocab_size × 512 ≈ 19M |
| Decoder Embedding | 1 | Shared with encoder |
| Encoder Self-Attention (per layer) | 6 | 4 × 512 × 512 = 1.05M × 6 |
| Encoder FFN (per layer) | 6 | 2 × 512 × 2048 = 2.1M × 6 |
| Decoder Self-Attention (per layer) | 6 | 1.05M × 6 |
| Decoder Cross-Attention (per layer) | 6 | 1.05M × 6 |
| Decoder FFN (per layer) | 6 | 2.1M × 6 |
| Output Linear | 1 | Shared with embedding |

Weight sharing between the encoder embedding, decoder embedding, and pre-softmax linear transformation is one of the elegant tricks — it reduces parameters significantly and provides a useful inductive bias.

---

## The Fork: BERT, GPT, and the Three Transformer Variants

Within a year of this paper, the Transformer split into three architectures that would define NLP for the next decade. Understanding this fork is essential context for everything that came after.

### Encoder-Only: BERT (2018)

Google's BERT took the Transformer encoder, threw away the decoder entirely, and trained it with a new objective: **Masked Language Modeling (MLM).** Randomly mask 15% of tokens, predict them from context. Because there's no decoder, every token attends to every other token in both directions — BERT is **bidirectional.**

This makes BERT exceptional at understanding tasks: classification, named entity recognition, question answering, sentiment analysis. But it can't generate text — there's no autoregressive mechanism. You can't ask BERT to write a paragraph.

BERT's key insight: for most NLP tasks, you don't need to generate text — you need to understand it. A 340M parameter BERT model dominated 11 NLP benchmarks simultaneously.

### Decoder-Only: GPT (2018)

OpenAI's GPT took the Transformer decoder, threw away the encoder and cross-attention, and trained it with standard **language modeling:** predict the next token given all previous tokens. The causal mask ensures each token only sees earlier positions — making it **unidirectional.**

This makes GPT a natural text generator. Give it a prompt, and it continues. The same architecture works for question answering (the answer follows the question), translation (source followed by target), summarization (document followed by summary). Everything is framed as "predict the next token."

GPT's key insight: a single autoregressive model, trained on enough text, learns to do many tasks without task-specific architectures. GPT-2 (1.5B parameters) showed emergent abilities. GPT-3 (175B) showed in-context learning. GPT-4 showed that this approach scales far beyond what anyone expected.

### Encoder-Decoder: T5 (2019)

Google's T5 kept the full Transformer architecture — encoder and decoder with cross-attention — but framed every NLP task as a text-to-text problem. Classification: "classify: The movie was great" → "positive." Translation: "translate English to French: Hello" → "Bonjour." Summarization: input the article, output the summary.

T5 showed that the original encoder-decoder architecture is a strong general-purpose choice. But it requires 2× the parameters of a decoder-only model for the same total size.

### Why Decoder-Only Won

By 2023, the field had converged on decoder-only:

1. **Simplicity.** One architecture for all tasks. No encoder, no cross-attention, no task-specific heads.
2. **Scaling efficiency.** All parameters contribute to a single thing: predicting the next token. In an encoder-decoder, the encoder parameters are "wasted" on tasks that don't need input understanding.
3. **Emergent capabilities.** In-context learning, chain-of-thought reasoning, and instruction following all emerged naturally from autoregressive training at scale. These properties don't emerge from MLM.
4. **Engineering simplicity.** One KV cache, one inference path, one training loop. Easier to optimize, easier to serve, easier to scale.

Every frontier model today — GPT-4, Claude, Gemini, LLaMA, Mistral, DeepSeek — is decoder-only. BERT-style models survive in narrow niches (embedding models, rerankers). The original Transformer's encoder-decoder lives on in specialized models (Whisper for speech, some translation systems). But the decoder-only variant is the architecture of the era.

---

## Decoding: How the Model Actually Generates Text

The Transformer outputs a probability distribution over the entire vocabulary at each position. During training, you compare this distribution to the correct answer (cross-entropy loss). During inference, you need to actually pick a token. How you pick matters enormously.

### Greedy Decoding

Take the highest-probability token at each step. Fast, deterministic, and often mediocre. The problem: the locally best choice at each step doesn't guarantee the globally best sequence. "The" might be the most likely first word, but "Once" could lead to a much better overall sentence.

### Beam Search

Maintain the top-k partial sequences (beams) at each step. The paper used beam search with beam size 4 for translation. At each step, expand each beam by all possible next tokens, score them, and keep the top 4. This explores more of the search space without the exponential cost of exhaustive search.

Beam search excels at structured tasks (translation, transcription) where there's a clearly "correct" output. It produces fluent, high-quality text — but it's repetitive and boring for open-ended generation.

### Temperature Sampling

Instead of taking the argmax, sample from the distribution. Temperature τ controls diversity:

$$p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}$$

- **τ < 1:** Sharper distribution, more deterministic. The model's top choices become even more likely.
- **τ = 1:** Raw model probabilities.
- **τ > 1:** Flatter distribution, more random. Even unlikely tokens get a chance.

### Top-k and Top-p (Nucleus) Sampling

**Top-k:** Only consider the top k tokens. Zero out everything else, renormalize. k=50 means the model picks from its 50 best guesses. Simple and effective, but the right k depends on context — sometimes 50 is too many (the model is very confident), sometimes too few (ambiguous context).

**Top-p (nucleus sampling):** Sort tokens by probability and keep the smallest set whose cumulative probability exceeds p. If p=0.9, you keep enough tokens to cover 90% of the probability mass. When the model is confident, this might be just 2-3 tokens. When it's uncertain, it might be 100+. This adapts naturally to context.

In practice, production LLM APIs typically combine temperature + top-p. Most ChatGPT responses use temperature ~0.7-1.0 with top-p ~0.95. Code generation uses lower temperature (more deterministic). Creative writing uses higher (more diverse).

### Why This Matters

The same model produces dramatically different outputs depending on decoding strategy. A Transformer that seems "dumb" with greedy decoding might seem "brilliant" with nucleus sampling at the right temperature. When you're evaluating a model, you're really evaluating the model × decoding strategy combination.

---

## From the Original Transformer to Modern LLMs: What Changed

The architecture in this paper was an encoder-decoder model for translation. Every major LLM today — GPT-4, LLaMA, Mistral, DeepSeek — uses a **decoder-only** variant with several key modifications. Here's what LLaMA (Meta's open model family) changed and why.

[IMAGE: 09-llama-architecture.html — Original Transformer vs LLaMA side-by-side comparison with all changes annotated]

### 1. Decoder-Only (drop the encoder + cross-attention)

The original Transformer had an encoder that reads the input bidirectionally and a decoder that generates output while cross-attending to the encoder. Modern LLMs drop both the encoder and cross-attention entirely. The prompt and completion are one continuous token stream — the causal mask is the only structure needed. Early layers implicitly learn to "encode" (understand), later layers "decode" (predict).

### 2. Pre-RMSNorm (instead of Post-LayerNorm)

The paper uses `LayerNorm(x + Sublayer(x))` — normalization **after** the residual addition. LLaMA flips this to `x + Sublayer(RMSNorm(x))` — normalization **before** the sublayer. This keeps the residual stream clean and makes training stable at scale without careful warmup schedules. RMSNorm is also simpler than LayerNorm (no mean subtraction, no bias), making it ~10-15% faster.

### 3. RoPE — Rotary Position Embeddings

The paper adds sinusoidal position encodings to the input embeddings **once**, before layer 1. LLaMA uses RoPE, which **rotates Q and K vectors at every attention layer** by position-dependent angles. The key benefit: the dot product between rotated Q and K encodes **relative** position (m−n), not absolute position. This means a model trained on 4K context can be extended to 128K+ with RoPE frequency scaling — the original sinusoidal encoding couldn't extrapolate.

### 4. Grouped-Query Attention (GQA)

The paper uses full Multi-Head Attention where every head has its own Q, K, and V. LLaMA 2 70B and all LLaMA 3 models use GQA: 32 Q heads but only **8 K/V heads** shared across groups of 4 Q heads. This reduces the KV-cache by 4× during inference (gigabytes saved for a 70B model with 80 layers) with negligible quality loss.

### 5. SwiGLU (instead of ReLU FFN)

The paper's FFN: `max(0, xW₁ + b₁) · W₂ + b₂` — two weight matrices with ReLU, which kills all negative values to exactly 0 ("dead neurons"). LLaMA uses SwiGLU: `(xW₁ ⊙ Swish(xW_gate)) · W₂` — three weight matrices with a smooth Swish gate. No dead neurons, better gradient flow, and a learnable gating mechanism. The FFN expansion ratio is adjusted from 4× to ~2.67× d_model to keep total parameter count similar.

### 6. No Bias Terms

LLaMA removes all bias vectors from every linear layer, attention projection, and normalization layer. At scale, biases add negligible expressivity but complicate quantization and tensor parallelism.

### One LLaMA Decoder Block (pseudocode)

```python
# Input: x of shape (B, T, 4096) — LLaMA 2 7B

# Sub-layer 1: Attention
h = RMSNorm(x)                         # pre-norm
Q = h @ W_Q → reshape (B, 32, T, 128)  # 32 query heads
K = h @ W_K → reshape (B, 8, T, 128)   # 8 KV heads (GQA)
V = h @ W_V → reshape (B, 8, T, 128)   # 8 KV heads
Q, K = apply_rope(Q, K, positions)      # rotate by position
K, V = repeat_kv(K, V, n_rep=4)        # expand 8→32 for GQA
attn = softmax(Q @ K.T / √128) @ V     # (B, 32, T, T) × (B, 32, T, 128)
attn = reshape(attn) @ W_O             # → (B, T, 4096)
x = x + attn                           # residual

# Sub-layer 2: SwiGLU FFN
h = RMSNorm(x)                         # pre-norm
gate = swish(h @ W_gate)               # (B, T, 11008)
up = h @ W_up                          # (B, T, 11008)
ffn = (gate ⊙ up) @ W_down             # (B, T, 4096)
x = x + ffn                            # residual

# Output: x of shape (B, T, 4096)
```

This block repeats 32 times (7B) or 80 times (70B). The final output goes through one more RMSNorm, then a linear projection to the vocabulary.

### LLaMA Model Configurations

| Parameter | LLaMA 2 7B | LLaMA 2 70B | LLaMA 3 8B | LLaMA 3 70B | LLaMA 3.1 405B |
|-----------|-----------|------------|-----------|------------|---------------|
| Layers | 32 | 80 | 32 | 80 | 126 |
| d_model | 4096 | 8192 | 4096 | 8192 | 16384 |
| Q Heads | 32 | 64 | 32 | 64 | 128 |
| KV Heads | 32 (MHA) | 8 (GQA) | 8 (GQA) | 8 (GQA) | 8 (GQA) |
| d_head | 128 | 128 | 128 | 128 | 128 |
| d_ff | 11008 | 28672 | 14336 | 28672 | 53248 |
| Vocab | 32000 | 32000 | 128256 | 128256 | 128256 |
| Context | 4096 | 4096 | 8192 | 8192 | 131072 |

Every major open model (Mistral, Qwen, DeepSeek, Gemma) uses essentially the same recipe: decoder-only + Pre-RMSNorm + RoPE + GQA + SwiGLU + no biases. The original Transformer gave us the blueprint — these six changes are the modern refinements that made it scale to hundreds of billions of parameters.

---

## What Made This Paper Revolutionary

Looking back, it wasn't just the attention mechanism. The paper combined several design decisions that together created something far greater than the sum of its parts:

1. **Pure attention, no recurrence.** The radical simplification that enabled everything else.
2. **Multi-head attention.** Multiple parallel perspectives without extra compute.
3. **Residual connections + LayerNorm.** Made it possible to stack 6+ layers without training collapse.
4. **Positional encoding.** Elegant solution to the permutation-invariance problem.
5. **The scaling factor.** A small mathematical detail that made deep attention networks trainable.
6. **Weight sharing.** Between embeddings and the output layer — parameter efficiency.
7. **Teacher forcing + masking.** Made decoder training fully parallel.

The architecture was so good that the entire field stopped trying to improve RNNs within two years. BERT (2018) used the encoder. GPT (2018) used the decoder. Every major model since has been a Transformer variant.

---

## Interview Questions You Should Be Ready For

**Q: Why divide by √d_k in attention?**
A: Dot products of high-dimensional vectors have variance proportional to d_k. Large values push softmax into saturated regions with near-zero gradients. Scaling by √d_k normalizes the variance to ~1, keeping gradients healthy.

**Q: Why multi-head instead of one big attention?**
A: Multi-head lets the model attend to information from different representation subspaces simultaneously. A single head averages across all patterns. Multiple heads specialize — some learn syntax, some learn semantics, some learn positional patterns. Total compute stays the same because each head operates on d_model/h dimensions.

**Q: How does masking work in the decoder?**
A: A matrix of -∞ values is added to the attention scores before softmax for all positions where i < j (above the diagonal). After softmax, e^(-∞) = 0, so those positions contribute zero weight. This ensures token i can only attend to tokens 0 through i.

**Q: Why LayerNorm instead of BatchNorm?**
A: Sequences in a batch have different lengths. BatchNorm normalizes across the batch dimension, which gives inconsistent statistics when sequences are padded to different lengths. LayerNorm normalizes across the feature dimension for each token independently — no batch-level dependency.

**Q: What's the computational complexity of self-attention?**
A: O(S² · d) per layer, where S is sequence length and d is d_model. The S² comes from the QK^T matrix (every token attending to every other). This is why very long sequences are expensive — the quadratic scaling is the main limitation of the Transformer.

**Q: How is the Transformer parallelizable but the decoder is autoregressive?**
A: During **training**, teacher forcing means the entire target sequence is available — masking prevents cheating, but computation is fully parallel. During **inference**, the decoder must generate one token at a time (autoregressive). The parallelism advantage is primarily a training-time benefit.

**Q: What would you change about this architecture today?**
A: Pre-RMSNorm instead of Post-LayerNorm (more stable training at scale). RoPE instead of sinusoidal positional encodings (relative positions, better length extrapolation). SwiGLU instead of ReLU in the FFN (no dead neurons, learnable gating). GQA instead of full MHA (4× less KV-cache). Drop the encoder entirely (decoder-only). Remove all bias terms. These are exactly the changes made in LLaMA, Mistral, and other modern models — see the section above for the full breakdown.

**Q: Why share weights between embeddings and the output linear layer?**
A: The embedding maps tokens → vectors. The output linear layer maps vectors → token probabilities. These are inverse operations on the same vocabulary space. Sharing weights means the model learns a single, consistent token representation used for both input and output, while reducing total parameters by ~19M.

**Q: What is the KV cache and why does it matter?**
A: During autoregressive generation, each new token needs to attend to all previous tokens. The KV cache stores the key and value projections from all prior tokens so you don't recompute them. Without it, generating N tokens costs O(N²) in projection compute. With it, you only project the new token and reuse cached K, V. The tradeoff is memory — for a 70B model, the KV cache can exceed 10 GB per sequence.

**Q: Why did decoder-only architectures win over encoder-decoder?**
A: Simplicity and scaling. Decoder-only models use all parameters for one objective (next-token prediction). No encoder means no parameter overhead for input-understanding components. In-context learning, chain-of-thought, and instruction-following all emerged naturally from autoregressive training at scale. Plus, one architecture handles every task — you just change the prompt, not the model structure.

**Q: Explain the difference between top-k and top-p (nucleus) sampling.**
A: Top-k keeps a fixed number of highest-probability tokens and zeros out the rest. Top-p keeps the smallest set of tokens whose cumulative probability exceeds p. Top-p adapts to the model's confidence — when the model is sure, it considers fewer tokens; when uncertain, more. This makes top-p more robust across different contexts.

**Q: What problem does FlashAttention solve?**
A: Standard attention materializes the full S × S attention matrix in GPU HBM (high-bandwidth memory), which is O(S²) in memory and bandwidth-bound. FlashAttention computes attention in tiles, keeping intermediate results in fast SRAM and never materializing the full matrix. It doesn't change the math — the output is identical — but it reduces memory from O(S²) to O(S) and is 2-4× faster by avoiding memory bottlenecks. This is what made long-context (32K+) models practical.

**Q: Why does LLaMA use RMSNorm instead of LayerNorm?**
A: LayerNorm normalizes by subtracting the mean and dividing by standard deviation, then applies learnable scale and bias. RMSNorm skips the mean subtraction and has no bias — it only divides by the root mean square. Empirically, the re-centering (mean subtraction) doesn't help at scale, but it costs compute. RMSNorm is ~10-15% faster with equivalent or better training stability.

**Q: What's the difference between absolute and relative positional encoding?**
A: Absolute (sinusoidal or learned) assigns a fixed vector to each position and adds it once at the input. Relative (RoPE, ALiBi) encodes the distance between tokens rather than their absolute positions. The key advantage of relative: a model trained on 4K context can extrapolate to longer sequences because it learned "tokens 5 apart relate like this" rather than "token at position 4000 looks like this." RoPE achieves this by rotating Q and K vectors at every layer, making the attention score a function of (position_m − position_n).

---

## The Legacy

The eight authors of this paper went on to shape the field:

- **Ashish Vaswani** and **Niki Parmar** founded Essential AI
- **Noam Shazeer** co-founded Character.AI, returned to Google for Gemini
- **Aidan Gomez** co-founded Cohere
- **Illia Polosukhin** co-founded NEAR Protocol
- **Llion Jones** founded Sakana AI
- **Jakob Uszkoreit** co-founded Inceptive

The architecture they proposed has been applied to text, images, audio, video, protein folding, weather prediction, and robotics. The "plan to apply them to other tasks" mentioned in their conclusion turned out to be the understatement of the decade.

---

*Next up: Chinchilla — the paper that told everyone they were training their models wrong, and rewrote the rules of scaling.*

---

*Talha Kılıç — SWE @ Microsoft. I read research papers and share what genuinely excites me. Follow for more deep dives into the papers behind modern AI.*
