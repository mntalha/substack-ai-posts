# Transformers from First Principles: The Complete (That's the Hope) Mathematical Breakdown

*A deep dive into how transformers actually work, with the equations, concrete numbers, and engineering reasoning behind every component.*

---

Understanding transformers at the level where you can derive the equations, trace the tensor shapes, and compute concrete examples by hand. That's what separates working knowledge from deep expertise. This article builds that depth, one concept at a time.

Each section follows the same structure: the precise mathematical formulation, a worked numerical example, and the engineering reasoning behind the design choice. The goal is a single reference that covers every core concept with enough rigor to reconstruct any part of the transformer from scratch.

---

## Part 1: The Hyperparameter Glossary: What Every Symbol Means

Before anything else, you need to know what every symbol in a transformer config actually controls. These aren't abstract: each one directly determines tensor shapes, parameter counts, and compute costs.

### Q: What is d_model?

**d_model** is the **embedding dimension**, the width of every representation in the model. Every token, at every layer, is represented as a vector of exactly d_model numbers.

When you see a token like "cat," it becomes a vector of d_model floats: `[0.12, -0.34, 0.56, ..., 0.78]`. This vector is the model's internal representation of that token. Every matrix multiplication in the model either starts from d_model or returns to d_model.

| Model | d_model |
|-------|---------|
| Transformer (base) | 512 |
| GPT-2 | 768 |
| LLaMA 2 7B | 4096 |
| LLaMA 2 70B | 8192 |
| LLaMA 3.1 405B | 16384 |

Larger d_model = richer token representations = more capacity to encode meaning. But compute scales as O(d_model²) in the attention and FFN layers.

### Q: What is N (number of layers)?

**N** is how many transformer blocks are stacked sequentially. Each block contains one attention sub-layer and one FFN sub-layer (plus normalization and residuals). The original paper uses N = 6 for both encoder and decoder.

Think of it as depth of reasoning. Layer 1 might capture basic syntactic patterns. Layer 16 might capture that "bank" means a financial institution, not a riverbank, based on context from 200 tokens ago. Layer 32 might compose those features into complex reasoning.

| Model | N (layers) |
|-------|-----------|
| Transformer (base) | 6 |
| GPT-2 | 12 |
| LLaMA 2 7B | 32 |
| LLaMA 2 70B | 80 |
| LLaMA 3.1 405B | 126 |

More layers = more sequential computation = deeper reasoning, but also slower inference (each token must pass through all N layers).

### Q: Heads vs layers: what's the difference?

This is a common point of confusion. Both "add more heads" and "add more layers" make the model bigger, but they do fundamentally different things.

**Heads = parallel width within one layer.** All 8 heads receive the same input (the output of the previous layer) and process it simultaneously. Each head learns a different relationship pattern on that same input. Their outputs get concatenated and mixed by W_O. Adding more heads means more relationship types detected *at the same time*, at the same depth.

**Layers = sequential depth across steps.** Each layer takes the *output of the previous layer* as input. Layer 1 builds features from raw embeddings. Layer 10 builds on the features from layer 9, which built on layer 8, and so on. Adding more layers means more rounds of refinement and composition.

```
Heads (parallel, same input):
  Input x → [Head 1] → concat → W_O → output
            [Head 2] ↗
            [Head 3] ↗
            ...all run on the SAME x

Layers (sequential, chained):
  x → [Layer 1] → x₁ → [Layer 2] → x₂ → ... → [Layer 32] → x₃₂
      each layer's output feeds the next
```

**Concrete example** with "The cat that I saw yesterday sat down":

- **Layer 1, Head 3** might learn: "that" → "cat" (relative pronoun links to nearest noun)
- **Layer 1, Head 5** might learn: "The" → "cat" (determiner links to its noun)
- **Layer 16** (after 15 layers of refinement) might learn: "cat" is the subject of "sat," not "saw," because layers 1-15 already resolved the relative clause structure

Layer 16 can only do this *because* earlier layers already did their work. A single layer with 100 heads could detect 100 patterns simultaneously, but it couldn't compose them the way stacked layers can. Conversely, 100 layers with 1 head each could compose deeply, but each layer would only see one relationship pattern at a time.

In practice, you need both. Modern models use 32-128 heads (parallel breadth) and 32-126 layers (sequential depth).

### Q: What is h (number of attention heads) and why not just one?

**h** is the number of parallel attention computations within each attention sub-layer. Each head gets its own Q, K, V projection matrices and computes its own attention pattern independently.

The critical relationship: **d_k = d_model / h**. If d_model = 512 and h = 8, each head operates on 64-dimensional vectors. The total compute is the same as a single head with 512 dimensions: you're splitting the same work into 8 parallel views. Eight heads on 64 dims = same FLOPs as one head on 512 dims, but much richer representation.

Why not just one head? A single head computes one weighted average: it can only capture one relationship pattern per layer. Multiple heads specialize. Research shows different heads learn to attend to different things:

**Concrete example** from a trained 8-head model processing "The cat that I saw yesterday sat down":

| Head | What it attends to | Example pattern |
|:----:|-------------------|-----------------|
| 1 | Subject → verb | "cat" → "sat" (long-range dependency, skipping relative clause) |
| 2 | Adjective/modifier → noun | "yesterday" → "saw" |
| 3 | Relative pronoun → antecedent | "that" → "cat" |
| 4 | Positional proximity | Each token → its immediate neighbors |
| 5 | Determiner → noun | "The" → "cat" |
| 6–8 | Other patterns | Discovered by training, often hard to interpret |

### Q: What are d_k and d_v?

**d_k** = dimension of each Query and Key vector per head. **d_v** = dimension of each Value vector per head.

In the original paper: d_k = d_v = d_model / h = 512 / 8 = 64.

Why are they separate symbols? Because they *could* be different. The Q·K^T dot product requires d_k to match between Q and K, but V could have a different dimension. In practice, every major model sets d_k = d_v = d_model / h, so they're always equal.

The attention score computation requires:
- Q shape: (seq_len, d_k)
- K shape: (seq_len, d_k), must match Q for dot product
- V shape: (seq_len, d_v), can differ, determines output width
- QK^T shape: (seq_len, seq_len), the attention matrix
- Output shape: (seq_len, d_v), weighted sum of V

### Q: Why divide by √d_k in attention? The full variance proof.

This is one of those details that separates surface-level understanding from real depth. Here's the full derivation.

Assume entries of Q and K are independent random variables with mean 0 and variance 1. The dot product of one query and one key is:

$$q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i$$

Each term $q_i \cdot k_i$ has: $E[q_i k_i] = 0$ and $\text{Var}(q_i k_i) = 1$

Since the terms are independent, the variance of the sum is:

$$\text{Var}(q \cdot k) = d_k \times 1 = d_k$$

For d_k = 64: the dot products have standard deviation $\sqrt{64} = 8$. Values routinely reach ±16 or more. Let's see what happens to softmax:

**Without scaling (d_k = 64):**
```
Dot products:     [12.5, -8.3, 15.1, 3.2]
softmax:          [0.07, 0.00, 0.93, 0.00]   ← nearly one-hot, gradient ≈ 0
```

**With scaling (÷ √64 = ÷ 8):**
```
Scaled products:  [1.56, -1.04, 1.89, 0.40]
softmax:          [0.26, 0.02, 0.36, 0.08]   ← smooth distribution, healthy gradients
```

Without scaling, softmax saturates: it pushes nearly all weight onto one token and the gradients vanish. The model stops learning. Dividing by √d_k normalizes the variance back to ~1, keeping the softmax in a gradient-friendly regime.

### Q: What is d_ff?

**d_ff** is the **inner dimension of the feed-forward network**. Each FFN expands from d_model to d_ff, applies an activation, then compresses back to d_model:

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

Where $W_1 \in \mathbb{R}^{d\_model \times d\_ff}$ and $W_2 \in \mathbb{R}^{d\_ff \times d\_model}$.

The original paper uses d_ff = 4 × d_model = 2048. This 4× expansion ratio is remarkably consistent across architectures:

| Model | d_model | d_ff | Ratio |
|-------|---------|------|-------|
| Transformer (base) | 512 | 2048 | 4.0× |
| GPT-2 | 768 | 3072 | 4.0× |
| LLaMA 2 7B | 4096 | 11008 | 2.69× (SwiGLU has 3 matrices) |

LLaMA's ratio looks smaller, but SwiGLU uses **3 weight matrices** instead of 2, so the total FFN parameter count is similar to a 4× ReLU FFN.

Why expand at all? The FFN is where the model stores factual knowledge. Research (Geva et al., 2021) shows FFN neurons act as key-value memories: the first layer's rows are "keys" that match input patterns, and the second layer's columns are "values" that produce the associated output. More d_ff = more memory slots = more knowledge capacity.

### Q: What is vocab_size and what does it actually mean?

**vocab_size** (V) is the total number of distinct tokens the model knows. It's the size of the model's "dictionary." Every token the model can read or generate must be one of these V tokens.

For GPT-2: V = 50,257. This means there are exactly 50,257 possible tokens. The model literally cannot output anything that isn't one of these 50,257 options. Each one maps to a specific string: token 464 = " the", token 198 = "\n", token 15496 = "Hello".

The vocab_size directly determines:
1. **Embedding matrix shape**: (V, d_model), each token gets its own d_model-dimensional vector
2. **Output layer shape**: (d_model, V), the final linear layer that produces logits
3. **Output logits shape**: at each position, the model outputs V numbers, one score per possible token

### Q: What is the relationship between vocab_size and parameter count?

The embedding and output layers are directly proportional to V:

$$\text{Embedding params} = V \times d\_model$$

For LLaMA 3 with V = 128,256 and d_model = 4096:

$$128{,}256 \times 4{,}096 = 525{,}336{,}576 \approx 525M \text{ parameters}$$

That's **525 million parameters** just to map tokens ↔ vectors. For LLaMA 2 with V = 32,000:

$$32{,}000 \times 4{,}096 = 131{,}072{,}000 \approx 131M \text{ parameters}$$

The 4× larger vocabulary costs 394M extra parameters, roughly **5% of a 7B model**. The tradeoff: larger vocabulary → shorter sequences (more words are single tokens) → less compute per sample. LLaMA 3's 128K vocabulary makes sequences ~15% shorter than LLaMA 2's 32K vocabulary, which saves attention compute (O(S²)).

**Why share weights between embeddings and the output layer?** The embedding maps tokens → vectors. The output linear layer maps vectors → token probabilities. These are inverse operations on the same vocabulary space. Sharing weights means the model learns a single, consistent token representation used for both input and output.

**Concrete savings:** For the original Transformer: V × d_model = 37,000 × 512 ≈ 19M parameters saved. For LLaMA 2: 32,000 × 4,096 ≈ 131M parameters saved. That's an entire small model's worth of parameters.

### Q: Where do all the parameters come from? Full parameter count breakdown.

Let's count every parameter in a decoder-only model (LLaMA 2 7B style):

**Per layer (× 32 layers):**

| Component | Weight Shape | Parameters |
|-----------|-------------|-----------|
| Q projection | (4096, 4096) | 16,777,216 |
| K projection | (4096, 4096) | 16,777,216 |
| V projection | (4096, 4096) | 16,777,216 |
| O projection | (4096, 4096) | 16,777,216 |
| FFN W_gate | (4096, 11008) | 45,088,768 |
| FFN W_up | (4096, 11008) | 45,088,768 |
| FFN W_down | (11008, 4096) | 45,088,768 |
| RMSNorm (×2) | (4096) × 2 | 8,192 |
| **Layer total** | | **201,684,160** |

**Global:**

| Component | Parameters |
|-----------|-----------|
| Token embedding | 32,000 × 4096 = 131,072,000 |
| Final RMSNorm | 4,096 |
| Output head (shared w/ embedding) | 0 (shared) |

**Total:** 32 × 201,684,160 + 131,072,000 + 4,096 ≈ **6.6B parameters**

The FFN accounts for **~67%** of per-layer parameters (3 matrices of ~45M each = 135M out of 201M). Attention accounts for ~33% (4 matrices of ~17M each = 67M). This is a universal ratio across transformer models.

### Q: What's the computational complexity of self-attention?

O(S² · d) per layer, where S is sequence length and d is d_model. The S² comes from the QK^T matrix: every token attending to every other token produces an (S × S) matrix.

**Concrete numbers for LLaMA 2 7B processing a 4096-token sequence:**

| Operation | Shape | FLOPs |
|-----------|-------|------:|
| Q = x @ W_Q | (4096, 4096) × (4096, 4096) | 34.4B |
| K = x @ W_K | same | 34.4B |
| V = x @ W_V | same | 34.4B |
| QK^T (all heads) | (4096, 4096) × (4096, 4096)^T | 34.4B |
| attn × V | (4096, 4096) × (4096, 4096) | 34.4B |
| × W_O | (4096, 4096) × (4096, 4096) | 34.4B |
| **Attention total** | | **~206B FLOPs** |
| FFN (3 matrices) | 3 × (4096, 11008) | ~270B |
| **Layer total** | | **~476B FLOPs** |
| **All 32 layers** | | **~15.2T FLOPs** |

For a single forward pass on 4096 tokens: ~15 trillion floating-point operations. At A100's 312 TFLOPS (FP16), that's ~49ms, close to real benchmarks (~45ms).

The quadratic S² scaling is the main limitation: doubling sequence length from 4K → 8K quadruples the attention FLOPs from 34.4B → 137.6B per operation.

---

## Part 2: Tokenization: The Zero-Foot View

### Q: What is Byte-Pair Encoding (BPE)? Walk through the algorithm.

BPE is a subword tokenization algorithm that builds a vocabulary by iteratively merging the most frequent adjacent pairs of tokens. Here's the exact algorithm:

**Step 0: Start with characters.** Split every word in the corpus into characters, plus a special end-of-word marker:

```
Corpus: "low" (5 times), "lower" (2 times), "newest" (6 times), "widest" (3 times)

Initial vocabulary: {l, o, w, e, r, n, s, t, i, d, </w>}

Token sequences:
"low"    → [l, o, w, </w>]       × 5
"lower"  → [l, o, w, e, r, </w>] × 2
"newest" → [n, e, w, e, s, t, </w>] × 6
"widest" → [w, i, d, e, s, t, </w>] × 3
```

**Step 1: Count all adjacent pairs:**

| Pair | Count |
|------|-------|
| (e, s) | 6 + 3 = 9 |
| (l, o) | 5 + 2 = 7 |
| (o, w) | 5 + 2 = 7 |
| (s, t) | 6 + 3 = 9 |
| (n, e) | 6 |
| (e, w) | 6 |
| (w, </w>) | 5 |
| ... | ... |

**Step 2: Merge most frequent pair.** Tie between (e,s) and (s,t) at 9. Pick (e,s) → new token `es`.

```
"low"    → [l, o, w, </w>]       × 5
"lower"  → [l, o, w, e, r, </w>] × 2
"newest" → [n, e, w, es, t, </w>] × 6
"widest" → [w, i, d, es, t, </w>] × 3
```

**Step 3: Recount and merge again.** Now (es, t) appears 6 + 3 = 9 times. Merge → `est`.

```
"newest" → [n, e, w, est, </w>] × 6
"widest" → [w, i, d, est, </w>] × 3
```

**Step 4: Continue.** (l, o) → `lo`, (lo, w) → `low`, (est, </w>) → `est</w>`, etc.

**Keep merging until you reach the target vocabulary size** (e.g., 32,000 or 50,257 tokens).

The result: common words like "the" become single tokens. Rare words like "transformative" decompose into known subwords: `["transform", "ative"]`. No word is ever `<UNK>`. BPE can always fall back to individual characters.

**Concrete example with a real tokenizer (LLaMA 3, tiktoken, V=128,256):**

```
"Hello, how are you?"           → [9906, 11, 1268, 527, 499, 30]     (6 tokens)
"transformative"                → [4806, 1413]                        (2 tokens: "transform" + "ative")
"Pneumonoultramicroscopicsilico → [47, 7907, 263, 8927, ...]          (many tokens, rare word)
  volcanoconiosis"
"the"                           → [1820]                               (1 token, very common)
"  the"                         → [220, 1820]                          (2 tokens, leading space is separate)
```

Notice: the tokenizer treats leading spaces, punctuation, and capitalization as part of the token. `"Hello"` and `" hello"` are different tokens.

### Q: What are the special tokens? Are they needed in training, inference, or both?

Special tokens are tokens with specific roles that exist outside the natural text vocabulary:

| Token | Symbol | Purpose | Used in |
|-------|--------|---------|---------|
| Beginning of Sequence | `<BOS>` or `<s>` | Signals the start of a new sequence | Training + Inference |
| End of Sequence | `<EOS>` or `</s>` | Signals the model should stop generating | Training + Inference |
| Padding | `<PAD>` | Fills shorter sequences in a batch to equal length | Training only |
| Unknown | `<UNK>` | Fallback for tokens not in vocabulary | Rarely used with BPE |

**In training:**
- `<BOS>` is prepended: `[<BOS>, The, cat, sat, on, the, mat, <EOS>]`
- The model learns to predict `<EOS>` when a sequence should end
- `<PAD>` is used when batching sequences of different lengths: padded positions are **masked out** of the loss function so they don't affect gradients

**Concrete batching example:**

```
Sequence A: [<BOS>, The, cat, sat, <EOS>]           (5 tokens)
Sequence B: [<BOS>, Hello, <EOS>, <PAD>, <PAD>]     (3 real + 2 padding)

Loss mask:  [  1,    1,     1,     0,      0  ]      (ignore PAD positions)
```

**In inference:**
- `<BOS>` is prepended to the prompt
- The model generates tokens until it outputs `<EOS>` or hits a max-length limit
- No `<PAD>` is needed (you process one sequence at a time, or use dynamic batching)

**Chat models** add more special tokens:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is attention?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

These chat template tokens are included in **both training and inference**: the model learns to generate text only after seeing `assistant<|end_header_id|>`, and to stop at `<|eot_id|>`.

### Q: What is the most well-known tokenization used today?

**Byte-level BPE** (used by GPT-2, GPT-3, GPT-4, LLaMA 3) is the dominant approach. It operates on raw UTF-8 bytes instead of Unicode characters:

1. Base vocabulary = 256 byte values (0x00 through 0xFF)
2. BPE merges are learned on top of these bytes
3. **Any text in any language** can be tokenized, no `<UNK>` ever

Why bytes instead of Unicode characters? Unicode has 150,000+ code points. If a character wasn't in the training data (a rare script, a new emoji), it becomes `<UNK>` and the model can't process it at all. Bytes solve this: every piece of text in every language is ultimately stored as a sequence of these 256 byte values in UTF-8 encoding. So the base vocabulary is small (256), fixed, and covers everything:

```
"Hello"  → bytes: [72, 101, 108, 108, 111]           (ASCII, 1 byte per char)
"猫"     → bytes: [231, 140, 171]                     (Chinese, 3 bytes per char)
"🔥"     → bytes: [240, 159, 148, 165]                (emoji, 4 bytes)
```

The tradeoff: rare languages and scripts get split into more tokens (3-4 bytes per character instead of 1 token), making sequences longer. But nothing is ever unrepresentable.

GPT-2's vocab size of 50,257 breaks down as: 256 base byte tokens + 50,000 learned BPE merges + 1 special `<|endoftext|>` token.

**SentencePiece BPE** (used by LLaMA 2, Mistral) is similar but operates on Unicode code points and treats the input as a raw character stream without pre-tokenization rules.

**tiktoken** (OpenAI) is a fast implementation of byte-level BPE with regex-based pre-tokenization that splits text into chunks before BPE, preventing merges across word boundaries.

| Tokenizer | Vocab Size | Base Unit | Models |
|-----------|-----------|-----------|--------|
| GPT-2 BPE | 50,257 | Bytes | GPT-2, GPT-3 |
| SentencePiece | 32,000 | Unicode chars | LLaMA 2, Mistral |
| tiktoken | 128,256 | Bytes | LLaMA 3, GPT-4 |

---

## Part 3: Training: Parallelism, Learning Rate, and Loss

### Q: How do we make training parallel? Is it only training, or also inference?

**Training parallelism** works at three levels:

**1. Sequence-level parallelism (within one GPU):** The causal mask lets the model process all T tokens in a sequence simultaneously. For a 1024-token sequence, the model makes 1024 next-token predictions in **one forward pass**, not 1024 sequential steps.

**Concrete example with 4 tokens.** Input sequence: `[<BOS>, The, cat, sat]`

Without parallelism (RNN-style), you'd need 4 sequential steps:
```
Step 1: Feed [<BOS>]                    → predict "The"   (can't start step 2 until done)
Step 2: Feed [<BOS>, The]              → predict "cat"   (needs hidden state from step 1)
Step 3: Feed [<BOS>, The, cat]         → predict "sat"   (needs hidden state from step 2)
Step 4: Feed [<BOS>, The, cat, sat]    → predict next    (needs hidden state from step 3)
```

Each step waits for the previous one. 4 tokens = 4 serial GPU operations.

With the transformer's causal mask, all 4 predictions happen in **one matrix multiply**:
```
Position 0 (<BOS>): sees [<BOS>]                    → predicts "The"
Position 1 (The):   sees [<BOS>, The]                → predicts "cat"
Position 2 (cat):   sees [<BOS>, The, cat]           → predicts "sat"
Position 3 (sat):   sees [<BOS>, The, cat, sat]      → predicts next token
```

The mask prevents cheating (position 2 can't see "sat" at position 3), so each prediction is mathematically identical to the sequential version. The QK^T matrix computes all 4 rows simultaneously on the GPU:

```
         <BOS>  The   cat   sat
<BOS>  [  ✓     ✗     ✗     ✗  ]   ← all 4 rows compute
The    [  ✓     ✓     ✗     ✗  ]   ← in parallel on
cat    [  ✓     ✓     ✓     ✗  ]   ← the GPU
sat    [  ✓     ✓     ✓     ✓  ]   ← simultaneously
```

Scale this up: for a 1024-token sequence, the RNN needs 1024 serial steps. The transformer does it in 1 step, making 1024 predictions simultaneously. That's why transformers train orders of magnitude faster on the same hardware.

```
Training:   [t₁, t₂, t₃, ..., t₁₀₂₄] → all predictions in ONE matrix multiply
RNN:        t₁ → t₂ → t₃ → ... → t₁₀₂₄  (1024 sequential steps)
```

The key: the causal mask (lower-triangular matrix of 1s) prevents token i from attending to tokens j > i, so it's mathematically equivalent to processing tokens one at a time, but computed in parallel.

**2. Data parallelism (across GPUs):** Copy the model to N GPUs, split each batch into N chunks, compute gradients independently, average them (all-reduce), update weights identically.

**Concrete example: training LLaMA 2 7B on 4 GPUs:**
```
Global batch size: 32 sequences × 4096 tokens = 131,072 tokens

GPU 0: processes sequences  1–8  → computes gradients
GPU 1: processes sequences  9–16 → computes gradients
GPU 2: processes sequences 17–24 → computes gradients
GPU 3: processes sequences 25–32 → computes gradients

All-reduce: average all 4 gradient tensors (6.6B floats each)
Update: every GPU applies identical weight update
```

**3. Model parallelism (for large models):** Tensor parallelism splits weight matrices across GPUs. Pipeline parallelism puts different layers on different GPUs.

**Concrete example: LLaMA 2 70B on 8 GPUs with tensor + pipeline parallelism:**
```
Pipeline: GPU 0-3 have layers 1-40, GPU 4-7 have layers 41-80
Tensor:   Within each pipeline stage, 4 GPUs split attention heads
          GPU 0 has Q heads 1-16, GPU 1 has Q heads 17-32, etc.
```

For large models (70B+ parameters), you typically use **all three**: data + tensor + pipeline parallelism.

**Inference parallelism is fundamentally different:**

- **Prefill phase** (processing the prompt): IS parallel. The entire prompt is processed in one forward pass, exactly like training. All tokens attend to each other (with causal mask) simultaneously. This is compute-bound: faster GPUs = faster prefill.
- **Decode phase** (generating new tokens): IS NOT parallel. Each new token depends on all previous tokens. You must generate token 1, then token 2, then token 3. This is inherently sequential and memory-bandwidth-bound, limited by how fast you can read model weights from GPU memory.

```
Inference prefill:  [prompt tokens]  → parallel, compute-bound, fast per token
Inference decode:   [one new token]  → sequential, memory-bound, slow per token
```

This is why "time to first token" (TTFT) and "tokens per second" (TPS) are measured separately: they're bottlenecked by completely different things.

The only ways to speed up decode:
- **KV cache**: avoid recomputing K, V for past tokens (see Part 5)
- **Speculative decoding**: a small draft model generates N candidate tokens, the large model verifies them in one parallel pass. If the draft guessed right, you get N tokens for the cost of 1 large-model pass (2–3× speedup).
- **Batch inference**: process multiple sequences' decode steps together (parallelism across sequences, not within)
- **Quantization**: fewer bytes per weight = faster memory reads

### Q: What should the learning rate be? Does a small difference matter?

Yes, learning rate is arguably the most sensitive hyperparameter. Too high and training diverges (loss goes to infinity). Too low and training crawls or gets stuck in bad local minima.

The original Transformer uses a **warmup-then-decay** schedule:

$$lr(step) = d_{model}^{-0.5} \cdot \min(step^{-0.5}, \; step \cdot warmup^{-1.5})$$

For d_model = 512 and warmup_steps = 4000, let's compute actual values:

| Step | Phase | Actual lr |
|------|-------|-----------|
| 1 | Warmup | 5.5 × 10⁻⁹ |
| 100 | Warmup | 1.1 × 10⁻⁵ |
| 1,000 | Warmup | 1.1 × 10⁻⁴ |
| 4,000 | Peak | **7.0 × 10⁻⁴** |
| 10,000 | Decay | 4.4 × 10⁻⁴ |
| 100,000 | Decay | 1.4 × 10⁻⁴ |

Phase 1 (steps 1–4000): **Linear warmup.** lr increases linearly from ~0 to peak (~7×10⁻⁴).
Phase 2 (steps 4000+): **Inverse square root decay.** lr decreases proportionally to 1/√step.

**Why warmup?** The Adam optimizer maintains running averages of gradients (first moment m) and squared gradients (second moment v). These are initialized to zero. In the first few steps:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(starts near 0, inaccurate)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(starts near 0, inaccurate)}$$

The bias correction helps: $\hat{m}_t = m_t / (1 - \beta_1^t)$. But early estimates are still noisy. A small learning rate during warmup prevents the model from making large, misguided updates based on unreliable moment estimates.

**Modern practice (LLaMA, GPT-3):** Peak lr ≈ 3×10⁻⁴ for 7B models, 1.5×10⁻⁴ for 70B models. Cosine decay schedule instead of inverse square root. The larger the model, the smaller the peak lr: big models amplify gradient signals through more layers, so they need gentler updates.

**Does a small difference matter?** Yes. Changing lr from 3×10⁻⁴ to 6×10⁻⁴ (2×) can cause training instability in large models. Changing from 3×10⁻⁴ to 1×10⁻⁴ (0.33×) can waste millions of dollars in compute by converging too slowly. The Chinchilla paper showed that hyperparameter tuning (especially lr) is critical for compute-optimal training.

### Q: What is label smoothing, why do we use it, and why does it hurt perplexity?

Before label smoothing, we need to understand **cross-entropy loss**, the objective function that transformers are trained to minimize.

### Q: What is cross-entropy loss? Walk through the computation.

Cross-entropy loss measures how far the model's predicted probability distribution is from the correct answer. It's computed at every token position during training.

**Concrete example.** Vocab = {The, cat, sat, on, mat} (5 tokens). The correct next token is "sat" (index 2).

**Step 1: Model outputs logits (raw scores from the final linear layer)**

```
logits = [1.2, 0.8, 3.5, 1.0, 0.3]
          The  cat  sat   on  mat
```

**Step 2: Softmax converts logits to probabilities**

$$P(token_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```
e^1.2 = 3.32,  e^0.8 = 2.23,  e^3.5 = 33.12,  e^1.0 = 2.72,  e^0.3 = 1.35
Sum = 42.74

P = [0.078, 0.052, 0.775, 0.064, 0.032]
     The    cat    sat     on    mat
```

The model puts 77.5% probability on "sat". Good prediction, but not perfect.

**Step 3: Cross-entropy loss**

The target is a one-hot vector: `[0, 0, 1, 0, 0]` (1 at "sat", 0 elsewhere).

$$\text{Loss} = -\sum_{i} y_i \cdot \log P(token_i)$$

Since y is one-hot, only the correct token's term survives:

$$\text{Loss} = -1 \cdot \log(0.775) = 0.255$$

The loss is just **negative log of the probability assigned to the correct token**. Higher confidence in the right answer = lower loss.

| Model's P(correct) | Loss = -log(P) | Meaning |
|:---:|:---:|---|
| 0.01 | 4.605 | Very wrong, barely considers the right answer |
| 0.10 | 2.303 | Poor, right answer is low probability |
| 0.50 | 0.693 | Mediocre, coin flip |
| 0.775 | 0.255 | Good (our example) |
| 0.95 | 0.051 | Very confident and correct |
| 0.999 | 0.001 | Near-perfect (but logits are getting dangerously large) |

**Step 4: How gradients push the model to improve**

The gradient of cross-entropy with respect to the logits has a simple form:

$$\frac{\partial \text{Loss}}{\partial z_i} = P(token_i) - y_i$$

For our example:

```
Gradients = P - y = [0.078, 0.052, 0.775, 0.064, 0.032] - [0, 0, 1, 0, 0]
                   = [+0.078, +0.052, -0.225, +0.064, +0.032]
                      push ↓   push ↓   push ↑   push ↓   push ↓
```

Positive gradient for wrong tokens (push their logits down), negative for the correct token (push its logit up). The magnitude tells you how much to push.

**Step 5: Loss over an entire sequence**

During training, cross-entropy is computed at every position and averaged:

```
Sequence: [<BOS>, The, cat, sat, on]
Targets:  [The,   cat, sat, on, mat]

Position 0: P(The)=0.12   → loss = -log(0.12) = 2.12
Position 1: P(cat)=0.08   → loss = -log(0.08) = 2.53
Position 2: P(sat)=0.775  → loss = -log(0.775) = 0.26
Position 3: P(on)=0.25    → loss = -log(0.25) = 1.39
Position 4: P(mat)=0.15   → loss = -log(0.15) = 1.90

Average loss = (2.12 + 2.53 + 0.26 + 1.39 + 1.90) / 5 = 1.64
```

All 5 positions contribute gradients simultaneously (that's the training parallelism from the causal mask). This average loss is what gets backpropagated through all layers to update every weight in the model.

### Q: Now, what is label smoothing and why does it change this?

**Standard cross-entropy** trains the model to output probability 1.0 for the correct token and 0.0 for all others. The target distribution is a one-hot vector:

$$y_{hard} = [0, 0, 0, 1, 0, 0, ..., 0] \quad \text{(1 at the correct token, 0 elsewhere)}$$

**Label smoothing** (ε = 0.1) replaces this with:

$$y_{smooth} = (1 - \varepsilon) \cdot y_{hard} + \varepsilon / V$$

**Concrete example** with V = 37,000 and the correct token at index 3:

$$y_{smooth}[3] = 0.9 + 0.1/37000 = 0.900003$$
$$y_{smooth}[\text{all others}] = 0.1/37000 = 0.0000027$$

**What the cross-entropy loss looks like with and without smoothing:**

Without smoothing: the model is incentivized to push the correct logit to infinity:
```
Step 1000:  P(correct) = 0.85  → loss = -log(0.85)  = 0.163  → keep pushing
Step 5000:  P(correct) = 0.95  → loss = -log(0.95)  = 0.051  → keep pushing
Step 10000: P(correct) = 0.999 → loss = -log(0.999) = 0.001  → logits exploding
```

With smoothing: the model converges to a stable target:
```
Step 1000:  P(correct) = 0.85  → loss against 0.9 target = moderate
Step 5000:  P(correct) = 0.89  → loss against 0.9 target = small
Step 10000: P(correct) = 0.90  → loss ≈ 0 → stable, no logit explosion
```

**Why it hurts perplexity:** Perplexity measures how well the model's predicted distribution matches the **hard** target (one-hot). A model trained with label smoothing is explicitly trained to NOT put 100% probability on the correct token, so its perplexity is worse by design. But **BLEU score** (translation quality) improves because the model's distribution is more calibrated: it assigns reasonable probability to synonyms and alternative phrasings.

$$\text{Perplexity: worse (by design)} \quad \text{BLEU: better (the actual goal)}$$

### Q: What is perplexity? How do you calculate it?

Perplexity measures how "surprised" the model is by the test data. Mathematically:

$$PPL = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(x_t | x_{<t})\right)$$

Or equivalently: $PPL = \exp(H)$ where H is the cross-entropy loss.

**Why $P(x_t | x_{<t})$ and not $P(x_t | \text{full sequence})$?** Because that's how the model actually works. The causal mask means each position can only see previous tokens:

```
Predicting "cat":   model sees [The]              → P(cat | The)
Predicting "sat":   model sees [The, cat]         → P(sat | The, cat)
Predicting "down":  model sees [The, cat, sat]    → P(down | The, cat, sat)
```

Each prediction is conditioned on everything *before* it, nothing after. During inference, future tokens don't exist yet so the model can't use them. Perplexity measures the model the way it's actually used. If we gave the model the full sequence for each prediction, we'd get artificially low perplexity that doesn't reflect real generation ability.

**Concrete example.** A model processes the sequence "The cat sat down" (4 tokens). At each position, the model assigns a probability to the correct next token:

| Position | Correct token | Model's P(correct) | -log P |
|----------|--------------|--------------------:|-------:|
| 1 | cat | 0.12 | 2.12 |
| 2 | sat | 0.08 | 2.53 |
| 3 | down | 0.25 | 1.39 |
| 4 | \<EOS\> | 0.15 | 1.90 |

$$H = \frac{1}{4}(2.12 + 2.53 + 1.39 + 1.90) = 1.985$$

$$PPL = e^{1.985} = 7.28$$

**Interpretation:** A perplexity of 7.28 means the model is, on average, as uncertain as if it were choosing uniformly among ~7 options at each step. Lower = better.

| PPL | Interpretation |
|-----|---------------|
| 1.0 | Perfect, model is 100% sure of every token |
| 10 | Choosing from ~10 equally likely options per token |
| 100 | Very uncertain, basically guessing |
| V (50,257) | Uniform random, worst possible |

**Important:** Perplexity is only comparable between models with the **same tokenizer**. A model with a 128K vocabulary that tokenizes "transformers" as one token will have different perplexity than a model with a 32K vocabulary that tokenizes it as ["transform", "ers"], even if they're equally good.

### Q: What is BLEU score? How do you calculate it?

BLEU (Bilingual Evaluation Understudy) measures how much a machine-generated translation overlaps with a human reference translation. It's a precision-based metric computed on n-grams.

**Step-by-step computation:**

**Bad candidate (shows why clipping matters):**

**Reference:** "The cat is on the mat"
**Candidate:** "The the the the the the"

For unigrams: "the" appears 6 times in candidate, but only 2 times in reference → clipped count = 2

$$p_1 = \frac{\text{clipped matches}}{\text{total candidate unigrams}} = \frac{2}{6} = 0.333$$

Without clipping, this degenerate candidate would score 6/6 = 1.0. Clipping prevents gaming the metric.

**Good candidate (full walkthrough):**

**Reference:** "The cat is on the mat"
**Candidate:** "The cat sat on the mat"

| n-gram | Candidate n-grams | Match in reference? | Clipped matches / Total |
|--------|------------------|--------------------:|:-----------------------:|
| 1-gram | "The"(2), "cat"(1), "sat"(1), "on"(1), "mat"(1) | "sat" doesn't match "is" | p₁ = 5/6 |
| 2-gram | "The cat", "cat sat", "sat on", "on the", "the mat" | "The cat"✓, "cat sat"✗, "sat on"✗, "on the"✓, "the mat"✓ | p₂ = 3/5 |
| 3-gram | "The cat sat", "cat sat on", "sat on the", "on the mat" | "on the mat"✓ | p₃ = 1/4 |
| 4-gram | "The cat sat on", "cat sat on the", "sat on the mat" | none match | p₄ = 0/3 |

**Geometric mean of n-gram precisions (n=1..4):**

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{4} \frac{1}{4} \log p_n\right)$$

Since p₄ = 0, BLEU = 0. Real systems use **smoothing** (add-1 or add-ε to zero counts) to avoid this.

**Brevity Penalty (BP):**

$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$$

Where c = candidate length, r = reference length. This penalizes translations that are too short.

**Concrete BP example:** c = 4 words, r = 6 words: $BP = e^{1 - 6/4} = e^{-0.5} = 0.607$. The short translation gets a 39% penalty.

| BLEU Score | Quality | Example |
|-----------|---------|---------|
| < 10 | Almost useless | Word salad with occasional correct words |
| 10–20 | Gist understandable | Correct words, wrong order and grammar |
| 20–30 | Good for many uses | Minor errors, meaning preserved |
| 30–40 | High quality | Near-fluent, occasional awkward phrasing |
| 40–50 | Very high quality | Hard to distinguish from human |

The Transformer base achieved 27.3 BLEU on EN-DE and 38.1 on EN-FR, beating all previous ensemble models.

**Perplexity vs BLEU: when do you use which?**

| Metric | Type | Measures | Needs generation? |
|--------|------|----------|:-:|
| **Perplexity** | Intrinsic | How well the model predicts next tokens | No |
| **BLEU** | Extrinsic | How similar generated text is to a reference | Yes |

Key insight from the original paper: label smoothing made perplexity worse but BLEU better. Perplexity rewards extreme confidence. BLEU rewards good generation. They measure different things.

---

## Part 4: Decoding Strategies: The Full Math

Here are logits from the last position of a language model. Vocabulary size = 8 (simplified). Let's walk through each decoding strategy with exact numbers.

**Raw logits** (output of the final linear layer):

| Token | the | cat | sat | on | mat | dog | ran | ate |
|-------|:---:|:---:|:---:|:--:|:---:|:---:|:---:|:---:|
| Logit | 5.0 | 3.2 | 2.8 | 2.0 | 1.5 | 1.0 | 0.5 | -1.0 |

### Q: Walk through Greedy Decoding.

**Step 1:** Apply softmax to convert logits to probabilities:

$$P(token_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

| Token | Logit z | e^z | P(token) |
|-------|:-------:|----:|:--------:|
| the | 5.0 | 148.41 | **0.563** |
| cat | 3.2 | 24.53 | 0.093 |
| sat | 2.8 | 16.44 | 0.062 |
| on | 2.0 | 7.39 | 0.028 |
| mat | 1.5 | 4.48 | 0.017 |
| dog | 1.0 | 2.72 | 0.010 |
| ran | 0.5 | 1.65 | 0.006 |
| ate | -1.0 | 0.37 | 0.001 |
| **Sum** | | **263.48** | **1.000** |

**Step 2:** Pick the token with the highest probability.

$$\hat{y} = \arg\max_i P(token_i) = \text{"the"} \quad (P = 0.563)$$

**That's it.** Greedy decoding always picks the most likely token. It's deterministic: same input always gives the same output.

**Problem:** Greedy is locally optimal but not globally optimal. Picking "the" now might lead to a worse overall sentence than picking "cat" now. It also produces repetitive, boring text for open-ended generation.

### Q: Walk through Beam Search (beam width = 2).

Beam search maintains the top-k (here k=2) partial sequences at each step.

**Step 1 (first token):** Pick top-2 tokens from the distribution above.

```
Beam 1: ["the"],  score: log(0.563) = -0.575
Beam 2: ["cat"],  score: log(0.093) = -2.375
```

**Step 2 (second token):** For each beam, run the model again to get next-token distributions. Suppose:

After "the": P(cat)=0.4, P(dog)=0.3, P(mat)=0.2, ...
After "cat": P(sat)=0.5, P(ran)=0.2, P(ate)=0.15, ...

Expand all beams × all tokens, score each (sum of log probs), keep top-2:

```
"the cat",  score: -0.575 + log(0.4) = -0.575 + (-0.916) = -1.491  ✓ keep
"the dog",  score: -0.575 + log(0.3) = -0.575 + (-1.204) = -1.779  ✓ keep
"the mat",  score: -0.575 + log(0.2) = -0.575 + (-1.609) = -2.184  ✗ pruned
"cat sat",  score: -2.375 + log(0.5) = -2.375 + (-0.693) = -3.068  ✗ pruned
```

**Continue until \<EOS\>.** The beam with the highest total score wins. Length normalization prevents favoring short sequences:

$$\text{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P(y_t | y_{<t})$$

Where α ∈ [0.6, 1.0] is a length penalty. Without it (α=0): "Good." scores higher than "Good morning, how are you?" because shorter sequences accumulate less negative log-prob.

### Q: Walk through Temperature Sampling. What happens at T=0.5 vs T=2.0?

Temperature scales the logits **before** softmax:

$$P(token_i) = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}$$

Using the same logits [5.0, 3.2, 2.8, 2.0, 1.5, 1.0, 0.5, -1.0]:

**T = 0.5 (sharper, more deterministic):**

| Token | Logit | z/0.5 | e^(z/0.5) | P(token) |
|-------|:-----:|:-----:|----------:|:--------:|
| the | 5.0 | 10.0 | 22,026.5 | **0.958** |
| cat | 3.2 | 6.4 | 601.8 | 0.026 |
| sat | 2.8 | 5.6 | 270.4 | 0.012 |
| on | 2.0 | 4.0 | 54.6 | 0.002 |
| mat | 1.5 | 3.0 | 20.1 | 0.001 |
| dog | 1.0 | 2.0 | 7.4 | 0.000 |
| ran | 0.5 | 1.0 | 2.7 | 0.000 |
| ate | -1.0 | -2.0 | 0.1 | 0.000 |

At T=0.5, "the" jumps from 56.3% to **95.8%**. The distribution is nearly deterministic. Good for factual, focused responses.

**T = 2.0 (flatter, more random):**

| Token | Logit | z/2.0 | e^(z/2.0) | P(token) |
|-------|:-----:|:-----:|----------:|:--------:|
| the | 5.0 | 2.5 | 12.18 | **0.291** |
| cat | 3.2 | 1.6 | 4.95 | 0.118 |
| sat | 2.8 | 1.4 | 4.06 | 0.097 |
| on | 2.0 | 1.0 | 2.72 | 0.065 |
| mat | 1.5 | 0.75 | 2.12 | 0.051 |
| dog | 1.0 | 0.5 | 1.65 | 0.039 |
| ran | 0.5 | 0.25 | 1.28 | 0.031 |
| ate | -1.0 | -0.5 | 0.61 | 0.015 |

At T=2.0, "the" drops from 56.3% to **29.1%**. Even "ate" (originally 0.1%) gets 1.5%. More creative but risks incoherence.

**Summary of temperature's effect on this distribution:**

| T | P("the") | P("cat") | P("ate") | Behavior |
|---|:--------:|:--------:|:--------:|----------|
| 0.1 | ~1.000 | ~0.000 | ~0.000 | Greedy (argmax) |
| 0.5 | 0.958 | 0.026 | 0.000 | Very focused |
| 1.0 | 0.563 | 0.093 | 0.001 | Model's raw distribution |
| 2.0 | 0.291 | 0.118 | 0.015 | Creative, some randomness |
| 10.0 | 0.146 | 0.133 | 0.108 | Nearly uniform random |

**T → 0:** Becomes greedy (argmax). **T → ∞:** Becomes uniform random.

### Q: Walk through Top-k Sampling (k=3).

**Step 1:** Sort tokens by probability (using T=1.0 softmax from above):

| Rank | Token | P(token) |
|:----:|-------|:--------:|
| 1 | the | 0.563 |
| 2 | cat | 0.093 |
| 3 | sat | 0.062 |
| 4 | on | 0.028 |
| 5–8 | mat, dog, ran, ate | 0.034 total |

**Step 2:** Keep only the top k=3 tokens. Zero out the rest.

**Step 3:** Renormalize so the remaining probabilities sum to 1:

$$\text{Sum} = 0.563 + 0.093 + 0.062 = 0.718$$

| Token | Original P | After top-k | Renormalized P |
|-------|:----------:|:-----------:|:--------------:|
| the | 0.563 | 0.563 | **0.784** |
| cat | 0.093 | 0.093 | **0.130** |
| sat | 0.062 | 0.062 | **0.086** |
| on–ate | 0.034 | 0 | 0 |

**Step 4:** Sample from the renormalized distribution. "the" gets picked ~78% of the time, "cat" ~13%, "sat" ~9%.

**Problem with top-k:** k is fixed regardless of the model's confidence:
- Model is 99% confident → k=50 wastes time on 49 irrelevant tokens
- Model is genuinely uncertain → k=3 cuts off many valid options

### Q: Walk through Top-p (Nucleus) Sampling (p=0.9).

This is the most important one to understand deeply. Top-p **adapts** the number of candidates to the model's confidence.

**Step 1:** Sort tokens by probability and compute cumulative:

| Rank | Token | P(token) | Cumulative P |
|:----:|-------|:--------:|:------------:|
| 1 | the | 0.5634 | 0.5634 |
| 2 | cat | 0.0931 | 0.6565 |
| 3 | sat | 0.0624 | 0.7189 |
| 4 | on | 0.0281 | 0.7470 |
| 5 | mat | 0.0170 | 0.7640 |
| 6 | dog | 0.0103 | 0.7743 |
| 7 | ran | 0.0063 | 0.7806 |
| 8 | ate | 0.0014 | 0.7820 |

In our toy vocabulary (V=8), even all tokens sum to only 0.782. In a real model (V=50,257), the remaining 50,249 tokens cover the remaining 0.218 of probability mass. For p=0.9, you'd include our 8 tokens plus the next ~100 highest-probability tokens until cumulative reaches 0.9.

**Step 2:** Find the smallest set of tokens whose cumulative probability ≥ p. Zero out everything else, renormalize, sample.

**Why top-p is better than top-k: two scenarios with the same model:**

*Scenario A: "The capital of France is" → Model is confident:*
```
P("Paris") = 0.95, P("the") = 0.02, P("a") = 0.01, ...
Top-k (k=50): considers 50 tokens despite 95% confidence → 49 wasted
Top-p (p=0.9): considers only 1 token ("Paris") → efficient, correct
```

*Scenario B: "I had a great time at the" → Model is uncertain:*
```
P("park") = 0.08, P("beach") = 0.07, P("party") = 0.07, P("concert") = 0.06, ...
Top-k (k=3): only considers park/beach/party → misses concert, museum, restaurant
Top-p (p=0.9): considers ~25 tokens → captures the full range of valid completions
```

Top-p adapts naturally to context. Top-k doesn't.

### Q: Why does the decoding strategy matter so much?

The same model with the same weights produces dramatically different text:

```
Prompt: "The meaning of life is"

Greedy:     "The meaning of life is to be happy. The meaning of life is to be happy. The meaning of..."
            (repetitive, degenerate, greedy gets stuck in loops)

Beam (k=5): "The meaning of life is to find purpose and fulfillment in one's work and relationships."
            (fluent but generic, beam search optimizes probability, not creativity)

T=0.7 + top-p=0.9:
            "The meaning of life is something each person must discover through their own experiences,
             failures, and moments of unexpected joy."
            (creative, coherent, the sweet spot for most applications)

T=2.0:     "The meaning of life is quantum basketball friendship the the the purple underneath."
            (incoherent noise, too much randomness)
```

If you evaluate a model with greedy decoding and declare it "bad," you may be evaluating the decoding strategy, not the model. This is why research papers always specify decoding parameters, and why API providers expose temperature/top-p controls.

**What production systems actually use:**

| Application | Strategy | Why |
|------------|----------|-----|
| Code generation | T=0.2, top-p=0.95 | Code needs to be correct, not creative |
| Chat/assistant | T=0.7, top-p=0.9 | Balance of coherence and natural variation |
| Creative writing | T=1.0, top-p=0.95 | More diversity and surprise |
| Translation | Beam search (k=4–5) | There's usually one "right" answer |
| Summarization | T=0.3, top-p=0.9 | Stay close to the source material |

---

## Part 5: Architecture: Decoder-Only, Masking, Normalization, and FFN

### Q: Why did decoder-only architectures win over encoder-decoder?

Four compounding reasons:

**1. Parameter efficiency.** In an encoder-decoder, you have ~2× the parameters split between understanding (encoder) and generating (decoder). In decoder-only, 100% of parameters serve one objective: predict the next token.

**Concrete comparison:** A 7B encoder-decoder splits ~3.5B for encoding + ~3.5B for decoding. A 7B decoder-only uses all 7B for every task. For a task like summarization, the decoder-only model has 2× the effective capacity.

**2. Emergent abilities from scale.** In-context learning, chain-of-thought reasoning, and instruction following all emerged from autoregressive pre-training at scale. These capabilities were not designed: they appeared when decoder-only models got big enough. Encoder-only (BERT) and encoder-decoder (T5) models don't exhibit the same emergent behaviors.

**3. Universal task format.** Every task becomes next-token prediction:
```
Translation:    "Translate to French: Hello"     → " Bonjour"
Classification: "Classify sentiment: Great movie!" → " Positive"
Code:           "def fibonacci(n):"              → "    if n <= 1: return n..."
Math:           "What is 2+2?"                   → " 4"
QA:             "Context: ... Question: ..."     → " Answer text"
```
No task-specific heads, no architectural changes. One model, one training loop, every task.

**4. Engineering simplicity.** One KV cache (not two). One forward pass path. One inference optimization stack. Production serving (vLLM, TGI, TensorRT-LLM) only needs to optimize one architecture.

### Q: Can we use a decoder-only model for translation?

**Yes.** You simply frame translation as next-token prediction:

```
Input:  "Translate English to French: The cat sat on the mat."
Output: " Le chat s'est assis sur le tapis."
```

The model generates the French translation token by token, exactly like any other text. No encoder needed: the prompt IS the "encoding" of the source sentence. The causal attention over the prompt tokens serves the same role as cross-attention in an encoder-decoder.

GPT-4, Claude, and LLaMA all do translation this way. They match or exceed specialized encoder-decoder translation models on most language pairs, despite never being explicitly trained on parallel corpora.

**One caveat:** For pure machine translation (WMT benchmarks), encoder-decoder models are slightly more efficient because the encoder processes the source bidirectionally and cross-attention is specifically designed for source-target alignment. But the versatility of decoder-only models (one model does everything) outweighs this marginal advantage.

### Q: What is the structure of a decoder-only model?

A decoder-only model is the Transformer decoder with two modifications:
1. **Remove cross-attention**: no encoder output to attend to
2. **Remove the encoder entirely**

What remains per layer:

```
Input x (B, T, d_model)
  │
  ├─→ RMSNorm(x)
  ├─→ Masked Self-Attention (causal mask: token i sees only tokens ≤ i)
  │     Q, K, V = RMSNorm(x) @ W_Q, W_K, W_V
  │     scores = (Q @ K.T) / √d_k + causal_mask
  │     attn = softmax(scores) @ V
  │     out = attn @ W_O
  ├─→ x = x + out                    ← residual
  │
  ├─→ RMSNorm(x)
  ├─→ FFN (SwiGLU)
  │     gate = swish(RMSNorm(x) @ W_gate)
  │     up = RMSNorm(x) @ W_up
  │     out = (gate ⊙ up) @ W_down
  ├─→ x = x + out                    ← residual
  │
  Output x (B, T, d_model)
```

Repeat this block N times (32 for 7B, 80 for 70B). After all layers:

```
final = RMSNorm(x)                   ← final normalization
logits = final @ W_embedding.T       ← (B, T, vocab_size)
```

The output logits at position t predict token t+1. During training, cross-entropy loss is computed at all positions simultaneously. During inference, only the logit at the last position matters.

### Q: How does causal masking work? Show the math.

The causal mask is a matrix added to the attention scores before softmax. For a 4-token sequence:

```
                t₁   t₂   t₃   t₄
    mask = t₁ [  0   -∞   -∞   -∞ ]
           t₂ [  0    0   -∞   -∞ ]
           t₃ [  0    0    0   -∞ ]
           t₄ [  0    0    0    0 ]
```

**Concrete example.** Raw attention scores (QK^T / √d_k) for 4 tokens:

```
scores = [[ 2.1,  0.8,  1.5,  0.3 ],     ← token 1's raw scores
          [ 1.2,  1.9,  0.7,  2.8 ],     ← token 2's raw scores
          [ 0.4,  3.1,  1.7,  0.9 ],     ← token 3's raw scores
          [ 0.2,  0.6,  2.5,  1.8 ]]     ← token 4's raw scores
```

After adding mask:
```
masked = [[ 2.1,  -∞,   -∞,   -∞  ],     ← token 1 can only see itself
          [ 1.2,  1.9,  -∞,   -∞  ],     ← token 2 sees tokens 1-2
          [ 0.4,  3.1,  1.7,  -∞  ],     ← token 3 sees tokens 1-3
          [ 0.2,  0.6,  2.5,  1.8 ]]     ← token 4 sees all
```

After softmax (e^(-∞) = 0):
```
weights = [[ 1.000, 0.000, 0.000, 0.000 ],  ← token 1: 100% self
           [ 0.331, 0.669, 0.000, 0.000 ],  ← token 2: 33% on t1, 67% on t2
           [ 0.056, 0.834, 0.110, 0.000 ],  ← token 3: mostly attends to t2
           [ 0.052, 0.078, 0.521, 0.349 ]]  ← token 4: mostly t3 and t4
```

Each row sums to 1. No token attends to future tokens. This is mathematically equivalent to processing tokens one at a time, but computed in one matrix operation.

### Q: What is the KV cache and why does it matter?

During autoregressive generation, each new token needs to attend to all previous tokens. Without caching, generating token N requires computing Q, K, V projections for all N tokens, that's O(N²) total projection work.

The KV cache stores the Key and Value projections from all prior tokens:

**Step-by-step: generating 3 tokens with KV cache:**

```
Step 1: Process prompt "Hello how are" (3 tokens)
  - Compute Q, K, V for all 3 tokens (parallel, like training)
  - Store K₁, K₂, K₃ and V₁, V₂, V₃ in cache
  - Use Q₃ (last position) to get next-token logits → generates "you"

Step 2: Generate "you" (1 token)
  - Compute Q₄, K₄, V₄ for ONLY the new token
  - Append K₄ to cache: [K₁, K₂, K₃, K₄]
  - Append V₄ to cache: [V₁, V₂, V₃, V₄]
  - Q₄ × [K₁,K₂,K₃,K₄]ᵀ → attention scores → weight [V₁,V₂,V₃,V₄]
  - Generates "?"

Step 3: Generate "?" (1 token)
  - Compute Q₅, K₅, V₅ for ONLY the new token
  - Cache grows: [K₁,...,K₅] and [V₁,...,V₅]
  - One dot product against 5 cached keys
```

Without cache: step 3 would recompute K and V for all 5 tokens. With cache: only 1 new K, V computation.

**KV cache memory cost** for a 70B model (80 layers, d_model=8192, 4096 tokens):

$$\text{KV cache} = 2_{(K+V)} \times 80_{layers} \times 4096_{tokens} \times 8192_{d\_model} \times 2_{bytes} = 10.7\text{ GB per sequence}$$

This is why **GQA** (Grouped-Query Attention) matters: with 8 KV heads instead of 64, the cache shrinks to ~1.3 GB. It's also why batch sizes during inference are memory-bound: each additional sequence in the batch needs its own 10.7 GB cache.

### Q: Why LayerNorm instead of BatchNorm?

Sequences in a batch have different lengths. BatchNorm normalizes across the batch dimension: it computes mean and variance over all examples at each feature position. With variable-length sequences padded to equal length, the padding tokens corrupt these statistics.

**Concrete example: BatchNorm fails:**
```
Batch of 2 sequences (padded to length 4):
Seq 1: [0.5, 0.8, 0.3, PAD]    ← real tokens + 1 pad
Seq 2: [0.7, PAD, PAD, PAD]    ← real token + 3 pads

BatchNorm at position 2: mean of [0.8, PAD_value] ← PAD corrupts statistics
```

LayerNorm normalizes across the feature dimension for each token independently, no batch-level dependency, no sensitivity to padding:
```
LayerNorm on token "cat" at position 2:
  x = [0.5, -0.3, 0.8, ..., 0.1]  (d_model values)
  μ = mean(x), σ = std(x)           (computed from this token only)
  output = (x - μ) / σ              (no dependency on other sequences)
```

### Q: What is Pre-RMSNorm and why does it replace Post-LayerNorm?

**Post-LayerNorm** (original Transformer):

$$x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

The normalization happens **after** the residual addition. The problem: the residual stream gets normalized at every layer, which distorts the accumulated signal and makes training unstable for deep networks (40+ layers).

**Pre-RMSNorm** (LLaMA, modern models):

$$x_{out} = x + \text{Sublayer}(\text{RMSNorm}(x))$$

The normalization happens **before** the sublayer, and the residual addition happens **after**, on the raw signal. The residual stream flows through all layers without ever being normalized: it stays "clean."

**LayerNorm vs RMSNorm: the math:**

LayerNorm:
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$
$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2 + \epsilon}$$

RMSNorm:
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

The differences:
1. **No mean subtraction** ($\mu$): empirically doesn't matter at scale
2. **No bias** ($\beta$): one fewer parameter vector per norm layer
3. **~10–15% faster**: fewer operations

**Concrete example.** Input vector x = [1.0, -2.0, 3.0, -1.0]:

LayerNorm:
$$\mu = (1 - 2 + 3 - 1)/4 = 0.25$$
$$\sigma = \sqrt{((0.75)^2 + (-2.25)^2 + (2.75)^2 + (-1.25)^2)/4} = \sqrt{3.6875} = 1.92$$
$$\text{output} = [0.75, -2.25, 2.75, -1.25] / 1.92 = [0.39, -1.17, 1.43, -0.65]$$

RMSNorm:
$$\text{RMS} = \sqrt{(1 + 4 + 9 + 1)/4} = \sqrt{3.75} = 1.94$$
$$\text{output} = [1.0, -2.0, 3.0, -1.0] / 1.94 = [0.52, -1.03, 1.55, -0.52]$$

RMSNorm preserves the sign pattern and relative magnitudes, but doesn't center the data around zero. For transformers at scale, this is sufficient.

### Q: What is SwiGLU and why does it replace ReLU in the FFN?

**Original FFN (ReLU):**

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1) \cdot W_2 + b_2$$

Where $\text{ReLU}(z) = \max(0, z)$. Two weight matrices: $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$.

**Problem with ReLU:** Any input that produces a negative pre-activation is **killed**: the output is exactly 0, and the gradient is exactly 0. These "dead neurons" waste parameters. Roughly 50% of neurons are dead for any given input.

**SwiGLU FFN (LLaMA):**

$$\text{SwiGLU}(x) = (\text{Swish}(xW_{gate}) \odot xW_{up}) \cdot W_{down}$$

Where:

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

Three weight matrices: $W_{gate} \in \mathbb{R}^{d \times d_{ff}}$, $W_{up} \in \mathbb{R}^{d \times d_{ff}}$, $W_{down} \in \mathbb{R}^{d_{ff} \times d}$.

**Step-by-step with concrete numbers.** Input x = [1.0, -0.5, 2.0] (d_model = 3, d_ff = 4):

**ReLU FFN:**
```
h = x @ W₁ = [2.1, -0.8, 1.5, -3.0]    (3→4 projection)
h = ReLU(h) = [2.1, 0.0, 1.5, 0.0]      (2 of 4 neurons are DEAD)
out = h @ W₂                              (4→3 projection)
```

50% of the inner dimension is wasted. Those dead neurons contribute nothing to the output or the gradient.

**SwiGLU:**
```
gate = x @ W_gate = [2.1, -0.8, 1.5, -3.0]
gate = Swish(gate) = [2.1×σ(2.1), -0.8×σ(-0.8), 1.5×σ(1.5), -3.0×σ(-3.0)]
     = [2.1×0.891, -0.8×0.310, 1.5×0.818, -3.0×0.047]
     = [1.871, -0.248, 1.227, -0.142]     ← ALL neurons contribute

up   = x @ W_up = [0.5, 1.2, -0.3, 2.1]  (different projection)

h = gate ⊙ up = [1.871×0.5, -0.248×1.2, 1.227×(-0.3), -0.142×2.1]
  = [0.936, -0.298, -0.368, -0.298]       ← ALL neurons active

out = h @ W_down                           (d_ff→d projection)
```

Every neuron contributes. Swish is smooth (no hard zero cutoff), so gradients always flow. The gate mechanism (element-wise multiply ⊙) lets the network learn to selectively amplify or suppress features, a learnable form of information routing.

**Swish vs ReLU activation comparison on specific values:**

| Input z | ReLU(z) | Swish(z) | Gradient ReLU | Gradient Swish |
|:-------:|:-------:|:--------:|:-------------:|:--------------:|
| 3.0 | 3.0 | 2.86 | 1.0 | 1.09 |
| 0.5 | 0.5 | 0.31 | 1.0 | 0.73 |
| 0.0 | **0.0** | **0.0** | **0.0** | **0.5** |
| -0.5 | **0.0** | **-0.19** | **0.0** | **0.27** |
| -3.0 | **0.0** | **-0.14** | **0.0** | **-0.09** |

ReLU: hard cutoff at 0. Dead neurons have exactly zero gradient, permanently dead.
Swish: smooth transition. Even negative inputs produce small non-zero outputs and gradients. The network can "revive" a neuron if its input changes.

**Parameter cost:** SwiGLU has 3 matrices instead of 2, so d_ff is reduced from 4d to ~2.67d to keep total parameters similar:

$$\text{ReLU: } 2 \times d \times 4d = 8d^2$$
$$\text{SwiGLU: } 3 \times d \times 2.67d = 8d^2$$

Same total parameters, significantly better performance.

### Q: What is FlashAttention and what problem does it solve?

Standard attention materializes the full S × S attention matrix in GPU HBM (high-bandwidth memory). This has two problems:

1. **Memory:** O(S²), for S = 32,768, that's 32K × 32K × 2 bytes = **2 GB per head per layer**
2. **Speed:** Reading/writing this matrix from HBM is the bottleneck, not computing it

**The GPU memory hierarchy:**
```
SRAM (on-chip): ~20 MB, 19 TB/s bandwidth     ← FAST but tiny
HBM (off-chip): ~80 GB, 2 TB/s bandwidth      ← 10× slower but large
```

Standard attention: compute QK^T → write full S×S matrix to HBM → read it back for softmax → write softmax result to HBM → read it back to multiply by V. Four HBM round-trips for one attention layer.

**FlashAttention:** Compute attention in tiles. Load a block of Q, K, V into SRAM. Compute the partial attention for that block. Accumulate the result. Never write the full S×S matrix to HBM.

```
Standard:      Q, K, V in HBM → compute S×S in HBM → output in HBM
FlashAttention: Q, K, V in HBM → load tiles to SRAM → compute in SRAM → output to HBM
```

**Results:** Memory drops from O(S²) to O(S). Speed improves 2–4×. The output is **bit-identical**, no approximation, just a smarter computation order. This is what made long-context (32K+) models practical.

### Q: What's the difference between absolute and relative positional encoding?

**Absolute (sinusoidal or learned):** Assigns a fixed vector to each position, added once at the input before layer 1. Position 0 always gets the same encoding regardless of context.

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Problem:** A model trained on max 4K positions has never seen position 5000. At inference, if you try sequence length 5K, the positional encoding is out-of-distribution. The model degrades.

**Relative (RoPE):** Encodes the **distance** between tokens by rotating Q and K vectors at every layer. The attention score becomes a function of (position_m − position_n), not position_m and position_n independently.

$$q_m \cdot k_n = (R_m q) \cdot (R_n k) = q^T R_{m-n} k$$

Where $R_m$ is a rotation matrix determined by position m. The dot product only depends on the **difference** m−n.

**Why this matters: concrete example:**

```
"The cat sat" at positions [0, 1, 2]:   "cat" attends to "The" at distance 1
"The cat sat" at positions [50, 51, 52]: "cat" attends to "The" at distance 1

Absolute: different PE vectors → different attention patterns → inconsistent
RoPE:     same relative distance → same attention pattern → consistent
```

**Extrapolation:** A RoPE model trained on 4K context learns "tokens 1 apart relate like this," "tokens 100 apart relate like this." When you extend to 128K, the relative distances still make sense. With simple frequency scaling (RoPE ABF), LLaMA 3.1 extended from 8K training to 128K context.

### Q: What would you change about the original architecture today?

Six changes, all adopted by LLaMA, Mistral, and modern models:

| Change | Original | Modern | Why |
|--------|----------|--------|-----|
| Normalization | Post-LayerNorm | Pre-RMSNorm | Stable training at scale, 10-15% faster |
| Positions | Sinusoidal (add once) | RoPE (rotate every layer) | Relative positions, length extrapolation |
| FFN activation | ReLU (2 matrices) | SwiGLU (3 matrices) | No dead neurons, learnable gating |
| Attention | Full MHA | GQA | 4× less KV-cache memory |
| Architecture | Encoder-decoder | Decoder-only | One model for all tasks |
| Bias terms | Present | Removed | Simplifies quantization, negligible expressivity loss |

---

## Summary: The Zero-Foot Checklist

If you can answer all of these from memory, with equations and concrete numbers, you have the zero-foot view:

| Topic | Can you... |
|-------|-----------|
| d_model, N, h, d_k, d_v, d_ff | Define each, give typical values, explain relationships? |
| √d_k scaling | Derive the variance proof and show softmax saturation? |
| Parameter count | Derive total params from hyperparameters for a 7B model? |
| Vocab size & weight sharing | Explain embedding param cost and the duality? |
| Attention complexity | Compute FLOPs for a 4096-token sequence? |
| BPE | Walk through the merge algorithm step by step? |
| Special tokens | List them, explain training vs inference usage? |
| Training parallelism | Explain 3 levels + why inference decode is sequential? |
| Learning rate schedule | Write the equation, compute values at specific steps? |
| Label smoothing | Write the smoothed target, explain the perplexity tradeoff? |
| Perplexity | Compute from probabilities on a 4-token example? |
| BLEU | Compute n-gram precisions, clipping, and brevity penalty? |
| Greedy decoding | Apply softmax, take argmax, explain the limitation? |
| Beam search | Expand beams, score with log probs, length-normalize? |
| Temperature | Compute the distribution at T=0.5 vs T=2.0 with numbers? |
| Top-k | Sort, truncate at k, renormalize? |
| Top-p | Sort, cumulate, find nucleus, explain adaptivity vs top-k? |
| Decoder-only structure | Draw the block, explain the causal mask with a 4×4 matrix? |
| KV cache | Walk through 3 generation steps, compute cache size for 70B? |
| LayerNorm vs BatchNorm | Explain why BatchNorm fails on variable-length sequences? |
| Pre-RMSNorm vs Post-LN | Write both equations, compute on x = [1, -2, 3, -1]? |
| SwiGLU vs ReLU | Write equations, show dead neurons vs smooth activation on concrete values? |
| FlashAttention | Explain SRAM vs HBM, tiling, memory reduction from O(S²) to O(S)? |
| RoPE vs sinusoidal | Explain relative vs absolute, the rotation, extrapolation? |

---

*This is Part 2 of my AI deep-dive series. [Part 1 covered the original Transformer architecture.]*

*For more AI research breakdowns, code, and visualizations, check out my work on [GitHub](https://github.com/mntalha).*
