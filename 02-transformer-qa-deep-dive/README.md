# 02 — Transformer Q&A: The Zero-Foot View

Every equation. Every computation. No hand-waving. The mathematical depth that separates "I know transformers" from "I can derive transformers on a whiteboard."

## Files

| File | Purpose |
|------|---------|
| `article.md` | Full Substack article — zero-foot Q&A with concrete math |

## Topics Covered

### Part 1 — Hyperparameter Glossary
- d_model, N, h, d_k, d_v, d_ff — what each controls with typical values
- vocab_size — impact on embeddings, parameter count, and sequence length
- Full parameter count breakdown (where do 7B parameters come from?)

### Part 2 — Tokenization
- Byte-Pair Encoding (BPE) — step-by-step merge algorithm with example
- Special tokens (BOS, EOS, PAD) — training vs inference usage
- Modern tokenizers: byte-level BPE, SentencePiece, tiktoken

### Part 3 — Training
- Parallel training (3 levels) vs sequential inference decode
- Learning rate schedule — exact equation, computed values, why warmup
- Label smoothing — the math, why it hurts perplexity but helps BLEU

### Part 4 — Decoding Strategies (with full numerical walkthrough)
- Greedy decoding — softmax → argmax
- Beam search — expand, score, prune, length normalize
- Temperature sampling — T=0.5 vs T=2.0 with exact numbers
- Top-k sampling — sort, truncate, renormalize
- Top-p (nucleus) sampling — cumulative threshold, adaptive behavior
- Why decoding strategy makes or breaks model evaluation

### Part 5 — Architecture
- Why decoder-only won over encoder-decoder
- Can decoder-only do translation?
- Full decoder-only block structure with causal mask

### Part 6 — Modern Changes
- Pre-RMSNorm vs Post-LayerNorm — both equations, computed on a vector
- SwiGLU vs ReLU FFN — equations, dead neuron problem, concrete example

### Part 7 — Evaluation Metrics
- Perplexity — formula, step-by-step calculation, interpretation
- BLEU score — n-gram precision, brevity penalty, full computation

## Publishing Checklist

- [ ] Article proofread
- [ ] Title and subtitle set
- [ ] Tags added
- [ ] Cross-post links ready
