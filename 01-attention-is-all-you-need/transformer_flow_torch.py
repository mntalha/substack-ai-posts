"""
Transformer Forward-Pass Walkthrough  —  PyTorch edition
=========================================================
Run:   python transformer_flow_torch.py
Debug: set breakpoints on any "# ── Stage ..." line and step through.

Stages ①–㉑ map 1-to-1 to  →  images/10-interactive-deep-dive.html

Dimensions are tiny so you can inspect every value in the debugger:
  vocab=50  d_model=16  n_heads=2  d_k=8  d_ff=64  S=6  T=4  layers=2

Prerequisites:  pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)

# ─── hyper-parameters ──────────────────────────────────────────────
VOCAB   = 50
D_MODEL = 16
N_HEADS = 2
D_K     = D_MODEL // N_HEADS   # 8
D_FF    = 64
S       = 6      # source seq len
T       = 4      # target seq len
N_LAYERS = 2

# ═══════════════════════════════════════════════════════════════════
#  Weight initialisation  (Xavier uniform, like the original paper)
# ═══════════════════════════════════════════════════════════════════

def init_linear(in_dim, out_dim):
    """Return (weight, bias) tensors for a linear layer."""
    w = torch.empty(in_dim, out_dim)
    torch.nn.init.xavier_uniform_(w)
    b = torch.zeros(out_dim)
    return w, b

# Shared embedding matrix
E = torch.empty(VOCAB, D_MODEL)
torch.nn.init.xavier_uniform_(E)

# Sinusoidal positional encoding  (fixed, not learned)
def build_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

PE = build_pe(max(S, T), D_MODEL)


# ─── per-layer weight containers ──────────────────────────────────
class AttnW:
    def __init__(self):
        self.Wq, self.bq = init_linear(D_MODEL, D_MODEL)
        self.Wk, self.bk = init_linear(D_MODEL, D_MODEL)
        self.Wv, self.bv = init_linear(D_MODEL, D_MODEL)
        self.Wo, self.bo = init_linear(D_MODEL, D_MODEL)

class FFNW:
    def __init__(self):
        self.W1, self.b1 = init_linear(D_MODEL, D_FF)
        self.W2, self.b2 = init_linear(D_FF, D_MODEL)

class NormW:
    def __init__(self):
        self.gamma = torch.ones(D_MODEL)
        self.beta  = torch.zeros(D_MODEL)

class EncLayerW:
    def __init__(self):
        self.attn  = AttnW()
        self.norm1 = NormW()
        self.ffn   = FFNW()
        self.norm2 = NormW()

class DecLayerW:
    def __init__(self):
        self.self_attn  = AttnW()
        self.norm1      = NormW()
        self.cross_attn = AttnW()
        self.norm2      = NormW()
        self.ffn        = FFNW()
        self.norm3      = NormW()

enc_layers = [EncLayerW() for _ in range(N_LAYERS)]
dec_layers = [DecLayerW() for _ in range(N_LAYERS)]
W_out, b_out = init_linear(D_MODEL, VOCAB)


# ═══════════════════════════════════════════════════════════════════
#  Pretty-printing helpers
# ═══════════════════════════════════════════════════════════════════

def banner(num, title, shapes):
    print()
    print("=" * 72)
    print(f"  {num}  {title}")
    print(f"      shapes: {shapes}")
    print("=" * 72)

def peek(t: torch.Tensor, label="", rows=2, cols=4):
    """Show top-left corner of a 2-D tensor."""
    if t.dim() == 3:
        t = t.squeeze(0)          # drop batch dim for display
    for i in range(min(rows, t.size(0))):
        vals = "  ".join(f"{t[i, j]:+.4f}" for j in range(min(cols, t.size(1))))
        extra = "  …" if t.size(1) > cols else ""
        print(f"    row {i}: [{vals}{extra}]")
    if t.size(0) > rows:
        print(f"    … ({t.size(0) - rows} more rows)")


# ═══════════════════════════════════════════════════════════════════
#  LayerNorm  (manual, so you can see every step)
# ═══════════════════════════════════════════════════════════════════

def layer_norm(x, gamma, beta, eps=1e-5):
    """x: (B, seq, D) → (B, seq, D)"""
    mu  = x.mean(dim=-1, keepdim=True)               # per-token mean
    var = x.var(dim=-1, keepdim=True, unbiased=False) # per-token variance
    normed = (x - mu) / torch.sqrt(var + eps)
    return gamma * normed + beta


# ═══════════════════════════════════════════════════════════════════
#  Multi-Head Attention  (step by step, NOT using nn.MultiheadAttention)
# ═══════════════════════════════════════════════════════════════════

def multi_head_attention(query_in, key_in, value_in, aw, mask=None,
                         stage_prefix="④–⑨", verbose=True):
    """
    query_in : (1, seq_q, D_MODEL)
    key_in   : (1, seq_k, D_MODEL)
    value_in : (1, seq_k, D_MODEL)
    mask     : None or (1, 1, seq_q, seq_k) broadcastable
    Returns  : (1, seq_q, D_MODEL)
    """
    B, seq_q, _ = query_in.shape
    seq_k = key_in.size(1)

    # ── Q, K, V projections  (linear layers) ──────────────────────
    Q = query_in @ aw.Wq + aw.bq      # (1, seq_q, D_MODEL)
    K = key_in   @ aw.Wk + aw.bk      # (1, seq_k, D_MODEL)
    V = value_in @ aw.Wv + aw.bv      # (1, seq_k, D_MODEL)

    if verbose:
        banner(f"{stage_prefix} Q,K,V", "Linear Projections",
               f"({seq_q},{D_MODEL}) @ W({D_MODEL},{D_MODEL}) → Q,K,V")
        print(f"  Q {tuple(Q.shape)}  K {tuple(K.shape)}  V {tuple(V.shape)}")
        peek(Q, "Q")

    # ── Reshape into heads ────────────────────────────────────────
    #  (1, seq, D_MODEL) → (1, N_HEADS, seq, D_K)
    Q = Q.view(B, seq_q, N_HEADS, D_K).transpose(1, 2)   # (1,2,seq_q,8)
    K = K.view(B, seq_k, N_HEADS, D_K).transpose(1, 2)   # (1,2,seq_k,8)
    V = V.view(B, seq_k, N_HEADS, D_K).transpose(1, 2)   # (1,2,seq_k,8)

    if verbose:
        banner(f"{stage_prefix} reshape", f"Reshape into {N_HEADS} Heads",
               f"→ Q {tuple(Q.shape)}  K {tuple(K.shape)}  V {tuple(V.shape)}")
        print(f"  Head 0, token 0: Q = {Q[0,0,0].tolist()}")
        print(f"  Head 1, token 0: Q = {Q[0,1,0].tolist()}")

    # ── Scaled dot-product:  QK^T / √d_k ─────────────────────────
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D_K)
    # scores: (1, N_HEADS, seq_q, seq_k)

    if verbose:
        banner(f"{stage_prefix} QKᵀ/√{D_K}", "Scaled Dot-Product Scores",
               f"Q({seq_q},{D_K}) @ Kᵀ({D_K},{seq_k}) / √{D_K} → ({seq_q},{seq_k})")
        print(f"  scores {tuple(scores.shape)}")
        print(f"  Head 0 scores (raw):")
        peek(scores[0, 0])

    # ── Apply mask (causal for decoder self-attn) ─────────────────
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
        if verbose:
            print(f"\n  After masking (head 0):")
            peek(scores[0, 0])

    # ── Softmax → attention weights ───────────────────────────────
    attn_weights = F.softmax(scores, dim=-1)
    # attn_weights: (1, N_HEADS, seq_q, seq_k)

    if verbose:
        banner(f"{stage_prefix} softmax", "Softmax → Attention Weights",
               f"({seq_q},{seq_k}) → ({seq_q},{seq_k})  rows sum to 1")
        print(f"  Head 0:")
        peek(attn_weights[0, 0])
        print(f"    row 0 sum = {attn_weights[0, 0, 0].sum().item():.6f}")

    # ── Weighted sum of values ────────────────────────────────────
    head_out = torch.matmul(attn_weights, V)
    # head_out: (1, N_HEADS, seq_q, D_K)

    if verbose:
        banner(f"{stage_prefix} attn×V", "Attention × Values",
               f"({seq_q},{seq_k}) @ ({seq_k},{D_K}) → ({seq_q},{D_K})")
        print(f"  head_out {tuple(head_out.shape)}")
        print(f"  Head 0:")
        peek(head_out[0, 0])

    # ── Concat heads + W_O ────────────────────────────────────────
    # (1, N_HEADS, seq_q, D_K) → (1, seq_q, D_MODEL)
    concat = head_out.transpose(1, 2).contiguous().view(B, seq_q, D_MODEL)
    out = concat @ aw.Wo + aw.bo

    if verbose:
        banner(f"{stage_prefix} W_O", f"Concat {N_HEADS} Heads + W_O Projection",
               f"concat ({seq_q},{D_MODEL}) @ W_O ({D_MODEL},{D_MODEL}) → ({seq_q},{D_MODEL})")
        print(f"  concat[0,0,:6] = {concat[0,0,:6].tolist()}")
        print(f"  after W_O:")
        peek(out)

    return out, attn_weights   # return weights so you can inspect them


# ═══════════════════════════════════════════════════════════════════
#  Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════

def ffn_forward(x, fw, stage="⑪", verbose=True):
    """x: (1, seq, D_MODEL) → (1, seq, D_MODEL)"""
    seq = x.size(1)
    hidden = F.relu(x @ fw.W1 + fw.b1)    # (1, seq, D_FF)
    out = hidden @ fw.W2 + fw.b2           # (1, seq, D_MODEL)

    if verbose:
        banner(stage, "Feed-Forward Network (position-wise)",
               f"({seq},{D_MODEL}) → ({seq},{D_FF}) → ({seq},{D_MODEL})")
        alive = (hidden[0, 0] > 0).sum().item()
        print(f"  hidden {tuple(hidden.shape)}")
        print(f"  ReLU kept {alive}/{D_FF} neurons alive ({100*alive/D_FF:.0f}%)")
        print(f"  hidden[0,0,:8] = {hidden[0,0,:8].tolist()}")
        print(f"  output:")
        peek(out)
    return out


# ═══════════════════════════════════════════════════════════════════
#  Causal mask for decoder self-attention
# ═══════════════════════════════════════════════════════════════════

def causal_mask(seq_len):
    """Returns (1, 1, seq_len, seq_len) — 1 where allowed, 0 where blocked."""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════
#  MAIN FORWARD PASS  —  follow along with 10-interactive-deep-dive.html
# ═══════════════════════════════════════════════════════════════════

print("\n" + "▓" * 72)
print("  TRANSFORMER FORWARD PASS  (PyTorch)  —  step-by-step")
print(f"  vocab={VOCAB}  d_model={D_MODEL}  heads={N_HEADS}  d_k={D_K}  "
      f"d_ff={D_FF}  S={S}  T={T}  layers={N_LAYERS}")
print("  → reference: images/10-interactive-deep-dive.html")
print("▓" * 72)


# ── Stage ① — Source Token IDs ─────────────────────────────────────
# BREAKPOINT HERE → inspect src_ids
src_ids = torch.randint(0, VOCAB, (1, S))   # (1, 6)
banner("①", "Source Token IDs",
       f"(1, {S})  — {S} random IDs from vocab of {VOCAB}")
print(f"  src_ids = {src_ids}")
print(f"  src_ids.shape = {tuple(src_ids.shape)}")
print(f"  dtype = {src_ids.dtype}")


# ── Stage ② — Token Embedding ─────────────────────────────────────
src_embed = E[src_ids]   # (1, S, D_MODEL) — indexing E rows by token ID
banner("②", "Token Embedding",
       f"E[src_ids]  →  {tuple(src_embed.shape)}")
print(f"  E.shape = {tuple(E.shape)}  (vocab × d_model)")
print(f"  src_embed.shape = {tuple(src_embed.shape)}")
print(f"  Token 0 embedding (first 8 dims):")
print(f"    {src_embed[0, 0, :8].tolist()}")


# ── Stage ③ — + Positional Encoding ───────────────────────────────
x = src_embed + PE[:S]   # (1, S, D_MODEL) — broadcast add
banner("③", "+ Positional Encoding",
       f"({S},{D_MODEL}) + PE({S},{D_MODEL}) → ({S},{D_MODEL})  element-wise add")
print(f"  PE.shape = {tuple(PE.shape)}")
print(f"  PE[0, :8] = {PE[0, :8].tolist()}")
print(f"  PE[1, :8] = {PE[1, :8].tolist()}")
print(f"  x = embed + PE   x.shape = {tuple(x.shape)}")
print(f"  x[0, 0, :8] = {x[0, 0, :8].tolist()}")
print(f"  ↑ this is the input to encoder layer 1")


# ═══════════════════════════════════════════════════════════════════
#  ENCODER  (stages ④–⑫ repeated per layer)
# ═══════════════════════════════════════════════════════════════════

for layer_i in range(N_LAYERS):
    lw = enc_layers[layer_i]
    print("\n" + "─" * 72)
    print(f"  ╔══  ENCODER LAYER {layer_i + 1} / {N_LAYERS}  ══╗")
    print("─" * 72)

    verbose = (layer_i == 0)

    # ── Stages ④–⑨: Self-Attention ────────────────────────────────
    # BREAKPOINT → step into multi_head_attention to see Q,K,V,scores
    attn_out, enc_attn_w = multi_head_attention(
        x, x, x, lw.attn, mask=None,
        stage_prefix="④–⑨", verbose=verbose
    )

    # ── Stage ⑩ — Add & Norm ─────────────────────────────────────
    x = layer_norm(x + attn_out, lw.norm1.gamma, lw.norm1.beta)
    if verbose:
        banner("⑩", "Residual + LayerNorm",
               f"x + attn_out → LayerNorm → {tuple(x.shape)}")
        peek(x)

    # ── Stage ⑪ — FFN ────────────────────────────────────────────
    ffn_out = ffn_forward(x, lw.ffn, stage="⑪", verbose=verbose)

    # ── Stage ⑫ — Add & Norm ─────────────────────────────────────
    x = layer_norm(x + ffn_out, lw.norm2.gamma, lw.norm2.beta)
    if verbose:
        banner("⑫", "Residual + LayerNorm (post-FFN)",
               f"{tuple(x.shape)}  — end of encoder layer")
        peek(x)

    if not verbose:
        print(f"  (layer {layer_i+1}: same ops, output {tuple(x.shape)})")
        peek(x)


# ── Stage ⑬ — Encoder Final Output ────────────────────────────────
encoder_out = x   # (1, S, D_MODEL)
banner("⑬", "Encoder Final Output",
       f"{tuple(encoder_out.shape)} — sent to ALL decoder layers as K,V")
peek(encoder_out)
print(f"  ↑ computed ONCE, reused in every decoder layer's cross-attention")


# ═══════════════════════════════════════════════════════════════════
#  DECODER INPUT  (stages ⑭–⑮)
# ═══════════════════════════════════════════════════════════════════

# ── Stage ⑭ — Target Token IDs ───────────────────────────────────
tgt_ids = torch.randint(0, VOCAB, (1, T))   # (1, 4)
banner("⑭", "Target Token IDs (shifted right)",
       f"(1, {T})  — during training: correct output shifted right by 1")
print(f"  tgt_ids = {tgt_ids}")


# ── Stage ⑮ — Target Embedding + PE ──────────────────────────────
tgt_embed = E[tgt_ids]                       # (1, T, D_MODEL)
y = tgt_embed + PE[:T]                       # (1, T, D_MODEL)
banner("⑮", "Target Embedding + Positional Encoding",
       f"E[tgt_ids] + PE → {tuple(y.shape)}")
peek(y)


# ═══════════════════════════════════════════════════════════════════
#  DECODER  (stages ⑯–⑲ repeated per layer)
# ═══════════════════════════════════════════════════════════════════

cmask = causal_mask(T)   # (1, 1, T, T)  — lower triangular
print(f"\n  Causal mask {tuple(cmask.shape)}:")
print(f"    (1 = can attend, 0 = blocked)")
for i in range(T):
    row = "  ".join(f"{int(cmask[0,0,i,j])}" for j in range(T))
    print(f"    [{row}]")

for layer_i in range(N_LAYERS):
    dw = dec_layers[layer_i]
    print("\n" + "─" * 72)
    print(f"  ╔══  DECODER LAYER {layer_i + 1} / {N_LAYERS}  ══╗")
    print("─" * 72)

    verbose = (layer_i == 0)

    # ── Stage ⑯ — Masked Self-Attention ───────────────────────────
    if verbose:
        print(f"\n  ⑯ Masked Self-Attention")
        print(f"     Q, K, V all from decoder (y)  {tuple(y.shape)}")
        print(f"     Causal mask → token i sees only 0..i")

    # BREAKPOINT → inspect attn_weights to see the causal pattern
    self_attn_out, dec_self_w = multi_head_attention(
        y, y, y, dw.self_attn, mask=cmask,
        stage_prefix="⑯", verbose=verbose
    )

    y = layer_norm(y + self_attn_out, dw.norm1.gamma, dw.norm1.beta)
    if verbose:
        print(f"\n  After masked self-attn + residual + LayerNorm:")
        peek(y)

    # ── Stage ⑰ — Cross-Attention ────────────────────────────────
    if verbose:
        print("\n" + "=" * 72)
        print(f"  ⑰ Cross-Attention (Decoder → Encoder)")
        print(f"     Q from DECODER  {tuple(y.shape)}")
        print(f"     K,V from ENCODER {tuple(encoder_out.shape)}")
        print(f"     Scores will be ({T}×{S}) — no mask, full access")

    # BREAKPOINT → inspect cross_w to see which encoder tokens the decoder attends to
    cross_out, cross_w = multi_head_attention(
        y, encoder_out, encoder_out, dw.cross_attn, mask=None,
        stage_prefix="⑰", verbose=verbose
    )

    y = layer_norm(y + cross_out, dw.norm2.gamma, dw.norm2.beta)
    if verbose:
        print(f"\n  After cross-attn + residual + LayerNorm:")
        peek(y)

    # ── Stages ⑱–⑲ — FFN + Add & Norm ────────────────────────────
    ffn_out = ffn_forward(y, dw.ffn, stage="⑱", verbose=verbose)
    y = layer_norm(y + ffn_out, dw.norm3.gamma, dw.norm3.beta)

    if verbose:
        banner("⑲", "Residual + LayerNorm (end of decoder layer)",
               f"{tuple(y.shape)}")
        peek(y)
    else:
        print(f"  (layer {layer_i+1}: same ops, output {tuple(y.shape)})")
        peek(y)


# ═══════════════════════════════════════════════════════════════════
#  OUTPUT  (stages ⑳–㉑)
# ═══════════════════════════════════════════════════════════════════

# ── Stage ⑳ — Linear projection to vocab ─────────────────────────
logits = y @ W_out + b_out   # (1, T, VOCAB)
banner("⑳", "Linear Projection → Vocabulary",
       f"{tuple(y.shape)} @ W_out({D_MODEL},{VOCAB}) → {tuple(logits.shape)}")
print(f"  logits.shape = {tuple(logits.shape)}  — raw unnormalised scores")
print(f"  logits[0, 0, :10] = {logits[0, 0, :10].tolist()}")


# ── Stage ㉑ — Softmax → probabilities ────────────────────────────
probs = F.softmax(logits, dim=-1)   # (1, T, VOCAB)
banner("㉑", "Softmax → Output Probabilities",
       f"{tuple(logits.shape)} → {tuple(probs.shape)}  each row sums to 1")

for t in range(T):
    row = probs[0, t]
    top5_vals, top5_idx = row.topk(5)
    pred = top5_idx[0].item()
    print(f"  position {t}: predicted token = {pred}  (prob {top5_vals[0]:.4f})")
    for rank in range(5):
        idx = top5_idx[rank].item()
        p = top5_vals[rank].item()
        bar = "█" * int(p * 50)
        print(f"    #{rank+1}  token {idx:3d}  p={p:.4f}  {bar}")
    print(f"    row sum = {row.sum().item():.6f}")


# ═══════════════════════════════════════════════════════════════════
#  BONUS: Inspect key tensors in the debugger
# ═══════════════════════════════════════════════════════════════════

# Set a breakpoint on the line below and explore these in VS Code:
#   - encoder_out.shape → (1, 6, 16)
#   - enc_attn_w.shape  → (1, 2, 6, 6)  self-attn weights from last enc layer
#   - dec_self_w.shape  → (1, 2, 4, 4)  masked self-attn weights (causal)
#   - cross_w.shape     → (1, 2, 4, 6)  cross-attn: 4 decoder → 6 encoder
#   - logits.shape      → (1, 4, 50)    raw scores
#   - probs.shape       → (1, 4, 50)    probabilities
debug_stop = True   # ← BREAKPOINT HERE to inspect everything

print("\n" + "▓" * 72)
print("  DONE — Full encoder→decoder→output pass (PyTorch)")
print(f"  Input:  {S} source tokens → encoder → (1, {S}, {D_MODEL})")
print(f"  Output: {T} target positions → (1, {T}, {VOCAB}) probabilities")
print()
print("  Key tensors to inspect in debugger:")
print(f"    encoder_out  {tuple(encoder_out.shape)}  — final encoder repr")
print(f"    enc_attn_w   {tuple(enc_attn_w.shape)}  — encoder self-attn weights")
print(f"    dec_self_w   {tuple(dec_self_w.shape)}  — decoder masked self-attn")
print(f"    cross_w      {tuple(cross_w.shape)}  — cross-attn (decoder→encoder)")
print(f"    logits       {tuple(logits.shape)}  — raw scores before softmax")
print(f"    probs        {tuple(probs.shape)}  — final output probabilities")
print("▓" * 72)
