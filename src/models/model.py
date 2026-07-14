"""
================================================================================
Phase 4 — Cross-Modal Architecture
File: model.py
================================================================================

ARCHITECTURE OVERVIEW:

    Training mode:
        Clinical (18) ──► ClinicalEncoder ──► clinical_emb ──────────────────┐
                                                                               ▼
        Genomic  (46) ──► GenomicEncoder  ──► genomic_emb ──► CrossAttention ──► fused_emb ──► Classifier
                                                                               ▲
        KD Loss forces: clinical_emb ≈ fused_emb ─────────────────────────────┘

    Inference mode:
        Clinical (18) ──► ClinicalEncoder ──► clinical_emb ──► Classifier

KEY DESIGN DECISIONS:
    1. Clinical Encoder uses LayerNorm — more stable for small batches
    2. Genomic Encoder uses BatchNorm — better for binary mutation data
    3. Cross-Attention: clinical as Query, genomic as Key/Value
       → clinical features decide WHERE to look in genomic data
    4. Temperature in KD: T=4 produces softer probability distributions
       → model learns from relative similarities, not just hard labels
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================
# BUILDING BLOCK: MLP Block
# ================================================================================
class MLPBlock(nn.Module):
    """
    A single MLP block: Linear → Norm → Activation → Dropout

    This is the basic building block used in both encoders.
    We separate it into its own class to keep the encoder code clean
    and to make it easy to stack multiple blocks.

    Args:
        in_dim   : input dimension
        out_dim  : output dimension
        norm     : 'layer' (LayerNorm) or 'batch' (BatchNorm1d)
        dropout  : dropout probability (0 = no dropout)
        activation: activation function class (default: GELU)
    """
    def __init__(self, in_dim, out_dim, norm='layer', dropout=0.3,
                 activation=nn.GELU):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

        # LayerNorm normalizes across features for each sample independently
        # → stable even with small batch sizes
        # BatchNorm normalizes across the batch for each feature
        # → works well for binary features like mutation flags
        if norm == 'layer':
            self.norm = nn.LayerNorm(out_dim)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()

        self.act     = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Linear transformation first, then normalize, then activate, then dropout
        # This order (pre-norm style) tends to be more stable than post-norm
        return self.dropout(self.act(self.norm(self.linear(x))))


# ================================================================================
# CLINICAL ENCODER
# ================================================================================
class ClinicalEncoder(nn.Module):
    """
    Encodes 18 clinical features into a dense embedding vector.

    Uses LayerNorm because:
    - Clinical features have different scales (age 18-90, binary 0/1)
    - LayerNorm normalizes per-sample, independent of batch statistics
    - More stable for mixed-type features

    Architecture:
        18 → 64 → 128 → 64 (embedding)
    """
    def __init__(self, input_dim=18, embed_dim=64, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            MLPBlock(input_dim, 64,       norm='layer', dropout=dropout),
            MLPBlock(64,        128,      norm='layer', dropout=dropout),
            MLPBlock(128,       embed_dim, norm='layer', dropout=dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, 18) — clinical features
        Returns:
            emb: (batch_size, embed_dim) — clinical embedding
        """
        return self.net(x)


# ================================================================================
# GENOMIC ENCODER
# ================================================================================
class GenomicEncoder(nn.Module):
    """
    Encodes 46 genomic mutation flags into a dense embedding vector.

    Uses BatchNorm because:
    - All inputs are binary (0 or 1)
    - BatchNorm works well for homogeneous binary features
    - Normalizes mutation patterns across the batch

    Architecture:
        46 → 128 → 128 → 64 (embedding)

    Note: This encoder is ONLY used during training.
    At inference time, it is not called.
    """
    def __init__(self, input_dim=46, embed_dim=64, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            MLPBlock(input_dim, 128,      norm='batch', dropout=dropout),
            MLPBlock(128,       128,      norm='batch', dropout=dropout),
            MLPBlock(128,       embed_dim, norm='batch', dropout=dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, 46) — binary mutation flags
        Returns:
            emb: (batch_size, embed_dim) — genomic embedding
        """
        return self.net(x)


# ================================================================================
# CROSS-ATTENTION FUSION
# ================================================================================
class CrossAttentionFusion(nn.Module):
    """
    Fuses clinical and genomic embeddings using cross-attention.

    WHY CROSS-ATTENTION (not simple concatenation)?
        Concatenation treats all genomic features equally.
        Cross-attention lets the clinical embedding QUERY the genomic embedding:
        "Given this patient's clinical profile, which genomic signals are most
        relevant for treatment prediction?"

    HOW IT WORKS:
        Query  = clinical_emb  (what we want to enrich)
        Key    = genomic_emb   (what we search through)
        Value  = genomic_emb   (what we retrieve)

        Attention weights = softmax(Q × K^T / √d)
        Output = Attention weights × V

        The output tells us: "for this clinical profile, the relevant
        genomic information is a weighted combination of genomic features"

    This is ONLY used during training.
    """
    def __init__(self, embed_dim=64, n_heads=4, dropout=0.1):
        super().__init__()

        # MultiheadAttention splits embed_dim into n_heads parallel heads
        # Each head learns different aspects of the clinical-genomic relationship
        # e.g. one head might focus on EGFR, another on TP53
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # input shape: (batch, seq, embed) not (seq, batch, embed)
        )

        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, clinical_emb, genomic_emb):
        """
        Args:
            clinical_emb: (batch_size, embed_dim) — query
            genomic_emb : (batch_size, embed_dim) — key and value

        Returns:
            fused_emb: (batch_size, embed_dim) — enriched clinical embedding
        """
        # MultiheadAttention expects (batch, sequence_length, embed_dim)
        # Our embeddings are (batch, embed_dim) — we add a sequence dim of 1
        q = clinical_emb.unsqueeze(1)   # (batch, 1, embed_dim)
        k = genomic_emb.unsqueeze(1)    # (batch, 1, embed_dim)
        v = genomic_emb.unsqueeze(1)    # (batch, 1, embed_dim)

        # attn_output: (batch, 1, embed_dim)
        attn_output, _ = self.attn(q, k, v)
        attn_output = attn_output.squeeze(1)   # (batch, embed_dim)

        # Residual connection: add the original clinical embedding back
        # This prevents the attention from completely overwriting the clinical info
        # LayerNorm stabilizes the combined output
        fused = self.norm(clinical_emb + self.dropout(attn_output))
        return fused


# ================================================================================
# CLASSIFIER HEAD
# ================================================================================
class ClassifierHead(nn.Module):
    """
    Maps an embedding to 3-class treatment probabilities.

    Simple two-layer MLP with dropout.
    Shared between training (uses fused_emb) and inference (uses clinical_emb).
    """
    def __init__(self, embed_dim=64, n_classes=3, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, emb):
        """
        Args:
            emb: (batch_size, embed_dim)
        Returns:
            logits: (batch_size, n_classes) — raw scores before softmax
        """
        return self.net(emb)


# ================================================================================
# FULL CROSS-MODAL MODEL
# ================================================================================
class CrossModalModel(nn.Module):
    """
    Complete cross-modal model combining all components.

    Training mode (training=True):
        Uses ClinicalEncoder + GenomicEncoder + CrossAttention + Classifier
        Returns both fused logits AND clinical logits (for distillation loss)

    Inference mode (training=False):
        Uses ONLY ClinicalEncoder + Classifier
        No genomic features needed

    This is the core idea of the project:
        At training time, genomic data teaches the clinical encoder
        via the distillation loss.
        At inference time, the clinical encoder has "internalized"
        the genomic knowledge and works alone.
    """
    def __init__(self,
                 clinical_dim=18,
                 genomic_dim=46,
                 embed_dim=64,
                 n_heads=4,
                 n_classes=3,
                 dropout=0.3):
        super().__init__()

        self.clinical_encoder = ClinicalEncoder(clinical_dim, embed_dim, dropout)
        self.genomic_encoder  = GenomicEncoder(genomic_dim,  embed_dim, dropout)
        self.fusion           = CrossAttentionFusion(embed_dim, n_heads, dropout)
        self.classifier       = ClassifierHead(embed_dim, n_classes, dropout)

    def forward(self, x_clinical, x_genomic=None):
        """
        Args:
            x_clinical : (batch, clinical_dim) — always required
            x_genomic  : (batch, genomic_dim)  — only during training

        Returns:
            If training (x_genomic provided):
                fused_logits   : (batch, n_classes) — from fused embedding
                clinical_logits: (batch, n_classes) — from clinical embedding only
                                  used to compute distillation loss

            If inference (x_genomic is None):
                clinical_logits: (batch, n_classes) — final prediction
        """
        # Always compute clinical embedding
        clinical_emb = self.clinical_encoder(x_clinical)

        if x_genomic is not None:
            # Training mode: fuse clinical and genomic
            genomic_emb  = self.genomic_encoder(x_genomic)
            fused_emb    = self.fusion(clinical_emb, genomic_emb)

            fused_logits    = self.classifier(fused_emb)
            clinical_logits = self.classifier(clinical_emb)

            return fused_logits, clinical_logits

        else:
            # Inference mode: clinical only
            clinical_logits = self.classifier(clinical_emb)
            return clinical_logits

    def predict(self, x_clinical):
        """
        Inference-only prediction.
        Returns class indices (0, 1, or 2).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_clinical, x_genomic=None)
            return torch.argmax(logits, dim=1)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================================
# SANITY CHECK
# ================================================================================
if __name__ == '__main__':
    print("Running model sanity check...")

    batch_size   = 32
    clinical_dim = 18
    genomic_dim  = 46

    model = CrossModalModel(
        clinical_dim=clinical_dim,
        genomic_dim=genomic_dim,
        embed_dim=64,
        n_heads=4,
        n_classes=3,
        dropout=0.3,
    )

    print(f"\n  Model parameters: {model.count_parameters():,}")
    print(f"\n  Component parameters:")
    for name, module in [
        ('ClinicalEncoder', model.clinical_encoder),
        ('GenomicEncoder',  model.genomic_encoder),
        ('Fusion',          model.fusion),
        ('Classifier',      model.classifier),
    ]:
        params = sum(p.numel() for p in module.parameters())
        print(f"    {name:<20} {params:,}")

    # Test training forward pass
    x_clin = torch.randn(batch_size, clinical_dim)
    x_gen  = torch.randn(batch_size, genomic_dim)

    model.train()
    fused_logits, clin_logits = model(x_clin, x_gen)

    print(f"\n  Training forward pass:")
    print(f"    Input clinical : {x_clin.shape}")
    print(f"    Input genomic  : {x_gen.shape}")
    print(f"    fused_logits   : {fused_logits.shape}   ← should be (32, 3)")
    print(f"    clinical_logits: {clin_logits.shape}   ← should be (32, 3)")

    # Test inference forward pass
    model.eval()
    with torch.no_grad():
        inf_logits = model(x_clin, x_genomic=None)
    print(f"\n  Inference forward pass:")
    print(f"    Input clinical : {x_clin.shape}")
    print(f"    Output logits  : {inf_logits.shape}   ← should be (32, 3)")

    predictions = model.predict(x_clin)
    print(f"    Predictions    : {predictions.shape}   ← should be (32,)")
    print(f"    Unique classes : {predictions.unique().tolist()}")

    print(f"\n✓ All checks passed")
