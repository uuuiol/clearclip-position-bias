"""ClearCLIP dense-inference wrapper around open_clip.

Implements the three last-block modifications from ClearCLIP
(https://www.alphaxiv.org/abs/2407.12442, https://github.com/mc-lan/ClearCLIP):

  1. drop_residual  — discard the residual connection X_res in the final block;
                       the visual token representation is X_attn alone.
  2. self_attn_mode — replace query-key attention with query-query ("qq") or
                       key-key ("kk") self-self attention in the final block.
  3. drop_ffn       — discard the feed-forward network in the final block.

All three are training-free: no weights are changed, only which tensors are
kept during the forward pass of the *last* transformer block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import open_clip


@dataclass
class ClearCLIPConfig:
    backbone: str = "ViT-B-16"
    pretrained: str = "openai"
    self_attn_mode: str = "qq"     # "qq" or "kk"
    drop_residual: bool = True
    drop_ffn: bool = True
    device: str = "cuda"


class ClearCLIPVisualEncoder:
    """Wraps an open_clip CLIP model to emit dense (per-patch) embeddings."""

    def __init__(self, cfg: ClearCLIPConfig):
        device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(device)

        # OpenAI's original CLIP checkpoints were trained with QuickGELU, not
        # the standard GELU open_clip defaults to for a plain "ViT-B-16" model
        # name. Silently mismatched activations degrade the whole network,
        # not just the last block — force it whenever pretrained="openai"
        # (leave other pretrained tags, e.g. laion2b, alone: those were
        # actually trained with standard GELU).
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.backbone, pretrained=cfg.pretrained,
            force_quick_gelu=(cfg.pretrained == "openai"),
        )
        self.tokenizer = open_clip.get_tokenizer(cfg.backbone)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.cfg = cfg

        visual = self.model.visual
        # open_clip's timm-free ViT stores blocks in `transformer.resblocks`.
        self.resblocks = visual.transformer.resblocks
        self.patch_size = visual.patch_size[0] if isinstance(visual.patch_size, (tuple, list)) else visual.patch_size

    @torch.no_grad()
    def encode_text(self, classnames: list[str], templates: list[str]) -> torch.Tensor:
        """Returns L2-normalized text embeddings, shape (n_classes, d)."""
        embeds = []
        for name in classnames:
            prompts = [t.format(name) for t in templates]
            tokens = self.tokenizer(prompts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1).mean(dim=0)
            embeds.append(F.normalize(feats, dim=0))
        return torch.stack(embeds, dim=0)

    @torch.no_grad()
    def _last_block_forward(self, x: torch.Tensor, block):
        """Custom forward for the FINAL resblock only, implementing the
        ClearCLIP surgery. `x` is (seq_len, batch, width) — open_clip's ViT
        uses this (L, N, D) layout internally before the final permute.

        Returns (x_visual, attn_weights) where attn_weights is head-averaged,
        shape (N, L, L) — kept for the Phase-1 attention-centrality diagnostic.
        """
        ln1_out = block.ln_1(x)

        attn = block.attn  # nn.MultiheadAttention
        embed_dim = attn.embed_dim
        num_heads = attn.num_heads
        head_dim = embed_dim // num_heads

        # Reproduce in_proj -> q, k, v exactly as nn.MultiheadAttention does,
        # so we can substitute the query-key pairing before the softmax.
        w = attn.in_proj_weight
        b = attn.in_proj_bias
        q_w, k_w, v_w = w.chunk(3, dim=0)
        q_b, k_b, v_b = (b.chunk(3, dim=0) if b is not None else (None, None, None))

        q = F.linear(ln1_out, q_w, q_b)
        k = F.linear(ln1_out, k_w, k_b)
        v = F.linear(ln1_out, v_w, v_b)

        L, N, _ = q.shape
        def reshape_heads(t):
            return t.reshape(L, N * num_heads, head_dim).transpose(0, 1)  # (N*heads, L, head_dim)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        if self.cfg.self_attn_mode == "qq":
            sim = torch.bmm(q, q.transpose(1, 2))
        elif self.cfg.self_attn_mode == "kk":
            sim = torch.bmm(k, k.transpose(1, 2))
        else:
            raise ValueError(f"unknown self_attn_mode: {self.cfg.self_attn_mode}")

        sim = sim / math.sqrt(head_dim)
        attn_weights = sim.softmax(dim=-1)
        out = torch.bmm(attn_weights, v)  # (N*heads, L, head_dim)
        out = out.transpose(0, 1).reshape(L, N, embed_dim)
        attn_out = F.linear(out, attn.out_proj.weight, attn.out_proj.bias)

        if self.cfg.drop_residual:
            x_visual = attn_out
        else:
            x_visual = x + attn_out

        if not self.cfg.drop_ffn:
            x_visual = x_visual + block.mlp(block.ln_2(x_visual))

        # (N*heads, L, L) -> average over heads -> (N, L, L). Two views are
        # kept because Phase 1 (diagnostics) and Phase 2 Method B
        # (calibration) need different stages of the same computation:
        #   attn_weights_avg — POST-softmax, rows sum to 1 -> the diagnostic's
        #                      c_i = sum_j A_ij*(1-r_j) weighted-average needs this.
        #   sim_avg          — PRE-softmax (already /sqrt(head_dim)) -> Method B
        #                      re-does softmax_j(sim/tau + position_bias), which
        #                      requires starting from logits, not probabilities.
        attn_weights_avg = attn_weights.reshape(N, num_heads, L, L).mean(dim=1)
        sim_avg = sim.reshape(N, num_heads, L, L).mean(dim=1)

        # (N*heads, L, head_dim) -> merge heads back to (N, L, embed_dim).
        # This is the value vector Method B needs to re-apply a *different*
        # attention matrix to (out = attn_weights' @ v); note it operates in
        # the head-merged space (see calibration.py's Method B docstring for
        # why this is a documented approximation, not an exact per-head redo).
        v_merged = v.transpose(0, 1).reshape(L, N, embed_dim).transpose(0, 1)

        return x_visual, attn_weights_avg, sim_avg, v_merged

    def shared_params(self) -> dict:
        """Post-last-attention parameters Method B needs to redo ln_post ->
        visual.proj -> normalize from a *recomputed* attn output. Shared
        across every image for a fixed backbone/checkpoint, so callers cache
        this once (see run.py's `_shared.npz`), not per image.
        """
        visual = self.model.visual
        block = self.resblocks[-1]
        return {
            "out_proj_weight": block.attn.out_proj.weight.detach().cpu().numpy(),
            "out_proj_bias": block.attn.out_proj.bias.detach().cpu().numpy(),
            "ln_post_weight": visual.ln_post.weight.detach().cpu().numpy(),
            "ln_post_bias": visual.ln_post.bias.detach().cpu().numpy(),
            "ln_post_eps": float(visual.ln_post.eps),
            "visual_proj": visual.proj.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def encode_image_dense(self, pixel_values: torch.Tensor):
        """Returns (patch_embeds, grid_h, grid_w, patch_attn, patch_sim, v_patches).

        patch_embeds: (n_patches, d) L2-normalized, CLS token dropped.
        patch_attn:   (n_patches, n_patches) POST-softmax, head-averaged —
                      Phase-1 diagnostics (attention_centrality).
        patch_sim:    (n_patches, n_patches) PRE-softmax, head-averaged —
                      Phase-2 Method B re-derives its own softmax from this.
        v_patches:    (n_patches, embed_dim) head-merged value vectors,
                      pre-out_proj — Method B's `A' @ v_patches` input.
        """
        visual = self.model.visual
        x = pixel_values.to(self.device)

        # Standard CLIP patch embedding + positional embedding, replicated
        # from open_clip.transformer.VisionTransformer.forward up to the
        # transformer stack so we can intercept only the final block.
        x = visual.conv1(x)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (N, n_patches, D)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x) if hasattr(visual, "patch_dropout") else x
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # -> (L, N, D) for resblocks
        for block in self.resblocks[:-1]:
            x = block(x)

        x_last, attn_weights, sim, v_merged = self._last_block_forward(x, self.resblocks[-1])
        x_last = x_last.permute(1, 0, 2)  # -> (N, L, D)

        patch_tokens = x_last[:, 1:, :]  # drop CLS position, keep spatial patches
        patch_tokens = visual.ln_post(patch_tokens)
        if visual.proj is not None:
            patch_tokens = patch_tokens @ visual.proj

        patch_embeds = F.normalize(patch_tokens, dim=-1).squeeze(0)
        # drop CLS row/col/token so callers get pure patch-to-patch tensors
        patch_attn = attn_weights[:, 1:, 1:].squeeze(0)
        patch_sim = sim[:, 1:, 1:].squeeze(0)
        v_patches = v_merged[:, 1:, :].squeeze(0)
        return patch_embeds, grid_h, grid_w, patch_attn, patch_sim, v_patches

    @torch.no_grad()
    def dense_logits(self, pixel_values: torch.Tensor, text_embeds: torch.Tensor):
        """Returns (logits, grid_h, grid_w, patch_attn, patch_sim, v_patches).
        logits: (n_patches, n_classes) cosine similarity, NOT yet calibrated —
        calibration.py's Method A mixes each patch's logits with an
        attention-weighted average of the rest (position-conditioned) on top
        of `logits`; Method B instead recomputes logits from
        `patch_sim`/`v_patches` with a reweighted attention (see
        calibration.py). patch_attn/patch_sim/v_patches are all kept for
        exactly these two downstream uses, not needed for the plain baseline.
        """
        patch_embeds, grid_h, grid_w, patch_attn, patch_sim, v_patches = self.encode_image_dense(pixel_values)
        logits = patch_embeds @ text_embeds.T
        return logits, grid_h, grid_w, patch_attn, patch_sim, v_patches
