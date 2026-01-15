# mg_behavior_learning/lm/behavior_head.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BehaviorHeadOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    lm_loss: torch.Tensor
    beh_loss: Optional[torch.Tensor]
    beh_logits: torch.Tensor
    hidden_states: Optional[Any] = None


class BehaviorHeadCausalLM(nn.Module):
    """
    Wrap a HF GPT2LMHeadModel (with cross-attn) and add a behavior classification head.
    Keeps .logits and .loss so sampler/trainer won't break.
    """
    def __init__(self, base_lm: nn.Module, num_behaviors: int = 10, dropout: float = 0.1, pooling: str = "mean"):
        super().__init__()
        self.base_lm = base_lm
        self.num_behaviors = num_behaviors
        self.pooling = pooling

        cfg = getattr(base_lm, "config", None)
        hidden = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if hidden is None:
            raise ValueError("Cannot infer hidden size from base_lm.config (need n_embd/hidden_size/d_model)")
        hidden = int(hidden)

        self.beh_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_behaviors),
        )

    def _pool(self, H: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # H: [B,T,D]
        if self.pooling == "first":
            return H[:, 0, :]
        if attention_mask is None:
            return H.mean(dim=1)
        m = attention_mask.unsqueeze(-1).float()  # [B,T,1]
        denom = m.sum(dim=1).clamp(min=1.0)
        return (H * m).sum(dim=1) / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        behavior_label: Optional[torch.Tensor] = None,
        lambda_beh: float = 0.1,
        **kwargs,
    ) -> BehaviorHeadOutput:
        out = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        lm_loss = out.loss
        logits = out.logits
        H = out.hidden_states[-1]  # [B,T,D]
        pooled = self._pool(H, attention_mask)
        beh_logits = self.beh_head(pooled)

        beh_loss = None
        loss = lm_loss
        if behavior_label is not None:
            beh_loss = F.cross_entropy(beh_logits, behavior_label)
            loss = lm_loss + float(lambda_beh) * beh_loss

        return BehaviorHeadOutput(
            loss=loss,
            logits=logits,
            lm_loss=lm_loss,
            beh_loss=beh_loss,
            beh_logits=beh_logits,
            hidden_states=out.hidden_states,
        )
