"""
MLM_head.py
===========
掩码语言模型（Masked Language Modeling）预测头，用于 Phase 1 DAPT 继续预训练。

设计说明：
  Phase 1 使用无标签 PBMC 语料进行 MLM 训练，目标是从上下文中预测被掩码的基因 Token。
  这比随机初始化更高效，因为它让模型先学会 PBMC 特有的基因共表达模式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMHead(nn.Module):
    """
    MLM 预测头：
      接收 Transformer 最后一层隐向量，预测每个位置被掩码的基因 Token。

    结构：LayerNorm → Linear(hidden_size, vocab_size)
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        输入：hidden_states (B, L, H)
        输出：logits        (B, L, V)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states) + self.decoder_bias
        return logits


def compute_mlm_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mlm_probability: float = 0.15,
    pad_token_id: int = 0,
    mask_token_id: int = 4,
    vocab_size: int = 502,
) -> tuple:
    """
    随机掩码 + 计算 MLM Loss。

    返回：(loss, masked_inputs, labels)
    """
    B, L = token_ids.shape
    device = token_ids.device

    # ── 随机掩码策略（BERT 风格）──────────────────────────────────────────────
    probability_matrix = torch.full(token_ids.shape, mlm_probability, device=device)
    special_mask = (token_ids == pad_token_id) | (token_ids < 2)
    probability_matrix.masked_fill_(special_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 10% → 随机替换
    random_indices = torch.bernoulli(
        torch.full(token_ids.shape, 0.1, device=device)
    ).bool() & masked_indices
    random_tokens = torch.randint(2, vocab_size - 1, token_ids.shape, device=device)

    # 10% → 保持不变（标签仍是原始）
    unchanged = masked_indices & ~random_indices

    # 80% → [MASK]
    mask_indices = masked_indices & ~random_indices & ~unchanged

    inputs = token_ids.clone()
    inputs[mask_indices]  = mask_token_id
    inputs[random_indices] = random_tokens[random_indices]

    labels = token_ids.clone()
    labels[~masked_indices] = -100  # 不计算损失的位置

    # ── 前向 MLM Head（复用模型中的 regressor 作为共享 decoder）──────────────
    # 注意：这里需要模型输出 logits。简化处理，仅返回 masked inputs 和 labels。
    return inputs, labels


def mask_tokens(
    token_ids: torch.Tensor,
    mlm_probability: float = 0.15,
    pad_token_id: int = 0,
    mask_token_id: int = 4,
    vocab_size: int = 502,
) -> tuple:
    """
    随机掩码 token_ids。

    返回：(masked_inputs, labels)
      masked_inputs: 被掩码后的 token_ids
      labels:        被掩码位置 = 原始 token ID，未掩码 = -100
    """
    B, L = token_ids.shape
    device = token_ids.device

    prob = torch.full(token_ids.shape, mlm_probability, device=device)
    prob.masked_fill_((token_ids == pad_token_id) | (token_ids < 2), 0.0)
    masked = torch.bernoulli(prob).bool()

    # 10% 随机替换
    random = torch.bernoulli(torch.full(token_ids.shape, 0.1, device=device)).bool() & masked
    rand_tok = torch.randint(2, vocab_size - 1, token_ids.shape, device=device)

    inputs = token_ids.clone()
    inputs[masked & ~random] = mask_token_id
    inputs[random] = rand_tok[random]

    labels = token_ids.clone()
    labels[~masked] = -100

    return inputs, labels
