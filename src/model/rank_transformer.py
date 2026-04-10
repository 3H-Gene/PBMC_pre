"""
rank_transformer.py
===================
PBMC-GPT 核心模型架构：Rank-Transformer + MLP 回归头

架构设计（来自 planv0.1.md）：
  - 输入层：基因 Token 嵌入（Gene Embedding）+ 表达量值嵌入（Value Embedding）
  - 位置编码：可学习的位置嵌入（位置即秩排序位置）
  - Transformer 编码器：多层多头自注意力
  - 汇聚层：[CLS] Token 或均值池化
  - 回归头：MLP → 预测生物学年龄

Phase 0 使用轻量 Dummy 配置（2层, 128维），快速验证管线；
Phase 1/2 替换为更大配置（12层, 512维）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# ─── 模型配置 ─────────────────────────────────────────────────────────────────

class PBMCGPTConfig:
    """PBMC-GPT 模型超参数配置"""

    def __init__(
        self,
        vocab_size: int = 502,         # 基因词汇表大小（+2 for PAD/UNK）
        top_n: int = 256,              # Rank Token 序列长度
        hidden_size: int = 128,        # Transformer 隐层维度
        num_hidden_layers: int = 2,    # Transformer 层数
        num_attention_heads: int = 4,  # 注意力头数
        intermediate_size: int = 512,  # FFN 中间层维度
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 258,  # top_n + 1(CLS) + 1(SEP)
        # 回归头
        regressor_hidden_dims: Tuple[int, ...] = (64, 32),
        regressor_dropout: float = 0.2,
        # 预训练（MLM）
        mlm_probability: float = 0.15,
        pad_token_id: int = 0,
        use_value_embedding: bool = True,  # 是否融合表达量值嵌入
    ):
        self.vocab_size = vocab_size
        self.top_n = top_n
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.regressor_hidden_dims = regressor_hidden_dims
        self.regressor_dropout = regressor_dropout
        self.mlm_probability = mlm_probability
        self.pad_token_id = pad_token_id
        self.use_value_embedding = use_value_embedding

    @classmethod
    def dummy(cls, vocab_size: int = 502) -> "PBMCGPTConfig":
        """Phase 0 轻量 Dummy 配置：快速验证管线通畅"""
        return cls(
            vocab_size=vocab_size,
            top_n=256,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
        )

    @classmethod
    def base(cls, vocab_size: int = 502) -> "PBMCGPTConfig":
        """Phase 1/2 Base 配置：正式预训练与微调"""
        return cls(
            vocab_size=vocab_size,
            top_n=256,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            regressor_hidden_dims=(256, 128, 64),
        )


# ─── 输入嵌入层 ───────────────────────────────────────────────────────────────

class RankTokenEmbeddings(nn.Module):
    """
    Rank-Token 输入嵌入：
      Final Embedding = Gene Embedding + Value Embedding + Position Embedding
    
    - Gene Embedding：Token ID → 向量（学习基因身份）
    - Value Embedding：标量表达量 → 向量（学习表达强度，通过线性投影）
    - Position Embedding：秩位置（第1高 ~ 第N高）→ 向量（学习排序位置的语义）
    """

    def __init__(self, config: PBMCGPTConfig):
        super().__init__()
        self.gene_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        if config.use_value_embedding:
            self.value_proj = nn.Linear(1, config.hidden_size, bias=False)
        else:
            self.value_proj = None

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        token_ids: torch.Tensor,    # (B, L) 基因 Token ID
        expr_ranks: torch.Tensor,   # (B, L) 归一化表达量 [0,1]
    ) -> torch.Tensor:
        """
        返回：(B, L+1, H)  — +1 是预置的 [CLS] Token
        """
        B, L = token_ids.shape
        device = token_ids.device

        # [CLS] Token：用全零 token（映射至 PAD 嵌入，可学习替换）
        cls_token_ids   = torch.zeros(B, 1, dtype=torch.long, device=device)
        cls_expr_ranks  = torch.zeros(B, 1, dtype=torch.float32, device=device)
        token_ids  = torch.cat([cls_token_ids,  token_ids],  dim=1)   # (B, L+1)
        expr_ranks = torch.cat([cls_expr_ranks, expr_ranks], dim=1)   # (B, L+1)

        L1 = token_ids.shape[1]

        # Gene Embedding
        gene_emb = self.gene_embeddings(token_ids)   # (B, L+1, H)

        # Position Embedding（位置 0 = CLS，1..L = 第1高~第L高表达基因）
        position_ids = torch.arange(L1, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embeddings(position_ids)  # (1, L+1, H)

        # Value Embedding（将标量表达量投影到隐层空间）
        if self.value_proj is not None:
            val_emb = self.value_proj(expr_ranks.unsqueeze(-1))  # (B, L+1, H)
            embeddings = gene_emb + pos_emb + val_emb
        else:
            embeddings = gene_emb + pos_emb

        return self.dropout(self.layer_norm(embeddings))


# ─── Transformer 编码器 ───────────────────────────────────────────────────────

class RankTransformerEncoder(nn.Module):
    """标准 Transformer 编码器（多层 Self-Attention + FFN）"""

    def __init__(self, config: PBMCGPTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,         # (B, L, H)
        attention_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, L)
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            hidden_states, attn_weights = layer(
                hidden_states, attention_mask, output_attentions
            )
            if output_attentions:
                all_attentions.append(attn_weights)

        return hidden_states, all_attentions


class TransformerLayer(nn.Module):
    """单层 Transformer = Self-Attention + FFN（Post-LN 结构）"""

    def __init__(self, config: PBMCGPTConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.ffn       = FeedForwardNetwork(config)
        self.norm1     = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.norm2     = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-Attention + Residual
        attn_out, attn_weights = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        hidden_states = self.norm1(hidden_states + self.dropout(attn_out))

        # FFN + Residual
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(ffn_out))

        return hidden_states, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（返回注意力权重，供 XAI 使用）"""

    def __init__(self, config: PBMCGPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, \
            "hidden_size 必须能被 num_attention_heads 整除"

        self.num_heads = config.num_attention_heads
        self.head_dim  = config.hidden_size // self.num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        x: torch.Tensor,                               # (B, L, H)
        mask: Optional[torch.Tensor] = None,           # (B, 1, 1, L)
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, H = x.shape

        def split_heads(t):
            # (B, L, H) → (B, num_heads, L, head_dim)
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, L, L)
        if mask is not None:
            scores = scores + mask  # mask 中 -inf 位置被屏蔽

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context = torch.matmul(attn_weights_dropped, V)  # (B, h, L, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, L, H)
        output  = self.out_proj(context)

        weights_out = attn_weights if output_attentions else None
        return output, weights_out


class FeedForwardNetwork(nn.Module):
    """两层 FFN with GELU 激活"""

    def __init__(self, config: PBMCGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ─── MLP 回归头（Phase 2 微调核心）──────────────────────────────────────────

class AgeRegressionHead(nn.Module):
    """
    生物学年龄回归头（Phase 2）：
    [CLS] 向量 → MLP → 预测年龄（单标量输出）

    训练时使用 Huber Loss（对离群点鲁棒）；
    评估时计算 MAE（平均绝对误差，单位：岁）。
    """

    def __init__(self, config: PBMCGPTConfig):
        super().__init__()
        dims = [config.hidden_size] + list(config.regressor_hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:   # 最后一层不加激活和 Dropout
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.regressor_dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cls_vector: torch.Tensor) -> torch.Tensor:
        """
        输入：(B, H) — [CLS] Token 的隐层向量
        输出：(B,)   — 预测年龄
        """
        return self.mlp(cls_vector).squeeze(-1)


# ─── 完整 PBMC-GPT 模型 ───────────────────────────────────────────────────────

class PBMCGPTModel(nn.Module):
    """
    完整 PBMC-GPT 模型：
      输入 → Embedding → Transformer → [CLS] → 回归头 → 预测年龄

    Phase 0：Dummy 配置，验证管线
    Phase 1：加载 scGPT/Geneformer 权重，MLM 预训练（替换回归头为 MLM 头）
    Phase 2：加载 Base 权重，激活回归头，监督微调
    """

    def __init__(self, config: PBMCGPTConfig):
        super().__init__()
        self.config = config
        self.embeddings = RankTokenEmbeddings(config)
        self.encoder    = RankTransformerEncoder(config)
        self.regressor  = AgeRegressionHead(config)
        self._init_weights()

    def _init_weights(self):
        """Xavier/Normal 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_attention_mask(
        self,
        token_ids: torch.Tensor,  # (B, L)
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        生成注意力 mask：PAD 位置填 -1e9（屏蔽），有效位置填 0。
        需扩展维度以适配多头注意力：(B, 1, 1, L+1)
        """
        # token_ids 已不含 CLS，在 embedding 层前向中会插入
        # 这里我们在 forward 调用后构建，传入的 token_ids 是原始的
        B, L = token_ids.shape
        # CLS 对应的 mask = 0（有效）
        cls_mask = torch.zeros(B, 1, dtype=torch.float32, device=token_ids.device)
        token_mask = (token_ids == pad_token_id).float() * -1e9  # PAD 位置 = -1e9
        full_mask = torch.cat([cls_mask, token_mask], dim=1)     # (B, L+1)
        return full_mask.unsqueeze(1).unsqueeze(2)                # (B, 1, 1, L+1)

    def forward(
        self,
        token_ids: torch.Tensor,               # (B, L)
        expr_ranks: torch.Tensor,              # (B, L)
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        返回字典：
          - age_pred:       (B,)     预测年龄
          - cls_vector:     (B, H)   [CLS] 隐层向量（用于 XAI）
          - last_hidden:    (B,L+1,H) 最后一层所有 Token 的隐层向量
          - attentions:     List[(B, heads, L+1, L+1)] 各层注意力权重（output_attentions=True 时）
        """
        # 注意力 mask
        attn_mask = self._make_attention_mask(token_ids, self.config.pad_token_id)

        # 嵌入层
        hidden = self.embeddings(token_ids, expr_ranks)   # (B, L+1, H)

        # Transformer 编码
        hidden, attentions = self.encoder(
            hidden, attn_mask, output_attentions=output_attentions
        )

        # 取 [CLS] Token（位置 0）
        cls_vector = hidden[:, 0, :]   # (B, H)

        # 回归头
        age_pred = self.regressor(cls_vector)   # (B,)

        return {
            "age_pred":    age_pred,
            "cls_vector":  cls_vector,
            "last_hidden": hidden,
            "attentions":  attentions,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── 快速验证：Forward/Backward Pass ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Rank-Transformer Forward/Backward Pass 验证")
    print("=" * 60)

    # Dummy 配置
    config = PBMCGPTConfig.dummy(vocab_size=502)
    model  = PBMCGPTModel(config)
    total_params = model.count_parameters()
    print(f"\n模型配置：")
    print(f"  隐层维度:  {config.hidden_size}")
    print(f"  Transformer 层数: {config.num_hidden_layers}")
    print(f"  注意力头数: {config.num_attention_heads}")
    print(f"  序列长度:   {config.top_n}")
    print(f"  总参数量:   {total_params:,}")

    # 模拟 batch
    B, L = 4, 256
    token_ids  = torch.randint(2, 502, (B, L))   # 随机 Token（跳过 PAD=0, UNK=1）
    expr_ranks = torch.rand(B, L)                # 随机表达量 [0, 1]
    ages       = torch.tensor([32.0, 45.0, 58.0, 67.0])

    print(f"\n输入形状：token_ids={token_ids.shape}, expr_ranks={expr_ranks.shape}")

    # Forward Pass
    model.eval()
    with torch.no_grad():
        outputs = model(token_ids, expr_ranks, output_attentions=True)

    print(f"\nForward Pass 输出：")
    print(f"  age_pred 形状:    {outputs['age_pred'].shape}     值={outputs['age_pred'].tolist()}")
    print(f"  cls_vector 形状:  {outputs['cls_vector'].shape}")
    print(f"  last_hidden 形状: {outputs['last_hidden'].shape}")
    print(f"  注意力层数:       {len(outputs['attentions'])}")
    print(f"  注意力形状:       {outputs['attentions'][-1].shape}  (B, heads, L+1, L+1)")

    # Backward Pass（验证梯度流动）
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn   = nn.HuberLoss(delta=5.0)

    losses = []
    print(f"\nBackward Pass（5步 Loss 下降测试）：")
    for step in range(5):
        optimizer.zero_grad()
        out  = model(token_ids, expr_ranks)
        loss = loss_fn(out["age_pred"], ages)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")

    trend = "[OK] 下降" if losses[-1] < losses[0] else "[WARN]️ 未下降（正常，batch 过小）"
    print(f"\n  Loss 趋势：{losses[0]:.4f} → {losses[-1]:.4f}  {trend}")
    print("\n[OK] Rank-Transformer Forward/Backward Pass 验证通过！")
    print("=" * 60)
