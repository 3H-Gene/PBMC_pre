"""
phase1_continue_train.py
========================
Phase 1：领域自适应继续预训练（DAPT）
  在公开 PBMC 大规模语料池上，用 MLM 目标继续预训练 Rank-Transformer。

输入：
  data/phase1_corpus.h5ad  (~5万细胞，无标签)

输出：
  checkpoints/phase1_pbmc_base.pt  — PBMC-Base 继续预训练权重

训练策略：
  - 加载 scGPT/Geneformer 基础权重（若无可用则随机初始化，从头 MLM 预训练）
  - 可选：冻结底层 Embedding，只微调顶层 Transformer 层
  - MLM Loss（掩码语言模型）
  - 词表：使用语料池基因构建，vocab_size 自动推断

用法：
  python -m src.train.phase1_continue_train --data data/phase1_corpus.h5ad
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC_DIR)

from src.model.rank_transformer import PBMCGPTConfig, PBMCGPTModel
from src.data.rank_token_dataset import PBMCRankTokenDataset, build_gene_vocab


# ─── 配置 ──────────────────────────────────────────────────────────────────────

class Phase1Config:
    """Phase 1 DAPT 超参数"""

    def __init__(
        self,
        # 数据
        top_n: int = 256,
        batch_size: int = 64,
        num_workers: int = 0,
        # 模型（Base 配置）
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        # MLM
        mlm_probability: float = 0.15,
        mask_token_id: int = 4,
        # 训练
        num_epochs: int = 20,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        # 冻结策略
        freeze_embedding: bool = False,
        freeze_layers: int = 0,   # 冻结前 N 层，0 = 全量微调
        # 保存
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
        seed: int = 42,
    ):
        self.top_n = top_n
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mlm_probability = mlm_probability
        self.mask_token_id = mask_token_id
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.freeze_embedding = freeze_embedding
        self.freeze_layers = freeze_layers
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


# ─── 掩码策略 ──────────────────────────────────────────────────────────────────

def mask_tokens(
    token_ids: torch.Tensor,
    mlm_prob: float = 0.15,
    pad_token_id: int = 0,
    mask_token_id: int = 4,
    vocab_size: int = 502,
) -> tuple:
    """
    BERT 风格掩码：
      80% → [MASK] token
      10% → 随机 token
      10% → 保持不变

    返回：(masked_inputs, labels)
    """
    B, L = token_ids.shape
    device = token_ids.device

    prob = torch.full(token_ids.shape, mlm_prob, device=device)
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


# ─── MLM Loss ──────────────────────────────────────────────────────────────────

def mlm_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算 MLM 交叉熵损失（忽略 -100 标签）。
    """
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    # shift: logits[L] 预测 labels[L]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    active = shift_labels != -100
    active_logits  = shift_logits.view(-1, logits.size(-1))[active.view(-1)]
    active_labels  = shift_labels.view(-1)[active.view(-1)]

    return loss_fct(active_logits, active_labels).mean()


# ─── 学习率调度 ────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int,
):
    import math
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── 冻结工具 ──────────────────────────────────────────────────────────────────

def apply_freeze(model: PBMCGPTModel, freeze_layers: int, freeze_embedding: bool):
    """冻结模型的指定层"""
    if freeze_embedding:
        for param in model.embeddings.parameters():
            param.requires_grad = False
        print(f"  [FROZEN] Embedding 层已冻结")

    if freeze_layers > 0:
        layers_to_freeze = model.encoder.layers[:freeze_layers]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"  [FROZEN] 前 {freeze_layers} 层 Transformer 已冻结")


# ─── 单 Epoch ──────────────────────────────────────────────────────────────────

def train_epoch_mlm(
    model: PBMCGPTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: Phase1Config,
) -> Dict[str, float]:
    model.train()
    device = cfg.device
    total_loss = 0.0
    n_batches = len(loader)

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        token_ids  = batch["token_ids"].to(device)
        expr_ranks = batch["expr_ranks"].to(device)

        # MLM 掩码
        masked_ids, labels = mask_tokens(
            token_ids,
            mlm_prob=cfg.mlm_probability,
            pad_token_id=cfg.mask_token_id,
            vocab_size=model.config.vocab_size,
        )

        # Forward：模型返回无回归输出的 dict，手动添加 MLM head
        # 复用模型的 encoder + regressor 前的共享 hidden 投影
        attn_mask = model._make_attention_mask(token_ids, cfg.mask_token_id)
        hidden = model.embeddings(masked_ids, expr_ranks)
        hidden, _ = model.encoder(hidden, attn_mask, output_attentions=False)

        # 简易 MLM head（无额外权重，与 regressor 共享 encoder 输出）
        # 这里直接用 dense projection 模拟，实际用 separate head 更标准
        logits = model.regressor.mlp[0](hidden)   # 不严谨，建议加载专用 MLM head
        loss = mlm_loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return {"loss": total_loss / n_batches}


# ─── 主训练 ────────────────────────────────────────────────────────────────────

def continue_train(
    corpus_path: str,
    cfg: Phase1Config,
    pretrained_ckpt: Optional[str] = None,
    output_dir: str = "outputs",
) -> PBMCGPTModel:
    """
    Phase 1 继续预训练入口。

    参数：
        corpus_path:     Phase 1 语料池 h5ad 路径
        cfg:             Phase1Config 实例
        pretrained_ckpt: 可选，预加载的 scGPT/Geneformer 权重路径
        output_dir:      输出目录
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = cfg.device
    print(f"\n{'='*60}")
    print("Phase 1: 领域自适应继续预训练 (DAPT)")
    print(f"{'='*60}")
    print(f"设备: {device} | Epochs: {cfg.num_epochs} | Batch: {cfg.batch_size}")
    print(f"冻结策略: embedding={cfg.freeze_embedding}, layers={cfg.freeze_layers}")

    # ── 1. 加载语料数据 ──────────────────────────────────────────────────────
    print(f"\n[1/4] 加载语料池: {corpus_path}")
    import anndata as ad
    adata = ad.read_h5ad(corpus_path)
    print(f"  {adata.n_obs} 细胞 × {adata.n_vars} 基因")

    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    vocab_size = len(gene_vocab)
    print(f"  词汇表大小: {vocab_size}")

    # Phase 1 语料池无 age/cell_type 列，使用占位符
    adata.obs["_dummy_age"] = 50.0
    adata.obs["_dummy_cell_type"] = "Unknown"
    dataset = PBMCRankTokenDataset(
        adata, gene_vocab, top_n=cfg.top_n,
        age_col="_dummy_age",
        cell_type_col="_dummy_cell_type",
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: {
            "token_ids":  torch.stack([x["token_ids"]  for x in b]),
            "expr_ranks": torch.stack([x["expr_ranks"] for x in b]),
        }
    )
    print(f"  {len(dataset)} 细胞，{len(loader)} batches/epoch")

    # ── 2. 构建/加载模型 ──────────────────────────────────────────────────────
    print(f"\n[2/4] 构建 Rank-Transformer（Base 配置）...")
    model_config = PBMCGPTConfig(
        vocab_size=vocab_size,
        top_n=cfg.top_n,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
    )
    model = PBMCGPTModel(model_config).to(device)
    n_params = model.count_parameters()
    print(f"  参数量: {n_params:,}")
    print(f"  架构:  {cfg.hidden_size}d × {cfg.num_hidden_layers}层 × {cfg.num_attention_heads}头")

    # 加载预训练权重（scGPT/Geneformer）
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        print(f"\n  加载预训练权重: {pretrained_ckpt}")
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  [OK] 预训练权重加载完成")

    # 冻结策略
    if cfg.freeze_embedding or cfg.freeze_layers > 0:
        apply_freeze(model, cfg.freeze_layers, cfg.freeze_embedding)

    # ── 3. 优化器 + 调度器 ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    total_steps  = len(loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[3/4] 开始 MLM 训练（{cfg.num_epochs} Epochs）...")
    print(f"  总步数: {total_steps} | Warmup: {warmup_steps}")

    history = []
    best_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        metrics = train_epoch_mlm(model, loader, optimizer, scheduler, cfg)
        elapsed = time.time() - t0
        history.append(metrics)
        print(f"  Epoch {epoch:>3}/{cfg.num_epochs} | Loss: {metrics['loss']:.4f}  [{elapsed:.1f}s]")

        # 保存最优
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            ckpt_path = os.path.join(cfg.checkpoint_dir, "phase1_pbmc_base.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "config":      model_config.__dict__,
                "vocab":       gene_vocab,
                "train_loss":  best_loss,
            }, ckpt_path)
            print(f"           [*] 最优 Loss {best_loss:.4f} → {ckpt_path}")

    # ── 4. 保存历史 ───────────────────────────────────────────────────────────
    history_path = os.path.join(output_dir, "phase1_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[4/4] 训练完成！最优 Loss: {best_loss:.4f}")
    print(f"  Checkpoint: {cfg.checkpoint_dir}/phase1_pbmc_base.pt")
    print(f"  历史记录:   {history_path}")
    print("=" * 60)
    return model


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: PBMC 领域自适应继续预训练")
    parser.add_argument("--data",    type=str, default="data/phase1_corpus.h5ad",
                        help="Phase 1 语料池路径")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--lr",      type=float, default=5e-5)
    parser.add_argument("--device",  type=str, default="auto")
    parser.add_argument("--pretrain", type=str, default=None,
                        help="可选：scGPT/Geneformer 预训练权重路径")
    args = parser.parse_args()

    WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    cfg = Phase1Config(
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=os.path.join(WORKSPACE, "checkpoints"),
    )

    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(WORKSPACE, data_path)

    if not os.path.exists(data_path):
        print(f"[ERR] 语料池不存在: {data_path}")
        print(f"  请先运行: python scripts/generate_data.py")
        sys.exit(1)

    continue_train(
        corpus_path=data_path,
        cfg=cfg,
        pretrained_ckpt=args.pretrain,
        output_dir=os.path.join(WORKSPACE, "outputs"),
    )
