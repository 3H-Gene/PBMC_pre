"""
train_aging.py
==============
Phase 2：免疫衰老年龄预测器的监督微调训练循环

功能：
  - 加载基线 h5ad（或扩增后数据），构建 Rank-Token DataLoader
  - 以 Donor ID 为单位进行严格 8:2 切分（防数据泄露）
  - 使用 Huber Loss 训练，MAE 评估
  - 支持梯度累加（Gradient Accumulation），在小显存下模拟大 Batch
  - 每 Epoch 保存最优 Checkpoint（Val MAE 最低）
  - 输出训练曲线数据（JSON）供后续可视化
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

# 添加 src 到路径
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from data.rank_token_dataset import (
    PBMCRankTokenDataset, build_gene_vocab, build_dataloaders, collate_fn
)
from model.rank_transformer import PBMCGPTConfig, PBMCGPTModel

import anndata as ad


# ─── 训练配置 ─────────────────────────────────────────────────────────────────

class TrainingConfig:
    """训练超参数"""

    def __init__(
        self,
        # 数据
        top_n: int = 256,
        batch_size: int = 32,
        test_ratio: float = 0.2,
        num_workers: int = 0,
        # 训练
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
        grad_accum_steps: int = 4,      # 梯度累加步数
        max_grad_norm: float = 1.0,
        huber_delta: float = 5.0,       # Huber Loss delta（岁）
        # 保存
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
        # 设备
        device: str = "auto",
        seed: int = 42,
    ):
        self.top_n = top_n
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.huber_delta = huber_delta
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.seed = seed

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


# ─── 学习率调度：线性 Warmup + Cosine Decay ───────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算 MAE、RMSE、R²"""
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ─── 单 Epoch 训练 ────────────────────────────────────────────────────────────

def train_epoch(
    model: PBMCGPTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.HuberLoss,
    cfg: TrainingConfig,
) -> Dict[str, float]:
    model.train()
    device = cfg.device
    total_loss = 0.0
    all_preds, all_targets = [], []
    n_steps = len(loader)

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        token_ids  = batch["token_ids"].to(device)
        expr_ranks = batch["expr_ranks"].to(device)
        ages       = batch["age"].to(device)

        outputs  = model(token_ids, expr_ranks)
        loss     = loss_fn(outputs["age_pred"], ages)

        # 梯度累加
        loss = loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == n_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.grad_accum_steps
        all_preds.extend(outputs["age_pred"].detach().cpu().numpy().tolist())
        all_targets.extend(ages.cpu().numpy().tolist())

    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    metrics["loss"] = total_loss / n_steps
    return metrics


# ─── 验证 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: PBMCGPTModel,
    loader: DataLoader,
    loss_fn: nn.HuberLoss,
    cfg: TrainingConfig,
) -> Dict[str, float]:
    model.eval()
    device = cfg.device
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        token_ids  = batch["token_ids"].to(device)
        expr_ranks = batch["expr_ranks"].to(device)
        ages       = batch["age"].to(device)

        outputs = model(token_ids, expr_ranks)
        loss    = loss_fn(outputs["age_pred"], ages)

        total_loss += loss.item()
        all_preds.extend(outputs["age_pred"].cpu().numpy().tolist())
        all_targets.extend(ages.cpu().numpy().tolist())

    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ─── 主训练流程 ───────────────────────────────────────────────────────────────

def train(
    adata_path: str,
    cfg: TrainingConfig,
    model_config: Optional[PBMCGPTConfig] = None,
    output_dir: str = "outputs",
) -> Tuple[PBMCGPTModel, Dict]:
    """
    完整训练流程入口。

    返回：(最优模型, 训练历史)
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = cfg.device
    print(f"\n{'='*60}")
    print(f"PBMC-GPT 训练启动")
    print(f"{'='*60}")
    print(f"设备: {device} | Epochs: {cfg.num_epochs} | Batch: {cfg.batch_size}")

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    print(f"\n[1/4] 加载数据: {adata_path}")
    adata = ad.read_h5ad(adata_path)
    print(f"  {adata.n_obs} 细胞 × {adata.n_vars} 基因")

    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    vocab_size = len(gene_vocab)
    print(f"  词汇表大小: {vocab_size}")

    train_loader, val_loader, split_info = build_dataloaders(
        adata,
        gene_vocab,
        top_n=cfg.top_n,
        test_ratio=cfg.test_ratio,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    print(f"  Train: {split_info['train_cells']} 细胞 ({len(train_loader)} batches)")
    print(f"  Val:   {split_info['test_cells']}  细胞 ({len(val_loader)} batches)")

    # ── 2. 构建模型 ──────────────────────────────────────────────────────────
    print(f"\n[2/4] 构建模型...")
    if model_config is None:
        model_config = PBMCGPTConfig.dummy(vocab_size=vocab_size)
    model_config.vocab_size = vocab_size
    model = PBMCGPTModel(model_config).to(device)
    print(f"  参数量: {model.count_parameters():,}")
    print(f"  隐层:  {model_config.hidden_size}d × {model_config.num_hidden_layers}层")

    # ── 3. 优化器 + 调度器 ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps   = (len(train_loader) // cfg.grad_accum_steps) * cfg.num_epochs
    warmup_steps  = int(total_steps * cfg.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn       = nn.HuberLoss(delta=cfg.huber_delta)

    print(f"\n[3/4] 开始训练（{cfg.num_epochs} Epochs）...")
    print(f"  总步数: {total_steps} | Warmup: {warmup_steps} | 梯度累加: {cfg.grad_accum_steps}")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train MAE':>9} | "
          f"{'Val Loss':>8} | {'Val MAE':>7} | {'Val R²':>6}")
    print("  " + "-" * 62)

    history = {"train": [], "val": []}
    best_val_mae = float("inf")
    best_ckpt_path = None

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, cfg)
        val_metrics   = evaluate(model, val_loader, loss_fn, cfg)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        elapsed = time.time() - t0
        print(f"  {epoch:>5} | {train_metrics['loss']:>10.4f} | "
              f"{train_metrics['mae']:>9.2f} | "
              f"{val_metrics['loss']:>8.4f} | "
              f"{val_metrics['mae']:>7.2f} | "
              f"{val_metrics['r2']:>6.3f}  [{elapsed:.1f}s]")

        # 保存最优 Checkpoint
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_ckpt_path = os.path.join(
                cfg.checkpoint_dir, "pbmc_aging_predictor_best.pt"
            )
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_mae":     best_val_mae,
                "config":      model_config.__dict__,
                "split_info":  split_info,
            }, best_ckpt_path)

    print(f"\n[4/4] 训练完成！最优 Val MAE: {best_val_mae:.2f} 岁")
    print(f"  最优 Checkpoint: {best_ckpt_path}")

    # 保存训练历史
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  训练历史: {history_path}")

    print("=" * 60)
    return model, history


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    WORKSPACE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    # 优先使用扩增数据；如无则用原始基线
    aug_path  = os.path.join(WORKSPACE, "data/simulated/baseline_augmented.h5ad")
    base_path = os.path.join(WORKSPACE, "data/simulated/baseline_all.h5ad")
    data_path = aug_path if os.path.exists(aug_path) else base_path

    if not os.path.exists(data_path):
        print("[WARN]️  未找到训练数据，请先运行 simulate_data.py 和 bootstrapping.py")
        sys.exit(1)

    cfg = TrainingConfig(
        num_epochs=15,
        batch_size=32,
        learning_rate=1e-4,
        grad_accum_steps=4,
        checkpoint_dir=os.path.join(WORKSPACE, "checkpoints"),
    )

    model, history = train(
        adata_path=data_path,
        cfg=cfg,
        output_dir=os.path.join(WORKSPACE, "outputs"),
    )
