"""
phase2_finetune_train.py
========================
Phase 2：任务特异性监督微调
  基于 Phase 1 产出的 PBMC-Base 权重，用 400 人带年龄/性别标签的基线数据
  微调为 PBMC-Age 临床年龄预测器。

输入：
  data/phase2_baseline_400.h5ad  (400人，含年龄/性别)
  checkpoints/phase1_pbmc_base.pt (Phase 1 PBMC-Base 权重，可选)

输出：
  checkpoints/phase2_pbmc_age.pt  — PBMC-Age 微调权重

关键设计：
  - 冻结 Embedding + 前 N 层（保守微调），只微调顶层 Transformer + MLP 回归头
  - 以 Donor ID 为单位进行 8:2 分层切分（防数据泄露）
  - Huber Loss（对离群年龄预测鲁棒）+ MAE 评估
  - 可选 In-silico Bootstrapping（运行 bootstrapping.py 预处理后生效）

用法：
  python -m src.train.phase2_finetune_train --data data/phase2_baseline_400.h5ad
  python -m src.train.phase2_finetune_train --data data/phase2_baseline_400.h5ad \\
      --pretrain checkpoints/phase1_pbmc_base.pt --freeze-embedding --epochs 30
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
from typing import Dict, Tuple, Optional

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC_DIR)

from src.model.rank_transformer import PBMCGPTConfig, PBMCGPTModel
from src.data.rank_token_dataset import (
    PBMCRankTokenDataset, build_gene_vocab,
    build_dataloaders, collate_fn, donor_stratified_split
)


# ─── 配置 ──────────────────────────────────────────────────────────────────────

class Phase2Config:
    """Phase 2 监督微调超参数"""

    def __init__(
        self,
        # 数据
        top_n: int = 256,
        batch_size: int = 32,
        test_ratio: float = 0.2,
        num_workers: int = 0,
        bootstrap: bool = False,       # 是否使用 Bootstrapping 扩增数据
        k_bootstrap: int = 3,          # 每供体伪样本数
        # 训练
        num_epochs: int = 30,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        grad_accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        huber_delta: float = 5.0,       # Huber Loss delta（岁）
        # 冻结策略（保守微调）
        freeze_embedding: bool = True,
        freeze_layers: int = 4,         # 冻结前 4 层
        # 保存
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
        seed: int = 42,
    ):
        self.top_n = top_n
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.bootstrap = bootstrap
        self.k_bootstrap = k_bootstrap
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.huber_delta = huber_delta
        self.freeze_embedding = freeze_embedding
        self.freeze_layers = freeze_layers
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


# ─── 学习率调度 ────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    import math
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── 冻结工具 ──────────────────────────────────────────────────────────────────

def apply_freeze(model: PBMCGPTModel, freeze_layers: int, freeze_embedding: bool):
    if freeze_embedding:
        for p in model.embeddings.parameters():
            p.requires_grad = False
        print(f"  [FROZEN] Embedding 层已冻结（仅微调顶层）")

    if freeze_layers > 0:
        for layer in model.encoder.layers[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        print(f"  [FROZEN] 前 {freeze_layers} 层 Transformer 已冻结")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数量: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


# ─── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mae   = float(np.mean(np.abs(preds - targets)))
    rmse  = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ─── 单 Epoch ──────────────────────────────────────────────────────────────────

def train_epoch(
    model, loader, optimizer, scheduler, loss_fn, cfg,
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

        outputs = model(token_ids, expr_ranks)
        loss    = loss_fn(outputs["age_pred"], ages)
        loss    = loss / cfg.grad_accum_steps
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


@torch.no_grad()
def evaluate(model, loader, loss_fn, cfg) -> Dict[str, float]:
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


# ─── 主训练 ────────────────────────────────────────────────────────────────────

def finetune_train(
    baseline_path: str,
    cfg: Phase2Config,
    pretrained_ckpt: Optional[str] = None,
    output_dir: str = "outputs",
) -> Tuple[PBMCGPTModel, Dict]:
    """
    Phase 2 监督微调入口。

    返回：(最优模型, 训练历史)
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = cfg.device
    print(f"\n{'='*60}")
    print("Phase 2: PBMC-Age 监督微调")
    print(f"{'='*60}")
    print(f"设备: {device} | Epochs: {cfg.num_epochs} | Batch: {cfg.batch_size}")
    print(f"冻结策略: embedding={cfg.freeze_embedding}, layers={cfg.freeze_layers}")

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    print(f"\n[1/4] 加载基线数据: {baseline_path}")
    import anndata as ad
    adata = ad.read_h5ad(baseline_path)
    print(f"  {adata.n_obs} 细胞 × {adata.n_vars} 基因，{adata.obs['donor_id'].nunique()} 供体")

    # 可选 Bootstrapping 扩增
    if cfg.bootstrap:
        sys.path.insert(0, os.path.join(SRC_DIR, "src", "data"))
        from bootstrapping import InSilicoBootstrapper
        boot = InSilicoBootstrapper(seed=cfg.seed)
        adata_aug = boot.bootstrap_dataset(
            adata, k_per_donor=cfg.k_bootstrap,
            sample_frac=0.8, include_original=True
        )
        aug_path = baseline_path.replace(".h5ad", "_augmented.h5ad")
        adata_aug.write_h5ad(aug_path)
        print(f"  [BOOT] 扩增后: {adata_aug.n_obs} 细胞 → {aug_path}")
        adata = adata_aug

    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    vocab_size = len(gene_vocab)
    print(f"  词汇表大小: {vocab_size}")

    train_loader, val_loader, split_info = build_dataloaders(
        adata, gene_vocab,
        top_n=cfg.top_n,
        test_ratio=cfg.test_ratio,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    print(f"  Train: {split_info['train_cells']} 细胞 | Val: {split_info['test_cells']} 细胞")

    # ── 2. 构建/加载模型 ──────────────────────────────────────────────────────
    print(f"\n[2/4] 构建模型（加载 Phase 1 权重）...")
    model_config = PBMCGPTConfig(
        vocab_size=vocab_size,
        top_n=cfg.top_n,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
    )

    # 优先加载 Phase 1 继续预训练权重
    phase1_ckpt = os.path.join(cfg.checkpoint_dir, "phase1_pbmc_base.pt")
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        ckpt_path = pretrained_ckpt
    elif os.path.exists(phase1_ckpt):
        ckpt_path = phase1_ckpt
    else:
        ckpt_path = None

    model = PBMCGPTModel(model_config).to(device)

    if ckpt_path:
        print(f"  加载 Phase 1 权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  [OK] Phase 1 权重加载完成")
    else:
        print(f"  [WARN] 未找到 Phase 1 权重，使用随机初始化继续训练（建议先完成 Phase 1）")

    # 冻结策略
    if cfg.freeze_embedding or cfg.freeze_layers > 0:
        apply_freeze(model, cfg.freeze_layers, cfg.freeze_embedding)

    # ── 3. 优化器 + 调度器 ───────────────────────────────────────────────────
    # 对 Embedding / 冻结层使用更小学习率
    embed_params = list(model.embeddings.parameters()) if not cfg.freeze_embedding else []
    frozen_params = []
    if cfg.freeze_layers > 0:
        frozen_params = [p for layer in model.encoder.layers[:cfg.freeze_layers] for p in layer.parameters()]
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    param_groups = []
    if embed_params:
        param_groups.append({"params": embed_params, "lr": cfg.learning_rate * 0.1})
    if trainable_params:
        param_groups.append({"params": trainable_params, "lr": cfg.learning_rate})
    optimizer = torch.optim.AdamW(param_groups if param_groups else model.parameters(),
                                   lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps   = (len(train_loader) // cfg.grad_accum_steps) * cfg.num_epochs
    warmup_steps  = int(total_steps * cfg.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn       = nn.HuberLoss(delta=cfg.huber_delta)

    print(f"\n[3/4] 开始监督微调（{cfg.num_epochs} Epochs）...")
    print(f"  总步数: {total_steps} | Warmup: {warmup_steps} | Grad Accum: {cfg.grad_accum_steps}")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train MAE':>9} | "
          f"{'Val Loss':>8} | {'Val MAE':>7} | {'Val R2':>6}")
    print("  " + "-" * 65)

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
            best_ckpt_path = os.path.join(cfg.checkpoint_dir, "phase2_pbmc_age.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "config":      model_config.__dict__,
                "split_info":  split_info,
                "val_mae":     best_val_mae,
                "val_r2":      val_metrics["r2"],
            }, best_ckpt_path)

    # ── 4. 保存历史 ───────────────────────────────────────────────────────────
    history_path = os.path.join(output_dir, "phase2_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n[4/4] 训练完成！最优 Val MAE: {best_val_mae:.2f} 岁 | Val R2: {val_metrics['r2']:.3f}")
    print(f"  Checkpoint: {best_ckpt_path}")
    print(f"  历史记录:   {history_path}")
    print("=" * 60)
    return model, history


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: PBMC-Age 监督微调")
    parser.add_argument("--data",      type=str, default="data/phase2_baseline_400.h5ad")
    parser.add_argument("--pretrain",  type=str, default=None,
                        help="Phase 1 PBMC-Base 权重路径（可选）")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch",     type=int, default=32)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--device",    type=str, default="auto")
    parser.add_argument("--no-freeze", action="store_true",
                        help="不禁用冻结，全量微调")
    parser.add_argument("--bootstrap",  action="store_true",
                        help="启用 In-silico Bootstrapping 数据扩增")
    args = parser.parse_args()

    WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(WORKSPACE, data_path)

    if not os.path.exists(data_path):
        print(f"[ERR] 基线数据不存在: {data_path}")
        print(f"  请先运行: python scripts/generate_data.py")
        sys.exit(1)

    cfg = Phase2Config(
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        freeze_embedding=not args.no_freeze,
        freeze_layers=4 if not args.no_freeze else 0,
        bootstrap=args.bootstrap,
        checkpoint_dir=os.path.join(WORKSPACE, "checkpoints"),
    )

    finetune_train(
        baseline_path=data_path,
        cfg=cfg,
        pretrained_ckpt=args.pretrain,
        output_dir=os.path.join(WORKSPACE, "outputs"),
    )
