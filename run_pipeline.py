"""
run_pipeline.py
===============
PBMC-GPT 端到端集成测试 Pipeline

按 planv0.1.md Phase 0 → Phase 2 → Phase 3 顺序依次执行：

  Step 1: 生成模拟 h5ad 数据（若不存在）
  Step 2: CellTypist 注释 + In-silico Bootstrapping
  Step 3: 验证 Rank-Token DataLoader
  Step 4: 验证模型 Forward/Backward Pass
  Step 5: 监督训练（Phase 2 微调）
  Step 6: N=1 推理 + XAI + 生成 PoC 报告（Phase 3）
"""

import os
import sys
import time

WORKSPACE = os.path.abspath(os.path.dirname(__file__))
SRC_DIR   = os.path.join(WORKSPACE, "src")
sys.path.insert(0, SRC_DIR)

import torch
import anndata as ad
import numpy as np

from data.simulate_data       import generate_baseline_data, generate_intervention_data
from data.bootstrapping        import CellTypistAnnotator, InSilicoBootstrapper
from data.rank_token_dataset   import build_gene_vocab, build_dataloaders
from model.rank_transformer    import PBMCGPTConfig, PBMCGPTModel
from train.train_aging         import TrainingConfig, train
from inference.inference_xai   import run_inference_pipeline


DATA_DIR   = os.path.join(WORKSPACE, "data", "simulated")
CKPT_DIR   = os.path.join(WORKSPACE, "checkpoints")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def step1_simulate_data():
    banner("Step 1: 生成模拟 h5ad 数据")
    baseline_path = os.path.join(DATA_DIR, "baseline_all.h5ad")
    pre_path  = os.path.join(DATA_DIR, "DONOR_INTERVENTION_Pre.h5ad")
    post_path = os.path.join(DATA_DIR, "DONOR_INTERVENTION_Post.h5ad")

    if os.path.exists(baseline_path) and os.path.exists(pre_path):
        print("  [OK] 模拟数据已存在，跳过生成。")
        return

    print("  正在生成基线数据（5 名供体）...")
    generate_baseline_data(DATA_DIR)
    print("  正在生成干预数据（N=1 Pre/Post）...")
    generate_intervention_data(DATA_DIR)
    print("  [OK] 模拟数据生成完毕。")


def step2_annotate_and_bootstrap():
    banner("Step 2: CellTypist 注释 + In-silico Bootstrapping")
    aug_path = os.path.join(DATA_DIR, "baseline_augmented.h5ad")

    if os.path.exists(aug_path):
        print("  [OK] 扩增数据已存在，跳过。")
    else:
        adata = ad.read_h5ad(os.path.join(DATA_DIR, "baseline_all.h5ad"))
        annotator   = CellTypistAnnotator(noise_level=0.1)
        bootstrapper = InSilicoBootstrapper(seed=42)

        adata = annotator.annotate(adata, force=True)
        adata_aug = bootstrapper.bootstrap_dataset(adata, k_per_donor=3, include_original=True)
        adata_aug.write_h5ad(aug_path)
        print(f"  [OK] 扩增数据保存 → {aug_path}")

    # 干预数据
    inter_dir = os.path.join(DATA_DIR, "intervention_pseudos")
    if os.path.exists(inter_dir) and len(os.listdir(inter_dir)) >= 10:
        print("  [OK] 干预伪样本已存在，跳过。")
    else:
        adata_pre  = ad.read_h5ad(os.path.join(DATA_DIR, "DONOR_INTERVENTION_Pre.h5ad"))
        adata_post = ad.read_h5ad(os.path.join(DATA_DIR, "DONOR_INTERVENTION_Post.h5ad"))
        annotator = CellTypistAnnotator(noise_level=0.1, seed=99)
        annotator.annotate(adata_pre,  force=True)
        annotator.annotate(adata_post, force=True)

        bootstrapper = InSilicoBootstrapper(seed=99)
        pre_list, post_list = bootstrapper.bootstrap_intervention(adata_pre, adata_post, k_samples=5)

        os.makedirs(inter_dir, exist_ok=True)
        for i, ps in enumerate(pre_list):
            ps.write_h5ad(os.path.join(inter_dir, f"pre_ps{i}.h5ad"))
        for i, ps in enumerate(post_list):
            ps.write_h5ad(os.path.join(inter_dir, f"post_ps{i}.h5ad"))
        print(f"  [OK] 干预伪样本（各5份）保存 → {inter_dir}")


def step3_validate_dataloader():
    banner("Step 3: 验证 Rank-Token DataLoader")
    aug_path  = os.path.join(DATA_DIR, "baseline_augmented.h5ad")
    base_path = os.path.join(DATA_DIR, "baseline_all.h5ad")
    data_path = aug_path if os.path.exists(aug_path) else base_path

    adata      = ad.read_h5ad(data_path)
    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    train_loader, val_loader, split_info = build_dataloaders(
        adata, gene_vocab, top_n=256, batch_size=32
    )
    batch = next(iter(train_loader))
    print(f"  token_ids 形状:  {batch['token_ids'].shape}")
    print(f"  expr_ranks 形状: {batch['expr_ranks'].shape}")
    print(f"  age 范围:        [{batch['age'].min():.1f}, {batch['age'].max():.1f}]")
    print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    print("  [OK] DataLoader 验证通过。")


def step4_validate_model():
    banner("Step 4: 模型 Forward/Backward Pass 验证")
    config = PBMCGPTConfig.dummy(vocab_size=502)
    model  = PBMCGPTModel(config)
    print(f"  参数量: {model.count_parameters():,}")

    B, L = 4, 256
    token_ids  = torch.randint(2, 502, (B, L))
    expr_ranks = torch.rand(B, L)
    ages       = torch.tensor([32.0, 45.0, 58.0, 67.0])
    loss_fn    = torch.nn.HuberLoss(delta=5.0)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        out  = model(token_ids, expr_ranks)
        loss = loss_fn(out["age_pred"], ages)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Loss 序列: {[f'{l:.4f}' for l in losses]}")
    print(f"  age_pred 输出: {out['age_pred'].detach().tolist()}")
    print("  [OK] Forward/Backward Pass 验证通过。")


def step5_train():
    banner("Step 5: Phase 2 监督训练（Huber Loss + 衰老年龄回归）")
    aug_path  = os.path.join(DATA_DIR, "baseline_augmented.h5ad")
    base_path = os.path.join(DATA_DIR, "baseline_all.h5ad")
    data_path = aug_path if os.path.exists(aug_path) else base_path

    cfg = TrainingConfig(
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        grad_accum_steps=2,
        checkpoint_dir=CKPT_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    adata      = ad.read_h5ad(data_path)
    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    vocab_size = len(gene_vocab)
    model_cfg  = PBMCGPTConfig.dummy(vocab_size=vocab_size)

    model, history = train(
        adata_path=data_path,
        cfg=cfg,
        model_config=model_cfg,
        output_dir=OUTPUT_DIR,
    )
    print("  [OK] 训练完成。")
    return history


def step6_inference():
    banner("Step 6: Phase 3 — N=1 推理 + XAI + PoC 报告")
    inter_dir   = os.path.join(DATA_DIR, "intervention_pseudos")
    ckpt_path   = os.path.join(CKPT_DIR, "pbmc_aging_predictor_best.pt")

    results = run_inference_pipeline(
        pre_pseudo_dir  = inter_dir,
        post_pseudo_dir = inter_dir,
        checkpoint_path = ckpt_path,
        output_dir      = OUTPUT_DIR,
        device          = "cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"  ΔAge = {results['delta_result']['delta_age']:+.2f} 岁")
    print(f"  报告 → {results['report_path']}")
    print("  [OK] Phase 3 完成。")
    return results


# ─── 主入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    print("\n" + "=" * 60)
    print("   PBMC-GPT 端到端集成 Pipeline 启动")
    print("=" * 60)

    step1_simulate_data()
    step2_annotate_and_bootstrap()
    step3_validate_dataloader()
    step4_validate_model()
    history = step5_train()
    results = step6_inference()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全部 Pipeline 完成！总耗时: {elapsed:.1f}s")
    print(f"  PoC 报告路径: {results['report_path']}")
    print(f"{'='*60}\n")
