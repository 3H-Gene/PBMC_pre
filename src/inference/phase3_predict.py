"""
phase3_predict.py
=================
Phase 3a：单样本年龄预测 + 群体偏移可解释性分析

输入：
  data/phase3_n1_pre.h5ad  或任意单样本 h5ad
  checkpoints/phase2_pbmc_age.pt  — PBMC-Age 模型权重

输出：
  outputs/phase3/prediction_results.json
  outputs/phase3/attention_top_genes.csv

分析内容：
  1. 单样本年龄预测（年龄点估计）
  2. 计算相对于训练集年龄分布的偏移（DeltaAge vs 群体均值）
  3. Attention XAI：提取 Top 影响基因，按细胞类型聚合

用法：
  python -m src.inference.phase3_predict --data data/phase3_n1_pre.h5ad
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import anndata as ad
import scipy.sparse as sp
from typing import Dict, Optional

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC_DIR)

from src.model.rank_transformer import PBMCGPTConfig, PBMCGPTModel
from src.data.rank_token_dataset import build_gene_vocab, cell_to_rank_tokens


# ─── 推理 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_adata(
    adata: ad.AnnData,
    model: PBMCGPTModel,
    gene_vocab: dict,
    top_n: int = 256,
    batch_size: int = 64,
    device: str = "cpu",
    output_attentions: bool = True,
) -> Dict:
    """对单个 AnnData 中所有细胞推理。"""
    model.eval()
    model = model.to(device)
    gene_names = np.array(adata.var_names)

    if sp.issparse(adata.X):
        X = adata.X.toarray().astype(np.float32)
    else:
        X = np.array(adata.X, dtype=np.float32)

    n_cells = X.shape[0]
    all_preds, all_attn_last = [], []
    cell_types = adata.obs.get("cell_type", pd.Series(["Unknown"] * n_cells)).values

    for i in range(0, n_cells, batch_size):
        batch_X = X[i: i + batch_size]
        batch_tok, batch_rank = [], []

        for expr_vec in batch_X:
            tid, rank = cell_to_rank_tokens(expr_vec, gene_names, gene_vocab, top_n)
            batch_tok.append(tid)
            batch_rank.append(rank)

        tok_t  = torch.tensor(np.array(batch_tok),  dtype=torch.long).to(device)
        rank_t = torch.tensor(np.array(batch_rank), dtype=torch.float32).to(device)

        out = model(tok_t, rank_t, output_attentions=output_attentions)
        all_preds.extend(out["age_pred"].cpu().numpy().tolist())

        if output_attentions and out["attentions"]:
            all_attn_last.append(out["attentions"][-1].cpu().numpy())

    attn = np.concatenate(all_attn_last, axis=0) if all_attn_last else None
    return {
        "age_preds":   np.array(all_preds, dtype=np.float32),
        "cell_types":  cell_types,
        "attentions":  attn,
    }


# ─── 群体偏移计算 ─────────────────────────────────────────────────────────────

def compute_population_deviation(
    age_preds: np.ndarray,
    cell_types: np.ndarray,
    training_stats: Optional[Dict] = None,
) -> Dict:
    """
    计算样本预测年龄相对于群体分布的偏移。
    """
    pred_median = float(np.median(age_preds))
    pred_mean   = float(np.mean(age_preds))
    pred_std    = float(np.std(age_preds))

    # 按细胞类型分组
    type_stats = {}
    for ct in np.unique(cell_types):
        mask = cell_types == ct
        if mask.sum() > 0:
            ct_preds = age_preds[mask]
            type_stats[ct] = {
                "n_cells": int(mask.sum()),
                "median":  round(float(np.median(ct_preds)), 2),
                "mean":    round(float(np.mean(ct_preds)), 2),
                "std":     round(float(np.std(ct_preds)), 2),
                "range":   [round(float(ct_preds.min()), 2), round(float(ct_preds.max()), 2)],
            }

    pop_mean = (training_stats or {}).get("mean_age", 50.0)
    pop_std  = (training_stats or {}).get("std_age",  15.0)
    z_score  = (pred_median - pop_mean) / (pop_std + 1e-8)

    return {
        "sample": {
            "n_cells":    len(age_preds),
            "median_age": pred_median,
            "mean_age":   pred_mean,
            "std_age":    pred_std,
            "z_score":    round(z_score, 3),
            "deviation":  round(pred_median - pop_mean, 2),
        },
        "cell_type_breakdown": type_stats,
    }


# ─── Attention XAI ─────────────────────────────────────────────────────────────

def extract_attention_top_genes(
    results: Dict,
    gene_vocab: dict,
    top_k: int = 20,
    cell_type_filter: Optional[str] = None,
) -> pd.DataFrame:
    """从 Attention 权重中提取 Top 影响基因。"""
    attn   = results.get("attentions")
    ctypes = results.get("cell_types")
    if attn is None or ctypes is None:
        return pd.DataFrame()

    id2gene = {v: k for k, v in gene_vocab.items() if v >= 2}

    if cell_type_filter:
        mask = np.array(ctypes) == cell_type_filter
        if mask.sum() == 0:
            return pd.DataFrame()
        attn = attn[mask]

    # CLS 对其他 Token 的平均注意力：(L,)
    cls_attn = attn[:, :, 0, 1:].mean(axis=(0, 1))
    top_idx  = np.argsort(cls_attn)[::-1][:top_k]

    # 映射 Token ID → 基因名
    gene_names = []
    for pos in top_idx:
        mode_tok = int(np.median(attn[:, 0, 0, pos + 1].astype(float)).round())
        gene_names.append(id2gene.get(mode_tok, f"Token_{mode_tok}"))

    return pd.DataFrame({
        "rank_position":   top_idx + 1,
        "gene_id":         gene_names,
        "attention_score": np.round(cls_attn[top_idx], 4),
    })


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def predict_single_sample(
    adata_path: str,
    checkpoint_path: str,
    output_dir: str = "outputs/phase3",
    top_n: int = 256,
    device: str = "cpu",
    training_stats: Optional[Dict] = None,
) -> Dict:
    """单样本推理入口。"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Phase 3a: 单样本年龄预测 + 可解释性分析")
    print(f"{'='*60}")

    # 1. 加载模型
    print(f"\n[1/3] 加载模型: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("  [WARN] Checkpoint 不存在，使用随机初始化模型")
        config = PBMCGPTConfig.dummy()
        model  = PBMCGPTModel(config)
    else:
        ckpt   = torch.load(checkpoint_path, map_location="cpu")
        config = PBMCGPTConfig(**ckpt["config"])
        model  = PBMCGPTModel(config)
        model.load_state_dict(ckpt["model_state"])
        print(f"  [OK] 加载完成 | Val MAE: {ckpt.get('val_mae', '?'):.2f}")
    model.eval()

    # 2. 加载样本并推理
    print(f"\n[2/3] 推理样本: {adata_path}")
    adata = ad.read_h5ad(adata_path)
    print(f"  {adata.n_obs} 细胞 x {adata.n_vars} 基因")

    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    results    = predict_adata(adata, model, gene_vocab, top_n, device=device)
    print(f"  预测完成 | age_preds.shape={results['age_preds'].shape}")

    # 3. 分析
    print(f"\n[3/3] 计算群体偏移 + XAI...")
    deviation = compute_population_deviation(
        results["age_preds"], results["cell_types"], training_stats
    )
    print(f"  预测年龄（样本中位数）: {deviation['sample']['median_age']:.2f} 岁")
    print(f"  群体偏移 (z-score):   {deviation['sample']['z_score']:.3f}")
    print(f"  偏离群体均值:          {deviation['sample']['deviation']:+.2f} 岁")

    top_genes = extract_attention_top_genes(results, gene_vocab, top_k=20)
    if not top_genes.empty:
        genes_path = os.path.join(output_dir, "attention_top_genes.csv")
        top_genes.to_csv(genes_path, index=False)
        print(f"  Attention Top-20 基因: {genes_path}")

    # 4. 保存
    output = {
        "sample_path":       adata_path,
        "checkpoint_path":  checkpoint_path,
        "n_cells":          int(adata.n_obs),
        "prediction":       deviation["sample"],
        "cell_type_breakdown": deviation["cell_type_breakdown"],
    }
    results_path = os.path.join(output_dir, "prediction_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 推理完成 -> {results_path}")
    print("=" * 60)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3a: 单样本年龄预测")
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2_pbmc_age.pt")
    parser.add_argument("--output",     type=str, default="outputs/phase3")
    parser.add_argument("--top-n",     type=int, default=256)
    parser.add_argument("--device",    type=str, default="auto")
    args = parser.parse_args()

    WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_p   = args.data    if os.path.isabs(args.data)    else os.path.join(WORKSPACE, args.data)
    ckpt_p   = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(WORKSPACE, args.checkpoint)
    output_p = args.output  if os.path.isabs(args.output)  else os.path.join(WORKSPACE, args.output)
    device   = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    if not os.path.exists(data_p):
        print(f"[ERR] 样本数据不存在: {data_p}")
        sys.exit(1)

    predict_single_sample(
        adata_path=data_p,
        checkpoint_path=ckpt_p,
        output_dir=output_p,
        top_n=args.top_n,
        device=device,
    )
