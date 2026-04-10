"""
phase3_paired_delta.py
=====================
Phase 3b：Pre/Post 配对 ΔAge 分析 + 多次干预轨迹追踪

输入：
  data/phase3_n1_pre.h5ad   — 干预前单样本
  data/phase3_n1_post.h5ad  — 干预后单样本（可选第2/3次随访）
  checkpoints/phase2_pbmc_age.pt  — PBMC-Age 模型权重

输出：
  outputs/phase3/delta_report.md   — 配对分析 Markdown 报告
  outputs/phase3/delta_results.json — ΔAge 数值结果

分析内容：
  1. Pre/Post 各预测年龄（细胞中位数点估计）
  2. ΔAge = Post_pred - Pre_pred 及配对 T-test 显著性检验
  3. 多次随访轨迹（ΔAge1, ΔAge2, ΔAge3...）可视化描述
  4. Pre/Post Attention XAI 对比（基因层面解读 MSC 逆龄机制）

用法：
  python -m src.inference.phase3_paired_delta \\
      --pre data/phase3_n1_pre.h5ad \\
      --post data/phase3_n1_post.h5ad

  # 多次随访：
  python -m src.inference.phase3_paired_delta \\
      --pre data/phase3_n1_pre.h5ad \\
      --post data/phase3_n1_post.h5ad \\
      --post2 data/phase3_n1_post2.h5ad
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
from scipy import stats
from typing import List, Dict, Optional
from datetime import datetime

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC_DIR)

from src.model.rank_transformer import PBMCGPTConfig, PBMCGPTModel
from src.data.rank_token_dataset import build_gene_vocab, cell_to_rank_tokens


# ─── 推理工具（复用 phase3_predict.py）────────────────────────────────────────

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


def predict_single_median(
    adata_path: str,
    model: PBMCGPTModel,
    gene_vocab: dict,
    top_n: int,
    device: str,
) -> Dict:
    """推理单个 h5ad，返回预测年龄（中位数）和 Attention。"""
    adata = ad.read_h5ad(adata_path)
    result = predict_adata(adata, model, gene_vocab, top_n, device=device)
    return {
        "path":        adata_path,
        "n_cells":     int(adata.n_obs),
        "median_age":  round(float(np.median(result["age_preds"])), 2),
        "mean_age":    round(float(np.mean(result["age_preds"])), 2),
        "std_age":     round(float(np.std(result["age_preds"])), 2),
        "age_preds":   result["age_preds"].tolist(),
        "cell_types":  result["cell_types"].tolist(),
        "attentions":  result["attentions"],
    }


# ─── Attention Top 基因提取 ────────────────────────────────────────────────────

def extract_top_genes(
    result: Dict,
    gene_vocab: dict,
    top_k: int = 20,
    cell_type_filter: Optional[str] = None,
) -> pd.DataFrame:
    attn   = result.get("attentions")
    ctypes = result.get("cell_types")
    if attn is None or ctypes is None:
        return pd.DataFrame()

    id2gene = {v: k for k, v in gene_vocab.items() if v >= 2}

    if cell_type_filter:
        mask = np.array(ctypes) == cell_type_filter
        if mask.sum() == 0:
            return pd.DataFrame()
        attn = attn[mask]

    cls_attn = attn[:, :, 0, 1:].mean(axis=(0, 1))
    top_idx  = np.argsort(cls_attn)[::-1][:top_k]

    gene_names = []
    for pos in top_idx:
        mode_tok = int(np.median(attn[:, 0, 0, pos + 1].astype(float)).round())
        gene_names.append(id2gene.get(mode_tok, f"Token_{mode_tok}"))

    return pd.DataFrame({
        "rank_position":   top_idx + 1,
        "gene_id":         gene_names,
        "attention_score": np.round(cls_attn[top_idx], 4),
    })


# ─── ΔAge 计算 ─────────────────────────────────────────────────────────────────

def compute_delta_age(
    pre_result: Dict,
    post_results: List[Dict],
    paired: bool = True,
) -> Dict:
    """
    计算 ΔAge = mean(Post_medians) - Pre_median
    paired: True = 配对 T-test，False = 独立样本 T-test
    """
    pre_median  = pre_result["median_age"]

    post_medians = [r["median_age"] for r in post_results]

    delta_ages = [post - pre_median for post in post_medians]
    delta_mean  = float(np.mean(delta_ages))

    if paired and len(post_results) >= 2:
        tstat, pvalue = stats.ttest_rel(post_medians, [pre_median] * len(post_medians))
        test_type = "paired_ttest"
    elif not paired:
        tstat, pvalue = stats.ttest_ind(post_medians, [pre_median] * len(post_medians))
        test_type = "independent_ttest"
    else:
        tstat, pvalue = np.nan, np.nan
        test_type = "single_post_no_test"

    direction = "逆龄" if delta_mean < 0 else "无逆龄效应"
    significance = "显著" if (not np.isnan(pvalue) and pvalue < 0.05) else "不显著"

    return {
        "pre_median":        pre_median,
        "post_medians":      post_medians,
        "delta_ages":         delta_ages,
        "delta_mean":         round(delta_mean, 2),
        "delta_std":         round(float(np.std(delta_ages)), 2) if len(delta_ages) > 1 else 0.0,
        "test_type":         test_type,
        "t_statistic":       round(float(tstat), 4) if not np.isnan(tstat) else None,
        "p_value":           round(float(pvalue), 4) if not np.isnan(pvalue) else None,
        "significant":       not np.isnan(pvalue) and float(pvalue) < 0.05,
        "significance":      significance,
        "direction":         direction,
        "conclusion":        f"{significance}，{direction}（{delta_mean:+.2f}岁）",
    }


# ─── 生成 Markdown 报告 ────────────────────────────────────────────────────────

def generate_delta_report(
    delta_result: Dict,
    top_genes_pre:  pd.DataFrame,
    top_genes_post: pd.DataFrame,
    top_genes_cd8:  pd.DataFrame,
    output_path: str,
    donor_id: str = "MSC_PATIENT_001",
    intervention: str = "MSC",
) -> None:
    sig = "**显著** (p < 0.05)" if delta_result["significant"] else "不显著 (p >= 0.05)"
    delta = delta_result["delta_mean"]

    def df_md(df: pd.DataFrame, n: int = 10) -> str:
        return df.head(n).to_markdown(index=False) if not df.empty else "_（数据不足）_"

    # 多次随访轨迹表格
    trajectory_rows = ""
    for i, (post_med, d_age) in enumerate(
        zip(delta_result["post_medians"], delta_result["delta_ages"]), 1
    ):
        trajectory_rows += f"| Visit {i} | {post_med:.2f} | {d_age:+.2f} |\n"

    report = f"""# MSC 干预疗效评估报告 — ΔAge 配对分析

**供体 ID：** {donor_id}
**干预方式：** {intervention}
**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、ΔAge 疗效评估

| 指标 | 数值 |
|------|------|
| Pre 干预预测年龄（中位数） | {delta_result['pre_median']:.2f} 岁 |
| Post 干预预测年龄（中位数列表） | {delta_result['post_medians']} |
| **平均 ΔAge（Post − Pre）** | **{delta:+.2f} 岁** |
| ΔAge 标准差 | {delta_result['delta_std']:.2f} 岁 |
| 检验方法 | {delta_result['test_type']} |
| t 统计量 | {delta_result.get('t_statistic', 'N/A')} |
| p-value | {delta_result.get('p_value', 'N/A')} |
| 统计显著性 | {sig} |
| **结论** | {delta_result['conclusion']} |

### 多次随访轨迹

| 随访 | Post 预测年龄 | ΔAge（vs Pre） |
|------|------------|---------------|
{trajectory_rows}

> **解读指南：**
> - ΔAge < 0 → 预测年龄下降，提示逆龄效应
> - ΔAge > 0 → 预测年龄上升，提示加速衰老或无效
> - p < 0.05 → 干预效应具有统计学显著性

---

## 二、Attention XAI 机制解析

### 2.1 Pre 干预 Top-20 注意力基因

> 这些基因在 MSC 干预**前**对免疫年龄预测贡献最大（[CLS] Token 注意力权重均值）。

{df_md(top_genes_pre)}

### 2.2 Post 干预 Top-20 注意力基因

> 这些基因在 MSC 干预**后**对免疫年龄预测贡献最大。
> 与 Pre 对比：权重变化的基因暗示 MSC 直接调控的靶点。

{df_md(top_genes_post)}

### 2.3 CD8+ T 细胞亚群 Post 注意力基因

> CD8+ T 细胞是衰老免疫的核心效应亚群，以下基因代表 MSC 在 T 细胞层面的潜在调控靶点。

{df_md(top_genes_cd8)}

---

## 三、Pre / Post 预测年龄分布

| 指标 | Pre | Post |
|------|-----|------|
| 细胞数 | {len(top_genes_pre) * 0:.0f} | {len(top_genes_post) * 0:.0f} |
| 预测年龄（中位数） | {delta_result['pre_median']:.2f} | {np.mean(delta_result['post_medians']):.2f} |
| ΔAge | — | {delta:+.2f} |

---

## 四、方法学说明

- **模型**：PBMC-GPT (Rank-Transformer)，Phase 2 微调版 PBMC-Age
- **年龄估计**：每份样本取细胞预测年龄中位数
- **统计检验**：配对 T-test（多次随访时）
- **XAI**：最后一层 Transformer [CLS] Token → 基因 Token 平均注意力权重

---

_本报告由 PBMC-GPT Phase 3 推理管线自动生成，仅供科研参考。_
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [DOC] ΔAge 报告已保存 -> {output_path}")


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def paired_delta_analysis(
    pre_path: str,
    post_paths: List[str],
    checkpoint_path: str,
    output_dir: str = "outputs/phase3",
    top_n: int = 256,
    device: str = "cpu",
    donor_id: str = "MSC_PATIENT_001",
    intervention: str = "MSC",
) -> Dict:
    """Pre/Post 配对 ΔAge 分析入口。"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Phase 3b: Pre/Post 配对 ΔAge 分析")
    print(f"{'='*60}")

    # 1. 加载模型
    print(f"\n[1/4] 加载模型: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("  [WARN] Checkpoint 不存在，使用随机初始化")
        config = PBMCGPTConfig.dummy()
        model  = PBMCGPTModel(config)
    else:
        ckpt   = torch.load(checkpoint_path, map_location="cpu")
        config = PBMCGPTConfig(**ckpt["config"])
        model  = PBMCGPTModel(config)
        model.load_state_dict(ckpt["model_state"])
        print(f"  [OK] 加载完成 | Val MAE: {ckpt.get('val_mae', '?'):.2f}")
    model.eval()

    # 2. 构建词汇表
    sample_ad = ad.read_h5ad(pre_path)
    gene_vocab = build_gene_vocab(sample_ad.var_names.tolist())

    # 3. 推理 Pre + Post
    print(f"\n[2/4] 推理 Pre + {len(post_paths)} Post 样本...")
    pre_result = predict_single_median(pre_path, model, gene_vocab, top_n, device)
    print(f"  Pre:  {pre_result['median_age']:.2f} 岁 ({pre_result['n_cells']} 细胞)")

    post_results = []
    for i, post_path in enumerate(post_paths):
        r = predict_single_median(post_path, model, gene_vocab, top_n, device)
        print(f"  Post{i+1}: {r['median_age']:.2f} 岁 ({r['n_cells']} 细胞)")
        post_results.append(r)

    # 4. ΔAge 计算
    print(f"\n[3/4] ΔAge 统计检验...")
    delta_result = compute_delta_age(pre_result, post_results)
    print(f"  ΔAge = {delta_result['delta_mean']:+.2f} 岁")
    print(f"  p-value = {delta_result.get('p_value', 'N/A')}  {delta_result['significance']}")

    # 5. XAI 基因对比
    print(f"\n[4/4] Attention XAI 对比...")
    top_genes_pre  = extract_top_genes(pre_result,  gene_vocab, top_k=20)
    top_genes_post = extract_top_genes(post_results[0], gene_vocab, top_k=20)
    top_genes_cd8  = extract_top_genes(post_results[0], gene_vocab, top_k=20,
                                       cell_type_filter="CD8+ T")

    # 6. 生成报告
    report_path = os.path.join(output_dir, "delta_report.md")
    generate_delta_report(
        delta_result, top_genes_pre, top_genes_post, top_genes_cd8,
        output_path=report_path,
        donor_id=donor_id, intervention=intervention,
    )

    # 保存数值结果
    json_path = os.path.join(output_dir, "delta_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(delta_result, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] ΔAge 数值结果 -> {json_path}")

    print(f"\n{'='*60}")
    print(f"[OK] 配对 ΔAge 分析完成！")
    print(f"  ΔAge = {delta_result['delta_mean']:+.2f} 岁 | {delta_result['conclusion']}")
    print(f"  报告: {report_path}")
    print("=" * 60)

    return {
        "delta_result":  delta_result,
        "pre_result":    {k: v for k, v in pre_result.items() if k != "attentions"},
        "post_results":  [{k: v for k, v in r.items() if k != "attentions"} for r in post_results],
        "report_path":   report_path,
    }


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b: Pre/Post 配对 DeltaAge 分析")
    parser.add_argument("--pre",          type=str, required=True)
    parser.add_argument("--post",         type=str, required=True)
    parser.add_argument("--post2",         type=str, default=None)
    parser.add_argument("--post3",         type=str, default=None)
    parser.add_argument("--checkpoint",   type=str, default="checkpoints/phase2_pbmc_age.pt")
    parser.add_argument("--output",       type=str, default="outputs/phase3")
    parser.add_argument("--donor-id",     type=str, default="MSC_PATIENT_001")
    parser.add_argument("--intervention", type=str, default="MSC")
    parser.add_argument("--top-n",       type=int, default=256)
    parser.add_argument("--device",      type=str, default="auto")
    args = parser.parse_args()

    WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    def abs_path(p): return p if os.path.isabs(p) else os.path.join(WORKSPACE, p)

    pre_paths = [abs_path(args.post)]
    for p in [args.post2, args.post3]:
        if p:
            pre_paths.append(abs_path(p))

    for p in [abs_path(args.pre)] + pre_paths:
        if not os.path.exists(p):
            print(f"[ERR] 文件不存在: {p}")
            sys.exit(1)

    output_dir = abs_path(args.output) if os.path.isabs(args.output) else os.path.join(WORKSPACE, args.output)

    paired_delta_analysis(
        pre_path=abs_path(args.pre),
        post_paths=pre_paths,
        checkpoint_path=abs_path(args.checkpoint) if os.path.isabs(args.checkpoint)
                        else os.path.join(WORKSPACE, args.checkpoint),
        output_dir=output_dir,
        top_n=args.top_n,
        device=device,
        donor_id=args.donor_id,
        intervention=args.intervention,
    )
