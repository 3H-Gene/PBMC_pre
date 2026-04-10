"""
rank_token_dataset.py
=====================
核心 DataLoader：实现 h5ad → Rank Token 的实时转换。

Rank Token 化逻辑（来自 planv0.1.md）：
  1. 提取 Unscaled Raw Counts（稀疏矩阵转 dense）
  2. 按表达量降序排列，截断 Top N 基因（默认 N=256）
  3. 映射至全局 Ensembl ID → 整数 Token ID 字典

Dataset 结构：
  - 每个样本 = 一个细胞
  - 输入：形状 (top_n,) 的 gene_token_ids（整数序列）
  - 标签：供体年龄（float，用于回归）
  - 辅助：donor_id, cell_type（用于数据切分与 XAI）
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Tuple

# ─── 全局 Ensembl ID → Token ID 词典 ─────────────────────────────────────────

def build_gene_vocab(gene_list: List[str], pad_token: str = "<PAD>",
                     unk_token: str = "<UNK>") -> Dict[str, int]:
    """
    构建基因词汇表：Ensembl ID → 整数 Token ID。
    索引 0 = PAD，索引 1 = UNK，基因从索引 2 开始。
    """
    vocab = {pad_token: 0, unk_token: 1}
    for gene in gene_list:
        if gene not in vocab:
            vocab[gene] = len(vocab)
    return vocab


# ─── Rank Token 转换工具函数 ──────────────────────────────────────────────────

def cell_to_rank_tokens(
    expr_vector: np.ndarray,
    gene_names: np.ndarray,
    gene_vocab: Dict[str, int],
    top_n: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将单个细胞的表达向量转换为 Rank Token 序列。

    参数：
        expr_vector: shape (n_genes,)，原始 counts（非归一化）
        gene_names:  shape (n_genes,)，对应基因 Ensembl ID
        gene_vocab:  Ensembl ID → Token ID 映射字典
        top_n:       截断保留的最高表达基因数

    返回：
        token_ids:  shape (top_n,)，按表达量降序排列的 Token ID（不足则 PAD=0）
        expr_ranks: shape (top_n,)，对应的标准化表达量（0-1 范围，可选特征）
    """
    # 1. 按表达量降序排序
    sorted_idx = np.argsort(expr_vector)[::-1]

    # 2. 截断 Top N
    top_idx = sorted_idx[:top_n]
    top_genes = gene_names[top_idx]
    top_expr = expr_vector[top_idx].astype(np.float32)

    # 3. 映射至 Token ID（未知基因映射为 UNK=1）
    token_ids = np.array(
        [gene_vocab.get(g, 1) for g in top_genes], dtype=np.int64
    )

    # 4. 对表达量做 min-max 标准化（归一化到 0-1）
    max_expr = top_expr.max()
    expr_ranks = top_expr / (max_expr + 1e-8)

    # 5. 如果细胞总基因数 < top_n，PAD 补齐
    actual_len = len(token_ids)
    if actual_len < top_n:
        pad_len = top_n - actual_len
        token_ids = np.concatenate([token_ids, np.zeros(pad_len, dtype=np.int64)])
        expr_ranks = np.concatenate([expr_ranks, np.zeros(pad_len, dtype=np.float32)])

    return token_ids, expr_ranks


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class PBMCRankTokenDataset(Dataset):
    """
    PBMC Rank-Token Dataset。

    支持：
    - 直接从 AnnData 对象构建
    - 按 donor_id 做严格切分（防数据泄露）
    - 可选：仅加载特定 donor_ids 的数据子集
    """

    def __init__(
        self,
        adata: ad.AnnData,
        gene_vocab: Dict[str, int],
        top_n: int = 256,
        donor_ids: Optional[List[str]] = None,
        age_col: str = "age",
        cell_type_col: str = "cell_type",
        donor_col: str = "donor_id",
    ):
        """
        参数：
            adata:         AnnData 对象（必须包含原始 counts）
            gene_vocab:    全局基因词汇表（由 build_gene_vocab 生成）
            top_n:         Rank Token 截断长度
            donor_ids:     若指定，则只加载这些供体的细胞（用于 Train/Test 切分）
            age_col:       obs 中年龄列名
            cell_type_col: obs 中细胞类型列名
            donor_col:     obs 中供体 ID 列名
        """
        self.gene_vocab = gene_vocab
        self.top_n = top_n
        self.gene_names = np.array(adata.var_names)

        # 按 donor_id 过滤
        if donor_ids is not None:
            mask = adata.obs[donor_col].isin(donor_ids)
            adata = adata[mask].copy()

        # 提取表达矩阵（dense，原始 counts）
        if sp.issparse(adata.X):
            self.X = adata.X.toarray().astype(np.float32)
        else:
            self.X = np.array(adata.X, dtype=np.float32)

        # 元数据
        self.ages = adata.obs[age_col].values.astype(np.float32)
        self.cell_types = adata.obs[cell_type_col].values
        self.donor_ids = adata.obs[donor_col].values
        self.cell_ids = adata.obs.index.values

        self.n_cells = len(self.ages)
        print(f"  Dataset 初始化：{self.n_cells} 个细胞，top_n={top_n}，"
              f"供体数={len(np.unique(self.donor_ids))}")

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        expr_vec = self.X[idx]
        token_ids, expr_ranks = cell_to_rank_tokens(
            expr_vec, self.gene_names, self.gene_vocab, self.top_n
        )
        return {
            "token_ids":   torch.tensor(token_ids,   dtype=torch.long),
            "expr_ranks":  torch.tensor(expr_ranks,  dtype=torch.float32),
            "age":         torch.tensor(self.ages[idx], dtype=torch.float32),
            "cell_type":   self.cell_types[idx],
            "donor_id":    self.donor_ids[idx],
        }


# ─── 供体级严格切分（防数据泄露）────────────────────────────────────────────

def donor_stratified_split(
    adata: ad.AnnData,
    test_ratio: float = 0.2,
    donor_col: str = "donor_id",
    age_col: str = "age",
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    以 Donor ID 为单位进行分层切分，防止同一人的细胞同时出现在 Train/Test 集。

    返回：(train_donor_ids, test_donor_ids)
    """
    # 每个供体取平均年龄用于分层
    donor_ages = (
        adata.obs.groupby(donor_col)[age_col]
        .mean()
        .reset_index()
        .sort_values(age_col)
    )
    donors = donor_ages[donor_col].values

    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(donors) * test_ratio))

    # 简单分层：按年龄排序后等间隔采样 test 集
    test_indices = np.linspace(0, len(donors) - 1, n_test, dtype=int)
    test_donors = donors[test_indices].tolist()
    train_donors = [d for d in donors if d not in test_donors]

    print(f"  数据切分：Train={len(train_donors)} 供体，Test={len(test_donors)} 供体")
    print(f"  Test 供体: {test_donors}")
    return train_donors, test_donors


# ─── 快速构建 DataLoader 的工厂函数 ──────────────────────────────────────────

def build_dataloaders(
    adata: ad.AnnData,
    gene_vocab: Dict[str, int],
    top_n: int = 256,
    test_ratio: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    构建训练/验证 DataLoader。

    返回：(train_loader, val_loader, split_info)
    """
    train_donors, test_donors = donor_stratified_split(
        adata, test_ratio=test_ratio, seed=seed
    )

    train_ds = PBMCRankTokenDataset(adata, gene_vocab, top_n, donor_ids=train_donors)
    val_ds   = PBMCRankTokenDataset(adata, gene_vocab, top_n, donor_ids=test_donors)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )

    split_info = {
        "train_donors": train_donors,
        "test_donors":  test_donors,
        "train_cells":  len(train_ds),
        "test_cells":   len(val_ds),
    }
    return train_loader, val_loader, split_info


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义 collate：合并数值字段为张量，保留字符串字段为列表。"""
    return {
        "token_ids":  torch.stack([b["token_ids"]  for b in batch]),
        "expr_ranks": torch.stack([b["expr_ranks"] for b in batch]),
        "age":        torch.stack([b["age"]         for b in batch]),
        "cell_type":  [b["cell_type"]  for b in batch],
        "donor_id":   [b["donor_id"]   for b in batch],
    }


# ─── 快速验证入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/simulated")
    )
    baseline_path = os.path.join(data_dir, "baseline_all.h5ad")

    if not os.path.exists(baseline_path):
        print("[WARN]️  未找到模拟数据，请先运行 simulate_data.py")
        sys.exit(1)

    print("=" * 60)
    print("Rank-Token DataLoader 验证")
    print("=" * 60)

    adata = ad.read_h5ad(baseline_path)
    print(f"\n加载数据: {adata.n_obs} 细胞 × {adata.n_vars} 基因")

    # 构建词汇表
    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    print(f"词汇表大小: {len(gene_vocab)}")

    # 构建 DataLoaders
    print("\n构建 DataLoaders（Top-256 Rank Tokens）...")
    train_loader, val_loader, split_info = build_dataloaders(
        adata, gene_vocab, top_n=256, batch_size=32
    )

    # 检查第一个 batch
    batch = next(iter(train_loader))
    print(f"\n第一个训练 Batch:")
    print(f"  token_ids 形状:  {batch['token_ids'].shape}   dtype={batch['token_ids'].dtype}")
    print(f"  expr_ranks 形状: {batch['expr_ranks'].shape}  dtype={batch['expr_ranks'].dtype}")
    print(f"  age 形状:        {batch['age'].shape}          dtype={batch['age'].dtype}")
    print(f"  age 范围:        [{batch['age'].min():.1f}, {batch['age'].max():.1f}]")
    print(f"  细胞类型样本:    {list(set(batch['cell_type']))[:3]}")
    print(f"  token_ids 示例:  {batch['token_ids'][0, :10].tolist()}")

    print(f"\n[OK] DataLoader 验证通过！")
    print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("=" * 60)
