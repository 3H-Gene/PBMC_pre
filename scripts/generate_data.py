"""
generate_data.py
================
一次性数据模拟脚本，仅运行一次，生成三阶段所需 h5ad 数据集。
运行后 data/ 目录（除 .gitkeep 外）建议加入 .gitignore，不推送到 GitHub。

生成数据集：
  Phase 1 语料池  → data/phase1_corpus.h5ad        (~5万细胞，无标签)
  Phase 2 训练池  → data/phase2_baseline_400.h5ad  (400人，含年龄/性别)
  Phase 3 推理池  → data/phase3_n1_pre.h5ad        (N=1 干预前)
                 → data/phase3_n1_post.h5ad       (N=1 干预后)

使用真实数据时替换此脚本即可。
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from datetime import datetime

# ─── 全局配置 ──────────────────────────────────────────────────────────────────

N_PHASE1_CELLS  = 50_000   # Phase 1 无标签 PBMC 语料池细胞数
N_PHASE2_DONORS = 400      # Phase 2 有标签供体数
N_PHASE3_CELLS  = 500      # Phase 3 N=1 单个供体细胞数（Pre / Post 各一份）
N_GENES         = 500      # 基因数（真实数据用 ~20000，这里用模拟的 500）
RANDOM_SEED     = 42

# 5种细胞类型（与 bootstrapping.py 中的 CELL_TYPE_MARKER_RANGES 对应）
CELL_TYPES = ["CD4+ T", "CD8+ T", "B Cell", "NK Cell", "Monocyte"]
CELL_TYPE_PROPORTIONS = [0.30, 0.20, 0.15, 0.15, 0.20]  # 近似 PBMC 真实分布

# marker 基因索引区间（用于模拟细胞类型特征）
MARKER_RANGES = {
    "CD4+ T":   (0,   80),
    "CD8+ T":   (80,  160),
    "B Cell":   (160, 250),
    "NK Cell":  (250, 350),
    "Monocyte": (350, 500),
}

np.random.seed(RANDOM_SEED)


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def make_gene_names(n_genes: int) -> list:
    """生成模拟 Ensembl 基因 ID"""
    return [f"ENSG{str(i).zfill(11)}" for i in range(1, n_genes + 1)]


def simulate_cell_expression(
    n_cells: int,
    n_genes: int,
    age: float = 50.0,
    sex: str = "M",
    noise_scale: float = 1.0,
    age_effect_scale: float = 0.5,
) -> np.ndarray:
    """
    模拟单个细胞的表达向量。
    age_effect_scale > 0：年龄越大，衰老 marker 基因表达越高。
    """
    X = np.random.randn(n_cells, n_genes).astype(np.float32) * noise_scale

    # 全局背景：均值偏移模拟批次效应
    X += np.random.randn() * 0.3

    for cell_idx in range(n_cells):
        # 随机分配细胞类型
        cell_type = np.random.choice(CELL_TYPES, p=CELL_TYPE_PROPORTIONS)
        s, e = MARKER_RANGES[cell_type]
        X[cell_idx, s:e] += np.abs(np.random.randn(e - s)).astype(np.float32) * 1.5

        # 年龄效应：衰老 marker 基因（使用 CD8+ T / Monocyte 的部分基因）
        aging_genes = list(range(80, 120)) + list(range(350, 390))
        aging_genes = [g for g in aging_genes if g < n_genes]
        age_signal = (age - 50.0) * age_effect_scale * np.random.rand(len(aging_genes))
        X[cell_idx, aging_genes] += age_signal.astype(np.float32)

    X = np.clip(X, 0, None)  # counts 不能为负
    return X


def simulate_donor(
    donor_id: str,
    n_cells: int,
    age: float,
    sex: str,
    base_date: str = "2023-01-01",
) -> ad.AnnData:
    """模拟单个供体的 AnnData"""
    X = simulate_cell_expression(n_cells, N_GENES, age=age, sex=sex)
    var = pd.DataFrame(index=make_gene_names(N_GENES))

    obs = pd.DataFrame({
        "donor_id":       [donor_id] * n_cells,
        "age":            [age] * n_cells,
        "sex":            [sex] * n_cells,
        "cell_type":      ["Unknown"] * n_cells,   # 后续由 bootstrapping.py 注释
        "n_genes_by_counts": (X > 0).sum(axis=1),
        "total_counts":       X.sum(axis=1),
        "sampling_date":   [base_date] * n_cells,
    }, index=[f"{donor_id}_cell_{i:04d}" for i in range(n_cells)])

    return ad.AnnData(sp.csr_matrix(X), obs=obs, var=var)


def simulate_phase1_corpus(output_path: str) -> None:
    """Phase 1：PBMC 语料池（无标签，用于继续预训练 DAPT）"""
    print(f"\n{'='*60}")
    print("生成 Phase 1 语料池...")
    print(f"{'='*60}")
    print(f"  细胞数: {N_PHASE1_CELLS:,}")

    # 模拟 200 个匿名供体（无年龄标签）
    n_anon_donors = 200
    cells_per_donor = N_PHASE1_CELLS // n_anon_donors
    all_adatas = []

    for i in range(n_anon_donors):
        donor_id = f"CORPUS_D{i:04d}"
        a = simulate_cell_expression(cells_per_donor, N_GENES, age=50.0, sex="M")
        var = pd.DataFrame(index=make_gene_names(N_GENES))
        obs = pd.DataFrame({
            "donor_id": [donor_id] * cells_per_donor,
            "sex":      [np.random.choice(["M", "F"])] * cells_per_donor,
            "n_genes_by_counts": (a > 0).sum(axis=1),
            "total_counts":     a.sum(axis=1),
        }, index=[f"{donor_id}_cell_{j:04d}" for j in range(cells_per_donor)])
        all_adatas.append(ad.AnnData(sp.csr_matrix(a), obs=obs, var=var))

    adata = ad.concat(all_adatas, join="outer")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"  [OK] 保存 → {output_path}")
    print(f"  {adata.n_obs} 细胞 × {adata.n_vars} 基因")


def simulate_phase2_baseline(output_path: str) -> None:
    """Phase 2：400 人有标签基线数据（用于监督微调）"""
    print(f"\n{'='*60}")
    print("生成 Phase 2 基线训练池...")
    print(f"{'='*60}")
    print(f"  供体数: {N_PHASE2_DONORS}")

    all_adatas = []
    ages = np.linspace(20, 80, N_PHASE2_DONORS)  # 20-80 岁均匀分布
    rng = np.random.default_rng(RANDOM_SEED)

    for i in range(N_PHASE2_DONORS):
        donor_id = f"DONOR_{i:04d}"
        age = float(ages[i]) + rng.normal(0, 2)  # 加少量噪声
        sex = rng.choice(["M", "F"])
        n_cells = rng.integers(300, 600)          # 每供体 300-600 细胞

        # 年龄越大，表达噪声越大（模拟免疫衰老异质性）
        noise_scale = 1.0 + (age - 50) / 100
        X = simulate_cell_expression(n_cells, N_GENES, age=age, sex=sex,
                                     noise_scale=noise_scale)

        var = pd.DataFrame(index=make_gene_names(N_GENES))
        obs = pd.DataFrame({
            "donor_id":          [donor_id] * n_cells,
            "age":               [round(age, 1)] * n_cells,
            "sex":               [sex] * n_cells,
            "cell_type":         ["Unknown"] * n_cells,
            "n_genes_by_counts": (X > 0).sum(axis=1),
            "total_counts":      X.sum(axis=1),
        }, index=[f"{donor_id}_cell_{j:04d}" for j in range(n_cells)])

        all_adatas.append(ad.AnnData(sp.csr_matrix(X), obs=obs, var=var))

    adata = ad.concat(all_adatas, join="outer")
    adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
    adata.obs["sex"]      = adata.obs["sex"].astype(str)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"  [OK] 保存 → {output_path}")
    print(f"  {adata.n_obs} 细胞 × {adata.n_vars} 基因")
    print(f"  年龄分布: {adata.obs['age'].min():.1f} ~ {adata.obs['age'].max():.1f} 岁")


def simulate_phase3_individual(output_dir: str) -> None:
    """Phase 3：N=1 干预数据（Pre + Post，各一份）"""
    print(f"\n{'='*60}")
    print("生成 Phase 3 N=1 干预数据...")
    print(f"{'='*60}")

    donor_id  = "MSC_PATIENT_001"
    base_age  = 65.0

    os.makedirs(output_dir, exist_ok=True)

    # Pre 干预：真实生物学年龄
    adata_pre = simulate_donor(
        donor_id=f"{donor_id}_Pre",
        n_cells=N_PHASE3_CELLS,
        age=base_age,
        sex="M",
    )
    adata_pre.write_h5ad(os.path.join(output_dir, "phase3_n1_pre.h5ad"))
    print(f"  [Pre]  {adata_pre.n_obs} 细胞 | 保存 → {output_dir}/phase3_n1_pre.h5ad")

    # Post 干预：模拟 MSC 逆龄效应（年龄感知下降 ~5岁）
    adata_post = simulate_donor(
        donor_id=f"{donor_id}_Post",
        n_cells=N_PHASE3_CELLS,
        age=base_age - 5.0,   # 模拟逆龄
        sex="M",
    )
    adata_post.write_h5ad(os.path.join(output_dir, "phase3_n1_post.h5ad"))
    print(f"  [Post] {adata_post.n_obs} 细胞 | 保存 → {output_dir}/phase3_n1_post.h5ad")


# ─── 主入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    WORKSPACE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    DATA_DIR = os.path.join(WORKSPACE, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print(f"PBMC-GPT 数据模拟脚本")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)

    # Phase 1: PBMC 语料池（继续预训练）
    simulate_phase1_corpus(os.path.join(DATA_DIR, "phase1_corpus.h5ad"))

    # Phase 2: 400 人基线（有标签）
    simulate_phase2_baseline(os.path.join(DATA_DIR, "phase2_baseline_400.h5ad"))

    # Phase 3: N=1 干预数据
    simulate_phase3_individual(DATA_DIR)

    print(f"\n{'='*60}")
    print("[OK] 所有数据集生成完毕！")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  建议：将 data/ 加入 .gitignore，不推送至 GitHub")
    print("=" * 60)
