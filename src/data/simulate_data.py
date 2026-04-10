"""
simulate_data.py
================
生成模拟 PBMC scRNA-seq h5ad 数据，用于 Phase 0 MVP 验证。

模拟逻辑：
- 5 个基线供体（Donor 01-05），每人 300-500 个细胞，覆盖 5 种 PBMC 细胞类型
- 1 个 MSC 干预供体（Donor_Intervention），包含 Pre/Post 两个时间点，各 400 个细胞
- 基因集：500 个模拟 Ensembl ID（ENSG00000000001 ~ ENSG00000000500）
- 年龄：随机生成 30-75 岁，性别随机
- 细胞类型：CD4+ T, CD8+ T, B Cell, NK Cell, Monocyte
- 干预效应：Post 样本整体"逆龄" -5 岁（通过细胞表达模式偏移实现）
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import os

# ─── 常量 ─────────────────────────────────────────────────────────────────────
SEED = 42
N_GENES = 500
CELL_TYPES = ["CD4+ T", "CD8+ T", "B Cell", "NK Cell", "Monocyte"]
N_BASELINE_DONORS = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/simulated")

# 每种细胞类型的特征基因索引范围（用于生成差异表达模式）
CELL_TYPE_MARKER_RANGES = {
    "CD4+ T":    (0,   80),
    "CD8+ T":    (80,  160),
    "B Cell":    (160, 250),
    "NK Cell":   (250, 350),
    "Monocyte":  (350, 500),
}


def make_ensembl_ids(n: int) -> list:
    """生成模拟 Ensembl Gene ID 列表"""
    return [f"ENSG{str(i).zfill(11)}" for i in range(1, n + 1)]


def simulate_cell_expression(
    cell_type: str,
    n_cells: int,
    donor_age: float,
    rng: np.random.Generator,
    age_noise_scale: float = 0.1,
) -> np.ndarray:
    """
    生成单个细胞类型的表达矩阵（原始 counts，整数型）。
    年龄越大，整体表达量轻微下降，同时特定基因噪音增大（模拟衰老）。
    """
    counts = np.zeros((n_cells, N_GENES), dtype=np.float32)
    start, end = CELL_TYPE_MARKER_RANGES[cell_type]

    # 细胞类型特异性高表达区域
    base_expr = rng.exponential(scale=3.0, size=(n_cells, end - start))
    # 衰老效应：年龄越大，marker 基因表达下调 + 噪声增大
    age_factor = 1.0 - (donor_age - 30) / 100 * age_noise_scale * 5
    counts[:, start:end] = base_expr * age_factor

    # 背景低表达
    background = rng.exponential(scale=0.3, size=(n_cells, N_GENES))
    counts += background

    # 转为整数原始计数（模拟 UMI counts）
    counts = np.round(counts).astype(np.int32)
    counts = np.clip(counts, 0, None)
    return counts


def create_donor_adata(
    donor_id: str,
    age: float,
    sex: str,
    n_cells: int,
    rng: np.random.Generator,
    intervention_timepoint: str = None,
    age_shift: float = 0.0,
) -> ad.AnnData:
    """
    创建单个供体的 AnnData 对象。
    intervention_timepoint: None | "Pre" | "Post"
    age_shift: Post 样本的逆龄偏移（如 -5）
    """
    # 随机分配细胞类型
    cell_type_probs = [0.30, 0.25, 0.20, 0.15, 0.10]
    cell_type_labels = rng.choice(CELL_TYPES, size=n_cells, p=cell_type_probs)

    # 按细胞类型生成表达矩阵
    all_counts = []
    for ct in CELL_TYPES:
        mask = cell_type_labels == ct
        n = mask.sum()
        if n == 0:
            continue
        expr = simulate_cell_expression(ct, n, age + age_shift, rng)
        all_counts.append(expr)

    X = np.vstack(all_counts).astype(np.float32)

    # 重排细胞顺序（打乱，对应 cell_type_labels 排序后的顺序）
    sorted_types = np.concatenate([
        np.full(int((cell_type_labels == ct).sum()), ct)
        for ct in CELL_TYPES
        if (cell_type_labels == ct).sum() > 0
    ])

    gene_names = make_ensembl_ids(N_GENES)
    cell_ids = [f"{donor_id}_cell_{i:04d}" for i in range(len(sorted_types))]

    obs = pd.DataFrame({
        "cell_id": cell_ids,
        "donor_id": donor_id,
        "age": age,
        "sex": sex,
        "cell_type": sorted_types,
    }, index=cell_ids)

    if intervention_timepoint:
        obs["timepoint"] = intervention_timepoint

    var = pd.DataFrame({"gene_id": gene_names}, index=gene_names)

    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.uns["donor_id"] = donor_id
    adata.uns["age"] = age
    adata.uns["sex"] = sex
    return adata


def generate_baseline_data(output_dir: str, seed: int = SEED):
    """
    生成 5 个基线供体的 h5ad 文件（含年龄标签），
    同时生成合并的 baseline_all.h5ad。
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    ages = [32, 45, 58, 67, 72]
    sexes = ["M", "F", "M", "F", "M"]
    adatas = []

    for i in range(N_BASELINE_DONORS):
        donor_id = f"DONOR_{i+1:02d}"
        n_cells = int(rng.integers(300, 501))
        adata = create_donor_adata(
            donor_id=donor_id,
            age=ages[i],
            sex=sexes[i],
            n_cells=n_cells,
            rng=rng,
        )
        path = os.path.join(output_dir, f"{donor_id}.h5ad")
        adata.write_h5ad(path)
        adatas.append(adata)
        print(f"  [OK] 基线供体 {donor_id}: 年龄={ages[i]}, 性别={sexes[i]}, "
              f"细胞数={adata.n_obs}, 写入 → {path}")

    # 合并为单个文件
    combined = ad.concat(adatas, join="outer", label="donor_id", keys=[
        f"DONOR_{i+1:02d}" for i in range(N_BASELINE_DONORS)
    ])
    combined_path = os.path.join(output_dir, "baseline_all.h5ad")
    combined.write_h5ad(combined_path)
    print(f"\n  [PKG] 合并基线数据 → {combined_path}")
    print(f"     总细胞数: {combined.n_obs}, 基因数: {combined.n_vars}")
    return adatas, combined


def generate_intervention_data(output_dir: str, seed: int = SEED + 100):
    """
    生成 1 个 MSC 干预供体的 Pre/Post h5ad 文件。
    Post 样本模拟逆龄效应（-5 岁等效表达偏移）。
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    donor_id = "DONOR_INTERVENTION"
    base_age = 65
    sex = "F"
    n_cells = 400

    # Pre 样本
    adata_pre = create_donor_adata(
        donor_id=f"{donor_id}_Pre",
        age=base_age,
        sex=sex,
        n_cells=n_cells,
        rng=rng,
        intervention_timepoint="Pre",
        age_shift=0.0,
    )
    pre_path = os.path.join(output_dir, f"{donor_id}_Pre.h5ad")
    adata_pre.write_h5ad(pre_path)
    print(f"  [OK] 干预供体 Pre: 年龄={base_age}, 细胞数={adata_pre.n_obs} → {pre_path}")

    # Post 样本（模拟逆龄 -5 岁）
    adata_post = create_donor_adata(
        donor_id=f"{donor_id}_Post",
        age=base_age,
        sex=sex,
        n_cells=n_cells,
        rng=rng,
        intervention_timepoint="Post",
        age_shift=-5.0,  # 逆龄效应
    )
    post_path = os.path.join(output_dir, f"{donor_id}_Post.h5ad")
    adata_post.write_h5ad(post_path)
    print(f"  [OK] 干预供体 Post: 年龄={base_age}（逆龄-5岁效应）, "
          f"细胞数={adata_post.n_obs} → {post_path}")

    return adata_pre, adata_post


if __name__ == "__main__":
    abs_output = os.path.abspath(OUTPUT_DIR)
    print("=" * 60)
    print("PBMC-GPT 模拟数据生成器")
    print("=" * 60)

    print("\n[1/2] 生成基线数据（5 名供体）...")
    baseline_adatas, baseline_combined = generate_baseline_data(abs_output)

    print("\n[2/2] 生成干预数据（N=1, Pre/Post）...")
    adata_pre, adata_post = generate_intervention_data(abs_output)

    print("\n" + "=" * 60)
    print("[OK] 所有模拟数据生成完毕！")
    print(f"   输出目录: {abs_output}")
    print("=" * 60)
