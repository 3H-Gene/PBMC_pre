"""
bootstrapping.py
================
原位重抽样扩增（In-silico Bootstrapping）+ 模拟 CellTypist 细胞注释

功能：
1. CellTypistAnnotator：模拟细胞类型注释（当无真实 CellTypist 模型时，
   基于 marker 基因表达启发式打分，产出与 CellTypist 相同的输出格式）。
2. InSilicoBootstrapper：按供体拆分，随机重抽样生成 K 份伪样本（Pseudo-samples），
   严格保证同一供体的伪样本不会横跨 Train/Test 集。
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from typing import List, Dict, Tuple, Optional


# ─── 1. 模拟 CellTypist 细胞类型注释 ─────────────────────────────────────────

# 每种细胞类型对应的 marker 基因索引范围（与 simulate_data.py 保持一致）
CELL_TYPE_MARKER_RANGES = {
    "CD4+ T":    (0,   80),
    "CD8+ T":    (80,  160),
    "B Cell":    (160, 250),
    "NK Cell":   (250, 350),
    "Monocyte":  (350, 500),
}

CELL_TYPES = list(CELL_TYPE_MARKER_RANGES.keys())


class CellTypistAnnotator:
    """
    模拟 CellTypist 细胞类型注释器。

    当真实数据已有 cell_type 列时直接跳过；
    当无 cell_type 或强制重新注释时，基于 marker 基因得分进行启发式分类，
    输出格式与 CellTypist predicted_labels 保持一致。

    真实使用时替换 annotate() 为：
        import celltypist
        predictions = celltypist.annotate(adata, model="Immune_All_Low.pkl")
        adata.obs["cell_type"] = predictions.predicted_labels
    """

    def __init__(self, noise_level: float = 0.15, seed: int = 42):
        """
        noise_level: 分类噪声比例（0-1），模拟真实注释的不确定性
        """
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def _score_cell_types(self, expr_matrix: np.ndarray) -> np.ndarray:
        """
        对每个细胞计算各细胞类型的 marker 基因平均表达得分。
        返回形状：(n_cells, n_cell_types)
        """
        n_cells = expr_matrix.shape[0]
        scores = np.zeros((n_cells, len(CELL_TYPES)), dtype=np.float32)

        for j, ct in enumerate(CELL_TYPES):
            start, end = CELL_TYPE_MARKER_RANGES[ct]
            # 截取 marker 基因范围，但确保不超出实际基因数
            actual_end = min(end, expr_matrix.shape[1])
            if actual_end > start:
                marker_expr = expr_matrix[:, start:actual_end]
                scores[:, j] = marker_expr.mean(axis=1)

        return scores

    def annotate(
        self,
        adata: ad.AnnData,
        cell_type_col: str = "cell_type",
        force: bool = False,
    ) -> ad.AnnData:
        """
        为 AnnData 添加细胞类型注释。

        参数：
            adata:         输入 AnnData
            cell_type_col: 输出的 obs 列名
            force:         是否强制重新注释（即使已有 cell_type 列）

        返回：带注释的 AnnData（in-place 修改）
        """
        if cell_type_col in adata.obs.columns and not force:
            print(f"  细胞类型注释已存在（'{cell_type_col}'），跳过。"
                  f"（共 {adata.n_obs} 个细胞）")
            return adata

        print(f"  正在进行细胞类型注释（{adata.n_obs} 个细胞）...")

        # 提取表达矩阵
        X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)

        # 计算得分并选最高分的细胞类型
        scores = self._score_cell_types(X)
        best_type_idx = np.argmax(scores, axis=1)
        predicted = np.array([CELL_TYPES[i] for i in best_type_idx])

        # 添加可控噪声（模拟注释不完美）
        n_noisy = int(adata.n_obs * self.noise_level)
        noisy_idx = self.rng.choice(adata.n_obs, size=n_noisy, replace=False)
        for idx in noisy_idx:
            # 随机翻转为次优细胞类型
            sorted_types = np.argsort(scores[idx])[::-1]
            if len(sorted_types) > 1:
                predicted[idx] = CELL_TYPES[sorted_types[1]]

        adata.obs[cell_type_col] = predicted

        # 输出注释统计
        type_counts = pd.Series(predicted).value_counts()
        print(f"  注释完成。细胞类型分布：")
        for ct, cnt in type_counts.items():
            pct = cnt / adata.n_obs * 100
            print(f"    {ct:<15} : {cnt:>4} 个 ({pct:.1f}%)")

        return adata


# ─── 2. In-silico Bootstrapping（原位重抽样扩增）────────────────────────────

class InSilicoBootstrapper:
    """
    原位重抽样扩增器。

    策略（来自 planv0.1.md）：
    - 对每个供体的细胞进行随机有放回抽样，生成 K 份"伪样本（Pseudo-samples）"
    - 每份伪样本保留原供体的年龄、性别等元数据
    - 用于基线训练集：400人 → 1000+ 伪样本
    - 用于 PoC 验证：N=1 Pre/Post → 各 5 份伪样本（共 10 份）

    数据泄露保护：
    - Pseudo-samples 全程携带原始 donor_id
    - 切分时依然以原始 donor_id 为单位
    - pseudo_sample_id 仅用于区分同一供体的不同伪样本
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def bootstrap_donor(
        self,
        adata_donor: ad.AnnData,
        k_samples: int = 3,
        sample_frac: float = 0.8,
        min_cells: int = 50,
    ) -> List[ad.AnnData]:
        """
        对单个供体生成 K 份伪样本。

        参数：
            adata_donor:  单个供体的 AnnData
            k_samples:    生成伪样本数量（3-5 份）
            sample_frac:  每份伪样本抽取的细胞比例（有放回）
            min_cells:    最少保留细胞数（避免过小的伪样本）

        返回：K 个 AnnData 伪样本列表
        """
        n_cells = adata_donor.n_obs
        n_sample = max(min_cells, int(n_cells * sample_frac))

        pseudo_samples = []
        for k in range(k_samples):
            # 有放回重抽样
            sampled_idx = self.rng.choice(n_cells, size=n_sample, replace=True)
            pseudo = adata_donor[sampled_idx].copy()

            # 为每个伪样本分配唯一 ID（格式：{donor_id}_ps{k}）
            original_donor = adata_donor.obs["donor_id"].values[0]
            pseudo_id = f"{original_donor}_ps{k}"
            pseudo.obs["donor_id"] = original_donor          # 保留原始 donor_id！
            pseudo.obs["pseudo_sample_id"] = pseudo_id       # 新增伪样本 ID
            pseudo.obs.index = [
                f"{pseudo_id}_cell_{i:04d}" for i in range(len(pseudo.obs))
            ]

            pseudo_samples.append(pseudo)

        return pseudo_samples

    def bootstrap_dataset(
        self,
        adata: ad.AnnData,
        k_per_donor: int = 3,
        sample_frac: float = 0.8,
        donor_col: str = "donor_id",
        include_original: bool = True,
    ) -> ad.AnnData:
        """
        对整个数据集的每个供体进行 Bootstrapping，返回扩增后的 AnnData。

        参数：
            adata:           输入 AnnData（多供体合并）
            k_per_donor:     每个供体生成的伪样本数
            sample_frac:     每份伪样本的细胞抽取比例
            donor_col:       obs 中供体 ID 列名
            include_original: 是否在扩增集中保留原始样本

        返回：扩增后的 AnnData
        """
        donors = adata.obs[donor_col].unique()
        all_pseudo = []

        if include_original:
            original = adata.copy()
            original.obs["pseudo_sample_id"] = (
                original.obs[donor_col].astype(str) + "_original"
            )
            all_pseudo.append(original)

        print(f"  Bootstrapping：{len(donors)} 个供体，每人生成 {k_per_donor} 份伪样本...")

        for donor_id in donors:
            donor_mask = adata.obs[donor_col] == donor_id
            adata_donor = adata[donor_mask].copy()

            pseudo_list = self.bootstrap_donor(
                adata_donor, k_samples=k_per_donor, sample_frac=sample_frac
            )
            all_pseudo.extend(pseudo_list)

        augmented = ad.concat(all_pseudo, join="outer")

        n_original = adata.n_obs
        n_augmented = augmented.n_obs
        n_pseudo_samples = len(donors) * k_per_donor + (len(donors) if include_original else 0)

        print(f"  扩增完成：{n_original} → {n_augmented} 个细胞")
        print(f"  伪样本总数：{n_pseudo_samples} 份（原始 {len(donors) if include_original else 0} + 扩增 {len(donors) * k_per_donor}）")

        return augmented

    def bootstrap_intervention(
        self,
        adata_pre: ad.AnnData,
        adata_post: ad.AnnData,
        k_samples: int = 5,
        sample_frac: float = 0.8,
    ) -> Tuple[List[ad.AnnData], List[ad.AnnData]]:
        """
        对 N=1 干预数据（Pre/Post）各生成 K 份伪样本（共 2K 份）。
        用于 PoC 验证中的 ΔAge 统计检验。

        返回：(pre_pseudo_list, post_pseudo_list)
        """
        print(f"  干预数据 Bootstrapping：Pre/Post 各 {k_samples} 份伪样本...")

        pre_pseudos  = self.bootstrap_donor(adata_pre,  k_samples=k_samples, sample_frac=sample_frac)
        post_pseudos = self.bootstrap_donor(adata_post, k_samples=k_samples, sample_frac=sample_frac)

        print(f"  Pre  伪样本：{k_samples} 份，每份 {pre_pseudos[0].n_obs} 细胞")
        print(f"  Post 伪样本：{k_samples} 份，每份 {post_pseudos[0].n_obs} 细胞")

        return pre_pseudos, post_pseudos


# ─── 快速验证入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/simulated")
    )

    baseline_path = os.path.join(data_dir, "baseline_all.h5ad")
    pre_path  = os.path.join(data_dir, "DONOR_INTERVENTION_Pre.h5ad")
    post_path = os.path.join(data_dir, "DONOR_INTERVENTION_Post.h5ad")

    for p in [baseline_path, pre_path, post_path]:
        if not os.path.exists(p):
            print(f"[WARN]️  未找到文件: {p}\n请先运行 simulate_data.py")
            sys.exit(1)

    print("=" * 60)
    print("CellTypist 注释 + In-silico Bootstrapping 验证")
    print("=" * 60)

    # ── 1. 细胞类型注释 ──────────────────────────────────────────────────────
    print("\n[1/3] 细胞类型注释...")
    adata = ad.read_h5ad(baseline_path)
    annotator = CellTypistAnnotator(noise_level=0.1)
    adata = annotator.annotate(adata, force=True)

    # ── 2. 基线数据 Bootstrapping ─────────────────────────────────────────────
    print("\n[2/3] 基线数据 In-silico Bootstrapping...")
    bootstrapper = InSilicoBootstrapper(seed=42)
    adata_aug = bootstrapper.bootstrap_dataset(
        adata, k_per_donor=3, sample_frac=0.8, include_original=True
    )

    # 保存扩增数据
    aug_path = os.path.join(data_dir, "baseline_augmented.h5ad")
    adata_aug.write_h5ad(aug_path)
    print(f"  保存扩增数据 → {aug_path}")

    # ── 3. 干预数据 Bootstrapping ─────────────────────────────────────────────
    print("\n[3/3] 干预数据 (N=1) Bootstrapping...")
    adata_pre  = ad.read_h5ad(pre_path)
    adata_post = ad.read_h5ad(post_path)
    annotator.annotate(adata_pre,  force=True)
    annotator.annotate(adata_post, force=True)

    pre_pseudos, post_pseudos = bootstrapper.bootstrap_intervention(
        adata_pre, adata_post, k_samples=5
    )

    # 保存
    inter_aug_dir = os.path.join(data_dir, "intervention_pseudos")
    os.makedirs(inter_aug_dir, exist_ok=True)
    for i, ps in enumerate(pre_pseudos):
        ps.write_h5ad(os.path.join(inter_aug_dir, f"pre_ps{i}.h5ad"))
    for i, ps in enumerate(post_pseudos):
        ps.write_h5ad(os.path.join(inter_aug_dir, f"post_ps{i}.h5ad"))
    print(f"  干预伪样本保存 → {inter_aug_dir}")

    print("\n" + "=" * 60)
    print("[OK] CellTypist 注释 + Bootstrapping 验证完成！")
    print("=" * 60)
