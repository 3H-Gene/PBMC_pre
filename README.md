# PBMC-GPT

> **基于 Rank-Transformer 的 4D 单细胞免疫衰老与 MSC 疗效评估大模型**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-0%2F1%2F2%2F3-yellow)]()

---

## 目录

- [项目简介](#项目简介)
- [核心创新](#核心创新)
- [架构概览](#架构概览)
- [快速上手](#快速上手)
- [项目目录结构](#项目目录结构)
- [分阶段路线图](#分阶段路线图)
- [运行结果（模拟数据 PoC）](#运行结果模拟数据-poc)
- [技术栈](#技术栈)
- [数据资产规划](#数据资产规划)
- [风险控制](#风险控制)
- [贡献指南](#贡献指南)

---

## 项目简介

PBMC-GPT 致力于打破传统静态表观遗传时钟的技术瓶颈，构建**专精于外周血单核细胞（PBMC）的时序衰老大模型**。

- **终极目标**：实现个体免疫生物学年龄的高精度预测，并在单细胞转录组水平上，对间充质干细胞（MSC）等抗衰干预提供：
  - 定量疗效评估（ΔAge）
  - 靶向机制解码（XAI / Attention 权重）

- **核心壁垒**：采用"表达量排序（Rank-based）"编码，**彻底免疫测序批次效应**；内置动态扰动反馈机制，实现从"静态观察"到"4D 动态干预评估"的范式跃迁。

---

## 核心创新

| 特性 | 说明 |
|------|------|
| **Rank Token 化** | 提取 Raw UMI Counts，按表达量降序排列，截断 Top-N 基因映射为 Token 序列，彻底规避批次归一化引入的系统偏差 |
| **Value Embedding** | 将标量表达量投影到隐层空间，模型同时学习"哪些基因"和"表达了多少" |
| **In-silico Bootstrapping** | 在细胞层面对每位供体随机重抽样，生成多份"伪样本"，大幅扩增有限标签数据集并估计统计不确定性 |
| **ΔAge 评估** | Pre/Post 各生成 K 份伪样本，配对 T-test 检验，量化干预疗效显著性 |
| **供体级严格切分** | 以 Donor ID 为单位分层切分，防止同一人的细胞跨 Train/Test 集，消除数据泄露 |

---

## 架构概览

```
Raw scRNA-seq Counts (h5ad)
        |
        v
  Rank Tokenization
  (Top-N 基因按表达量降序排列 → Token ID 序列)
        |
        v
  Input Embedding
  = Gene Embedding + Value Embedding + Position Embedding
        |
        v
  Transformer Encoder
  (L 层 Multi-Head Self-Attention + FFN)
        |
        v
  [CLS] Token 向量
        |
   _____|_____
  |           |
  v           v
MLP 回归头   Attention XAI
(预测年龄)   (Top 权重基因提取)
```

---

## 快速上手

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/3H-Gene/PBMC_pre.git
cd PBMC_pre

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 安装依赖
pip install -r requirements.txt
```

### 端到端运行（模拟数据）

```bash
# 直接运行全流程 Pipeline（自动生成模拟数据 → 训练 → 推理 → 报告）
python run_pipeline.py
```

> **Windows 用户**：若 PowerShell 存在编码问题，使用辅助脚本：
> ```powershell
> python run_launcher.py
> # 输出日志位于 outputs/pipeline_run.log
> ```

### 分模块运行

```bash
# Step 1：生成模拟 h5ad 数据
python src/data/simulate_data.py

# Step 2：CellTypist 注释 + In-silico Bootstrapping
# （集成在 run_pipeline.py 中，也可直接 import 使用）

# Step 3：验证 Rank-Token DataLoader
python src/data/rank_token_dataset.py

# Step 4：验证 Rank-Transformer 模型
python src/model/rank_transformer.py

# Step 5：Phase 2 监督训练
python src/train/train_aging.py

# Step 6：Phase 3 推理 + XAI 报告
python src/inference/inference_xai.py
```

### 运行测试

```bash
pip install pytest
pytest tests/ -v
```

---

## 项目目录结构

```
PBMC_pre/
├── README.md                         # 本文档
├── planv0.1.md                       # 项目开发计划书 V1.0
├── requirements.txt                  # Python 依赖列表
├── run_pipeline.py                   # 端到端集成 Pipeline（Phase 0 → 2 → 3）
├── run_launcher.py                   # Windows 日志输出辅助脚本
│
├── src/                              # 核心源码包
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── simulate_data.py          # 模拟 h5ad 数据生成器（Phase 0 MVP）
│   │   ├── rank_token_dataset.py     # H5ad → Rank-Token DataLoader（核心）
│   │   └── bootstrapping.py         # CellTypist 注释 + In-silico Bootstrapping
│   ├── model/
│   │   ├── __init__.py
│   │   ├── rank_transformer.py      # Rank-Transformer 模型（含 MLP 回归头）
│   │   └── MLM_head.py             # MLM 预测头（Phase 1 DAPT）
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_aging.py           # Phase 2 监督训练循环（Huber Loss）
│   │   ├── phase1_continue_train.py # Phase 1 DAPT 继续预训练
│   │   └── phase2_finetune_train.py # Phase 2 监督微调（保守冻结策略）
│   └── inference/
│       ├── __init__.py
│       ├── inference_xai.py         # Phase 3 推理 + Attention XAI + ΔAge 报告
│       ├── phase3_predict.py        # Phase 3a 单样本年龄预测
│       └── phase3_paired_delta.py   # Phase 3b Pre/Post 配对 ΔAge 分析
│
├── data/
│   └── simulated/                   # 模拟 h5ad 数据（由 simulate_data.py 生成）
│       ├── DONOR_01.h5ad ~ DONOR_05.h5ad   # 5 名基线供体
│       ├── baseline_all.h5ad               # 合并基线数据
│       ├── baseline_augmented.h5ad         # Bootstrapping 扩增后数据
│       ├── DONOR_INTERVENTION_Pre.h5ad     # N=1 干预前
│       ├── DONOR_INTERVENTION_Post.h5ad    # N=1 干预后
│       └── intervention_pseudos/           # Pre/Post 各 5 份伪样本
│
├── checkpoints/
│   └── pbmc_aging_predictor_best.pt # 最优模型 Checkpoint（Val MAE 最低）
│
├── outputs/
│   ├── training_history.json         # 训练曲线数据
│   ├── delta_age_results.json        # ΔAge 数值结果
│   └── reports/
│       └── poc_report.md             # 自动生成的 PoC 报告（Markdown）
│
└── tests/
    ├── __init__.py
    └── test_pipeline_smoke.py        # Smoke 测试（数据生成、DataLoader、模型前向）
```

---

## 分阶段路线图

### Phase 0 — MVP 数据管线验证 ✅ 已完成

- [x] 模拟 h5ad 数据生成（5 名基线供体 + N=1 MSC 干预供体）
- [x] H5ad → Rank-Token DataLoader（Top-256，供体级严格切分）
- [x] CellTypist 模拟注释 + In-silico Bootstrapping（3 份/供体）
- [x] Rank-Transformer 模型（Dummy 配置：128d × 2层，372K 参数）
- [x] Forward/Backward Pass 验证（Loss 稳定下降）
- [x] Phase 2 监督训练（10 Epoch，Val MAE 从 60.6→12.5 岁）
- [x] Phase 3 N=1 推理 + ΔAge + T-test + Attention XAI + PoC 报告

### Phase 1 — 领域自适应预训练 (DAPT) — ✅ 已完成

- [x] 生成 Phase 1 语料池（50,000 细胞）
- [x] MLM 无监督连续预训练（Dummy 配置：128d × 2层，504K 参数）
- [x] 训练 2 Epoch，Loss 从 0.9398 → 0.9355
- [x] 里程碑：**`PBMC-GPT-Base`** 模型权重已保存

> **下一步**：加载 scGPT / Geneformer 基础权重，使用更大配置（512d × 8层）进行正式 DAPT 训练

### Phase 2 — 任务微调 — *待开始*

- [ ] 整合真实 400 人横断面 h5ad 数据集（含年龄/性别标签）
- [ ] 替换为 Base 配置（512d × 8层），加载 DAPT 权重
- [ ] Huber Loss 端到端监督微调（8:2 供体分层切分）
- [ ] 里程碑：**`PBMC-Aging-Predictor`** 临床级表观年龄预测器

### Phase 3 — N=1 临床推理与机制解码 — *待开始（真实数据）*

- [ ] 接入真实 N=1 Pre/Post MSC 干预配对 h5ad 数据
- [ ] ΔAge 评估 + GSEA/WikiPathways 通路富集分析
- [ ] 里程碑：**《单细胞级 MSC 逆龄疗效与靶点机制报告》**

---

## 运行结果（模拟数据 PoC）

基于 Phase 0 Dummy 模型（128d × 2层）在模拟数据上的运行结果：

### 训练曲线（Phase 2 监督微调，10 Epoch）

| Epoch | Train Loss | Train MAE (岁) | Val Loss | Val MAE (岁) | Val R² |
|-------|-----------|--------------|---------|------------|--------|
| 1     | ~12.0     | 60.6         | ~12.0   | 60.6       | ~0.00  |
| 5     | ~6.5      | ~30.0        | ~6.0    | ~28.0      | ~0.05  |
| 10    | ~3.0      | ~14.0        | ~2.8    | **12.50**  | ~0.15  |

### Phase 3 ΔAge 评估（模拟 MSC 逆龄，-5 岁等效）

| 指标 | 数值 |
|------|------|
| Pre 预测年龄均值 | ~65.x 岁 |
| Post 预测年龄均值 | ~60.x 岁 |
| **ΔAge** | **约 -5 岁** |
| p-value | < 0.05 |
| 结论 | 逆龄效应显著 |

> 注：以上结果基于模拟数据，仅用于验证管线完整性。真实数据结果以 Phase 2 训练后为准。

---

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch 2.0+ | 模型训练与推理 |
| 单细胞 I/O | AnnData / Scanpy | h5ad 读写与处理 |
| 模型管理 | HuggingFace Transformers（Phase 1） | DAPT 基础权重加载 |
| 可解释性 | Captum（Phase 3 扩展） | XAI 梯度归因 |
| 细胞注释 | CellTypist（Phase 1）| 自动化细胞类型标注 |
| 统计检验 | SciPy | T-test、ΔAge 显著性 |

---

## 数据资产规划

| 数据池 | 来源 | 规模 | 用途 |
|--------|------|------|------|
| DAPT 语料池 | CZ CELLxGENE API | ~500 万细胞 | Phase 1 无监督预训练 |
| 基线训练池 | 合作机构 | 400 人横断面 h5ad | Phase 2 监督微调 |
| PoC 验证池 | 临床合作 | N=1 Pre/Post MSC | Phase 3 疗效评估 |

**数据安全原则**：所有真实数据不上传至 GitHub。模拟数据通过 `src/data/simulate_data.py` 本地生成。

---

## 风险控制

1. **数据泄露**：严格以 Donor ID 为单位切分，同一供体的扩增伪样本永远不会跨 Train/Test 集。
2. **显存 OOM**：DataLoader 采用 Lazy Loading；支持 Gradient Accumulation（默认 4 步），在小 Batch 下模拟大 Batch 训练。
3. **临床标签噪音**：极端离群样本（如免疫疾病供体）在 Huber Loss 中自动降低惩罚权重；可在预处理阶段人工过滤。

---

## 贡献指南

1. Fork 本仓库并基于 `main` 创建 feature 分支。
2. 遵循现有代码风格（函数文档字符串、类型注解、ASCII 日志标记）。
3. 新功能请同步添加 `tests/` 测试用例。
4. 提交 PR 前请确保 `pytest tests/ -v` 全部通过。

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)

---

*本项目由 PBMC-GPT 团队开发维护。如有合作意向或数据共享需求，欢迎联系。*
