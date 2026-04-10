# 👑 项目开发计划书 (Project Plan)
**项目名称：** 基于 Rank-Transformer 的 4D 单细胞免疫衰老与 MSC 疗效评估大模型 (PBMC-GPT)
**文档版本：** V1.0
**核心架构：** 纯 scRNA-seq + 秩排序 Token 化 + Transformer 注意力机制

---

## 一、 项目愿景与目标 (Project Vision & Objectives)

本项目旨在打破传统静态表观遗传时钟的技术瓶颈，构建全球首个**专精于外周血单核细胞（PBMC）的时序衰老大模型**。
* **终极目标：** 实现个体免疫生物学年龄的高精度预测，并首次在单细胞转录组水平上，提供针对间充质干细胞（MSC）等抗衰干预的疗效定量评估（$\Delta$ Age）与靶向机制解码（XAI）。
* **核心壁垒：** 采用"表达量排序（Rank-based）"编码，彻底免疫测序批次效应；内置动态扰动反馈机制，实现从"静态观察"到"4D 动态干预评估"的范式跃迁。

---

## 二、 数据资产与预处理策略 (Data Assets & Preprocessing)

项目由"三级数据池"驱动，并严格执行原位数据扩增策略：

### 1. 数据池定义
* **DAPT 语料池：** ~500万公开 PBMC scRNA-seq 数据（来源：CZ CELLxGENE API）。
* **基线训练池：** 400人 高质量 PBMC 横断面 `h5ad` 数据（含年龄、性别）。
* **PoC 验证池：** N=1 (Pre & Post) MSC 纵向干预配对 `h5ad` 数据。

### 2. 数据工程核心法则
* **细胞亚群注释：** 所有入模数据需提前完成自动化细胞类型注释（如使用 `CellTypist`），为后续 XAI 解释提供颗粒度。
* **原位重抽样扩增 (In-silico Bootstrapping)：**
  * 将 400 人的细胞按个体拆分，单客随机抽取生成 3-5 份"伪样本（Pseudo-samples）"，扩增训练集至 1000+ 规模，增强抗过拟合能力。
  * 将 N=1 个案的 Pre/Post 数据各拆分为 5 份伪样本，确保最终 $\Delta \text{Age}$ 的统计学显著性检验（如 T-test）。
* **Rank Token 化逻辑：** 提取 `Unscaled Raw Counts`，按表达量降序排列，截断 Top $N$（如 256/512）基因，映射至全局 Ensembl ID 字典。

---

## 三、 分阶段开发路线图 (Phased Roadmap)

### 📍 Phase 0: 概念验证与数据管线跑通 (MVP) [预计耗时：1-2周]
* **目标：** 跑通端到端的数据链路，验证 h5ad 质量。
* **关键任务：**
  * [ ] 编写 Python 自定义 DataLoader，实现 `h5ad` $\rightarrow$ Rank Token 的实时转换。
  * [ ] 抽取 5 人基线数据 + 1 人干预数据，实施细胞注释与数据拆分脚本测试。
  * [ ] 加载 HuggingFace 开源小参数预训练模型（Dummy Model），测试 Forward/Backward Pass 和 Loss 下降趋势。

### 📍 Phase 1: 领域自适应预训练 (DAPT) [预计耗时：3-4周]
* **目标：** 打造 PBMC 专属的底层理解引擎。
* **关键任务：**
  * [ ] 编写自动化脚本，从公开数据库高通量下载并清洗 500 万 PBMC 细胞。
  * [ ] 加载 `scGPT` 或 `Geneformer` 基础权重。
  * [ ] 冻结底层通用特征，启动掩码语言模型（MLM）无监督连续预训练。
* **里程碑：** 交付 **`PBMC-GPT-Base`** 模型权重。

### 📍 Phase 2: 任务特异性微调 (Task-Specific Fine-Tuning) [预计耗时：2-3周]
* **目标：** 赋予模型"免疫衰老感知"与预测能力。
* **关键任务：**
  * [ ] 严格按 8:2 比例（分层采样，防数据泄露）划分 400 人的个体数据集。
  * [ ] 在 `PBMC-GPT-Base` 顶部添加 MLP 回归头。
  * [ ] 使用 Huber Loss 进行端到端监督训练，结合验证集监控收敛状态（MAE 评估）。
* **里程碑：** 交付 **`PBMC-Aging-Predictor`** 临床级表观年龄预测器。

### 📍 Phase 3: N=1 临床推理与机制解码 (Zero-shot Inference & XAI) [预计耗时：1-2周]
* **目标：** 输出极具商业与科研价值的干预评估报告。
* **关键任务：**
  * [ ] 将 N=1 拆分后的 10 份伪样本（5 Pre + 5 Post）输入模型进行零样本推理。
  * [ ] 计算均值 $\Delta \text{Age}$ 及 p-value。
  * [ ] 提取 Transformer 最后一层 Attention 权重，按细胞类型（如 CD8+ T 细胞）聚合。
  * [ ] 映射 Attention Top 权重基因，解析 MSC 逆龄的核心驱动通路。
* **里程碑：** 产出 **《单细胞级 MSC 逆龄疗效与靶点机制报告 (PoC Edition)》**。

---

## 四、 技术架构与计算资源要求 (Infrastructure & Stack)

* **核心软件栈：**
  * Python 3.10+
  * `PyTorch` 2.0+ (底层引擎)
  * `HuggingFace Transformers` (模型管理)
  * `Scanpy` & `AnnData` (单细胞 I/O)
  * `Captum` (模型可解释性 XAI)
* **硬件资源规划：**
  * **存储：** ~4TB NVMe SSD（用于高频读写 h5ad 与 Checkpoints）。
  * **算力 (Phase 1/2)：** 2-4 × NVIDIA A100 (80GB) 或同等算力集群（建议按需租赁公有云，如 AWS EC2 p4d 实例）。
  * **算力 (Phase 0/3)：** 本地 RTX 3090/4090 或单张企业级 GPU 即可胜任推理与代码调试。

---

## 五、 风险控制与应对 (Risk Management)

1. **数据泄露风险 (Data Leakage)：**
   * **应对：** 严格在代码层面执行"以 Donor ID 为单位"的切分，确保同一人的扩增伪样本永远不会同时出现在 Train 和 Test 集中。
2. **算力 OOM (Out-of-Memory) 风险：**
   * **应对：** DataLoader 采用 `Lazy Loading` 策略，不将完整 400 人矩阵一次性读入内存；应用 `Gradient Accumulation`（梯度累加）技术，在小 Batch Size 下模拟大 Batch 训练效果。
3. **临床标签噪音：**
   * **应对：** 在微调阶段，对极端离群样本（如生理年龄 30 岁但具有严重基础免疫疾病）进行筛查，或将其权重在 Loss 计算中调低。

---

## 六、 近期行动计划 (Immediate Next Steps)

1. 配置本地/云端 Python 开发环境及 GPU 驱动。
2. 基于提供的 5 人基线 + 1 人干预数据，编写并测试 `In-silico Bootstrapping` (原位重抽样扩增) 与 `CellTypist` 注释脚本。
3. 编写核心的 `H5ad-to-Rank-Token` 转换器。
