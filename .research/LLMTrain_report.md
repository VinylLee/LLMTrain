# 基于LLMORPH蜕变关系目录的LLM微调研究方案

**LLMORPH MR Fine-Tuning Research Proposal**

---

> 撰写日期：2026-06-01
> 项目背景：MetTrain (MR-driven semi-supervised NLI training) + LLMORPH (191 MRs across 24 NLP tasks)
> 状态：研究提案 / 实验设计阶段

---

## 目录

1. [摘要](#1-摘要)
2. [背景与相关工作](#2-背景与相关工作)
3. [核心研究问题与假设](#3-核心研究问题与假设)
4. [实验框架设计](#4-实验框架设计)
5. [实施路线图](#5-实施路线图)
6. [可行性分析](#6-可行性分析)
7. [风险与缓解策略](#7-风险与缓解策略)
8. [预期结果与解释框架](#8-预期结果与解释框架)
9. [参考文献](#9-参考文献)

---

## 1. 摘要

### 1.1 研究动机

蜕变测试（Metamorphic Testing, MT）在LLM评估中已展现出强大的缺陷发现能力——LLMORPH论文报告了平均18%的蜕变关系违反率，其中约62%经人工验证为真实缺陷。然而，蜕变关系（Metamorphic Relations, MRs）的潜力远未被充分挖掘：它们不仅可以作为**测试预言**（test oracle），还可以作为**训练信号**（training signal）。

MetTrain已初步验证了这一方向：利用14个MR生成训练数据，配合R-Drop正则化，实现了跨域泛化能力的显著提升（macro-F1最高+7pp）。但MetTrain的MR覆盖范围有限（仅NLI任务），且未系统研究LLM是否能通过微调**习得**蜕变关系的逻辑约束。

本提案旨在将LLMORPH的191个MR目录与MetTrain的MR驱动训练范式相结合，系统探索以下核心问题：**LLM能否通过蜕变关系微调，习得跨任务、跨MR的逻辑一致性能力？**

### 1.2 核心理念

```
源样本 x  ──MR变换──▶  跟随样本 x'
   │                      │
   ▼                      ▼
标签 y    ──MR逻辑──▶  标签 y' = f_MR(y)
   │                      │
   └────── 微调LLM ──────┘
       使LLM对(x, x')的预测
       满足MR约束关系
```

### 1.3 研究价值

| 维度 | 贡献 |
|------|------|
| **理论价值** | 首次系统研究LLM能否通过微调习得蜕变关系知识，揭示MR可学习性的边界 |
| **方法学价值** | 提出三种递进式MR微调策略（MR-as-Data → MR-as-Pair → MR-as-Instruction）的统一框架 |
| **实用价值** | 利用LLMORPH的191 MR目录，可快速扩展到24个NLP任务，产生可复现的benchmark |
| **学术价值** | 填补蜕变测试与LLM微调交叉领域的空白，相关工作极少（仅MetTrain一篇） |

---

## 2. 背景与相关工作

### 2.1 MetTrain：MR驱动的半监督NLI训练

**核心思想**：将蜕变关系从测试阶段前移到训练阶段。

**技术路线**：
- 使用冻结的4B LLM（Gemma 4B）作为MR执行器，将源NLI样本变换为跟随样本
- MR变换会系统性地改变标签：`flip`类MR（如negation_flip）反转标签关系，`preserve`类MR（如synonym_replacement）保持标签关系
- 训练目标NLI模型时引入R-Drop正则化（KL散度），约束源样本和跟随样本的输出分布一致性
- 训练数据同时包含有标签数据和大量无标签数据（后者仅通过MR生成跟随样本）

**关键结果**：
- Macro-F1在多个OOD数据集上提升最高7pp
- 尤其在训练数据稀疏场景下（如100-1000样本）效果显著
- 复合MR（composite MRs）进一步提升了泛化性能

**局限性**：
- 仅覆盖NLI任务和14个MR
- 未验证LLM是否真正"学会"了MR的语义逻辑，而非仅仅利用数据增广的统计信号
- 未探索MR跨任务迁移的可能性

### 2.2 LLMORPH：LLM蜕变测试的系统性研究

**核心贡献**：
- 通过系统文献综述收集了191个MR，覆盖24个NLP任务
- 实现了36个MR的自动化框架，在4个任务（QA, NLI, SA, RE）上测试了3个LLM
- 共运行561K次蜕变测试

**关键发现**：

| 发现 | 数据 |
|------|------|
| 平均MR违反率λ | 18% |
| 真阳性率（人工验证） | ~62% |
| MT发现的缺陷中被传统测试遗漏的比例 | ~11% |
| 任务无关MR的有效性 | MR-9, MR-102跨任务均高效 |

**MR分类体系**（LLMORPH实现的36个MR）：

| 任务 | MR ID范围 | 示例 |
|------|-----------|------|
| 通用（General） | MR-1 ~ MR-20 | 同义词替换(MR-1)、关键词替换(MR-2)、大小写变换(MR-8)、数字替换(MR-9)、命名实体替换(MR-10) |
| NLI特定 | MR-71 ~ MR-80 | 前提-假设交换(MR-71)、否定翻转(MR-72)、矛盾条件添加(MR-73) |
| QA特定 | MR-84 ~ MR-100+ | 问题改写(MR-84)、答案扰动(MR-85)、上下文扰动(MR-90) |
| SA特定 | MR-101 ~ MR-120 | 情感词替换(MR-101)、否定插入(MR-102)、程度词修改(MR-105) |
| RE特定 | MR-121 ~ MR-136 | 实体交换(MR-121)、关系扰动(MR-122) |

**LLMORPH的关键局限**（为我们的研究留下了空间）：
1. MR仅用于**测试**，未用于训练
2. 未研究LLM是否可以**学习**MR约束
3. 未探索MR之间的关联和层级结构
4. 假阳性问题（主要来自输入变换质量和输出比较方法）

### 2.3 文献定位：我们处于什么位置？

| 工作 | MR用于训练？ | MR用于测试？ | 覆盖任务数 | 研究MR可学习性？ |
|------|:-----------:|:-----------:|:----------:|:---------------:|
| MetTrain | Yes (14 MRs) | No | 1 (NLI) | 部分（仅OOD泛化） |
| LLMORPH | No | Yes (36 MRs) | 4 | No |
| METAL (ICSE'20) | No | Yes | 3 (NLP) | No |
| CheckList (ACL'20) | No | Yes (behavioral) | 3 (NLP) | No |
| 本提案 | **Yes** (LLMORPH的191 MRs) | Yes | **24** | **Yes (核心研究问题)** |

**关键空白**：目前**没有任何工作**系统研究LLM是否可以通过微调习得蜕变关系知识。这是蜕变测试与LLM训练交叉领域的一个根本性问题——如果MR是可学习的，那么它们不仅是被动的质量保证工具，而是可以主动提升模型鲁棒性的训练基础设施。

### 2.4 相关技术背景

**自监督一致性训练**：
- FixMatch, UDA等半监督方法通过数据增强的一致性约束提升性能
- 但数据增强（如回译、同义词替换）缺乏**语义逻辑保真度**
- MR提供了**有逻辑保证**的变换——MR定义了什么应该变、什么不应该变

**指令微调（Instruction Tuning）**：
- FLAN, T0等工作表明，将任务格式化为自然语言指令可提升LLM的zero-shot泛化
- MR逻辑可以自然地编码为指令：*"如果你看到句子A蕴含句子B，那么在否定B之后，A应该不蕴含否定后的句子"*

**对抗训练**：
- 传统对抗训练在嵌入空间添加扰动
- MR-对抗训练在**语义空间**添加扰动（MR违反样本），具有更好的可解释性

---

## 3. 核心研究问题与假设

### 3.1 总体研究问题（Overarching RQ）

> **RQ0**: LLM能否通过蜕变关系微调，习得跨任务的逻辑一致性能力？这种能力能否泛化到未见过的MR和新任务？

### 3.2 子研究问题

#### RQ1: MR可学习性 (Learnability)

| 子问题 | 假设 |
|--------|------|
| **RQ1a**: LLM能否在微调后降低已知MR的违反率？ | **H1a**: 微调后，训练集中出现过的MR的违反率将显著下降（预期从~18%降至<5%） |
| **RQ1b**: MR的可学习性是否与MR类型相关？ | **H1b**: 词汇层面MR（如同义词替换）比语义逻辑层面MR（如否定翻转）更容易学习 |
| **RQ1c**: 小模型（3-14B）是否能习得MR知识？ | **H1c**: 14B模型可有效学习，3B模型可能容量不足以编码复杂MR逻辑 |

#### RQ2: 微调策略比较 (Strategy Comparison)

| 子问题 | 假设 |
|--------|------|
| **RQ2a**: S1 (MR-as-Data) vs S2 (MR-as-Pair)——对偶结构相比随机混入单样本是否有增益？ | **H2a**: S2优于S1，对偶结构让模型隐式感知MR变换规律，MRV降低更多 |
| **RQ2b**: S2 vs S3 (MR-as-Instruction)——在已有对偶结构的基础上，显式添加MR规则描述能带来多少额外提升？ | **H2b**: S3显著优于S2——这是最关键假设，直接验证"MR规则本身是否可学习"。显式规则描述让模型从隐式模式匹配提升为显式规则推理 |
| **RQ2c**: S1 → S2 → S3 的递进增益是否单调？瓶颈在哪一步？ | **H2c**: S2→S3的增益（规则知识）> S1→S2的增益（对偶结构），因为规则描述的信息量远大于对比结构 |

#### RQ3: 跨任务与跨MR泛化 (Generalization)

| 子问题 | 假设 |
|--------|------|
| **RQ3a**: 在NLI上微调的MR知识能否迁移到SA/QA任务？ | **H3a**: 任务无关MR（MR-9, MR-102等）可以实现跨任务迁移；任务特定MR则不能 |
| **RQ3b**: 在部分MR上微调，能否泛化到同一逻辑类型的未见MR？ | **H3b**: 同一MR类别内（如所有"否定"类MR）存在泛化，跨类别泛化较弱 |

#### RQ4: 鲁棒性与副作用 (Robustness)

| 子问题 | 假设 |
|--------|------|
| **RQ4a**: MR微调是否会损害原始任务的性能（灾难性遗忘）？ | **H4a**: 适当的MR微调不会显著损害原始性能（<2%下降）；过度微调会导致遗忘 |
| **RQ4b**: MR微调是否能提升对自然分布漂移的鲁棒性？ | **H4b**: MR微调可提升OOD鲁棒性，因为它迫使模型学习不变的逻辑约束 |

### 3.3 假设检验框架

```
正面结果（支持研究价值）:
├── 微调后MR违反率显著下降（>50%相对降低）
├── 跨MR泛化成立（未见MR的违反率也下降>20%）
├── 跨任务迁移成立（任务无关MR在新任务上有效）
└── 原始任务性能不显著下降（<2%）

负面结果（不支持/部分支持）:
├── 微调仅记忆训练MR的特定模式，无泛化
├── 跨任务迁移失败（MR知识是任务绑定的）
├── 小模型（<7B）无法编码MR逻辑
└── 灾难性遗忘严重（>5%原始任务下降）
```

---

## 4. 实验框架设计

### 4.1 实验一：MR可学习性基础实验

**目标**：验证LLM是否能通过微调学习MR知识

**设计**：
```
自变量:
  - 模型规模: 3B, 8B, 14B (3 levels)
  - MR类型: 词汇层 / 句法层 / 语义层 (3 levels, 每种选4个MR)
  - 训练数据量: 100, 500, 2000, 全量 (4 levels)

因变量:
  - MR违反率 (MRV) — 主要指标
  - 标准任务准确率 — 保留原始能力检查
  - 未见MR的违反率 — 泛化能力

控制:
  - 同一基座模型系列（如Qwen2.5系列: 3B, 7B, 14B）
  - 相同训练超参数（epochs, learning rate, batch size）
  - 相同评测数据集
```

**MR选择策略**（从LLMORPH的36个实现MR中选取）：

| 类别 | MR ID | 描述 | NLI任务适用性 |
|------|-------|------|:---:|
| 词汇层 | MR-1 | 同义词替换 | Yes |
| 词汇层 | MR-2 | 关键词替换 | Yes |
| 词汇层 | MR-8 | 大小写变换 | Yes |
| 词汇层 | MR-10 | 命名实体替换 | Yes |
| 句法层 | MR-5 | 主动-被动语态转换 | Yes |
| 句法层 | MR-71 | 前提-假设交换 | NLI only |
| 句法层 | MR-6 | 句子重组 | Yes |
| 语义层 | MR-72 | 否定翻转 | NLI only |
| 语义层 | MR-73 | 矛盾条件添加 | NLI only |
| 语义层 | MR-102 | 否定插入 | 跨任务 |
| 语义层 | MR-9 | 数字替换 | 跨任务 |

### 4.2 实验二：微调策略对比

**目标**：对比三种MR微调策略的效果，验证显式MR规则编码的增量价值

**关键前提**：所有策略均使用**标准LM loss（next-token prediction SFT）**。与MetTrain有本质区别——MetTrain面向BERT分类器，使用CrossEntropy + R-Drop KL散度（需两次前向传播，约束源/跟随样本的输出分布一致性）；而本工作面向LLM生成式模型，使用标准SFT（单次前向传播）。在LLM微调中，MR的角色从"训练时的正则化约束"转变为"数据结构化方式"——区别仅在于数据如何格式化送入模型。

#### 三种策略的递进关系

```
S1: MR-as-Data          纯数据增强，模型不知道MR存在
    └─ 源样本和跟随样本随机混入训练集，单样本SFT

S2: MR-as-Pair          对偶结构，模型隐式感知MR
    └─ 源-跟随对在同一context中，模型从对比中学习

S3: MR-as-Instruction   S2的对偶结构 + 显式MR规则描述
    └─ 模型不仅看到变换前后的对比，还被告知MR的逻辑规则
    └─ ★ 核心策略：目标是让LLM直接学到MR本身
```

三种策略均使用标准LM loss，区别仅在于**数据如何格式化**：

---

#### S1: MR-as-Data（纯数据增强 SFT）

**核心思想**：MR仅作为**数据生成工具**。使用MR变换扩充训练集，模型通过标准SFT在增强数据上训练，**隐式**受益于MR引入的多样性——但模型完全不知道MR的存在。

**与MetTrain的关键区别**：
- **MetTrain**: BERT分类器 + R-Drop KL散度（两个前向传播，约束源/跟随样本的输出分布一致性）——MR是训练时的正则化约束
- **本工作S1**: LLM + 标准SFT（单次前向，next-token prediction）——MR纯粹是数据生成工具，用于产生更多样的训练样本

**数据格式**（标准NLI任务prompt，不含任何MR信息）：

```
源样本（原始数据）:
Premise: A man is playing a guitar.
Hypothesis: A musician is performing.
Question: What is the relationship between premise and hypothesis?
(A) entailment (B) neutral (C) contradiction
Answer: (A) entailment
```

```
跟随样本（MR变换后，以negation_flip为例）:
Premise: A man is playing a guitar.
Hypothesis: A musician is not performing.    ← MR变换后的hypothesis
Question: What is the relationship between premise and hypothesis?
(A) entailment (B) neutral (C) contradiction
Answer: (C) contradiction                     ← MR变换后的标签
```

- 源样本和跟随样本**随机混入**训练集，每个样本独立训练
- 模型不知道哪些样本经过了MR变换——它只看到"更多的训练数据"
- 训练目标: **标准LM loss（next-token prediction）**
- 模型的预期行为: 通过数据的多样性隐式提升鲁棒性，但并不"理解"MR

---

#### S2: MR-as-Pair（对偶样本对比学习）

**核心思想**：源样本和MR跟随样本**成对**放入同一上下文窗口。不描述MR规则，但提供"变换前 vs 变换后"的对比结构。模型从对比中**隐式**发现MR的变换规律。

**数据格式**：

```
Original example:
Premise: A man is playing a guitar.
Hypothesis: A musician is performing.
Label: entailment

After applying a transformation (the hypothesis has been changed):
Premise: A man is playing a guitar.
Hypothesis: A musician is not performing.
Question: What is the label now?
Answer: contradiction
```

- 源-跟随对出现在同一context窗口中
- 模型从对比中隐式发现规律（如 "hypothesis被否定 → label翻转"）
- 训练目标: **标准LM loss（next-token prediction）**
- 不告知任何MR规则名称或逻辑

---

#### S3: MR-as-Instruction（对偶 + 显式MR规则学习）★ 核心策略

**核心思想**：在S2的对偶结构基础上，将MR的**逻辑规则显式写入prompt**。S2让模型看到变换前后的对比但不理解规则，S3更进一步——明确告诉模型"这是一条MR规则，逻辑如下..."，同时展示规则在具体样本上的应用。模型既能从抽象规则中推理，又能通过具体对偶验证理解。

**S2 → S3 的递进关系**：

```
S2 (MR-as-Pair):
  源-跟随对 + 无规则描述
  → 模型看到"变换前是entailment，变换后是contradiction"
  → 模型隐式感知规律，但不理解为什么

S3 (MR-as-Instruction):
  S2的对偶结构 + MR规则描述
  = 抽象规则 + 具体示范
  → 模型既知道"negation应该翻转标签"（规则）
  → 也看到"这个具体例子里entailment→contradiction"（示范）
  → 目标是学到MR本身，将规则内化为可调用的推理能力
```

**数据格式**（MR规则 + 源-跟随对）：

```
Below is a metamorphic relation rule for Natural Language Inference:

Rule: NEGATION_FLIP
Description: If the hypothesis H is negated, the entailment relationship 
between premise P and the negated hypothesis ¬H should reverse.
- If P entails H → P should contradict ¬H
- If P contradicts H → P should entail ¬H
- If P is neutral to H → P should remain neutral to ¬H

Original example:
Premise: A man is playing a guitar.
Hypothesis: A musician is performing.
Label: entailment

After applying NEGATION_FLIP:
Premise: A man is playing a guitar.
Hypothesis: A musician is not performing.
Question: What is the label now?
Answer: contradiction
```

- **规则描述 + 对偶结构**：模型同时获得抽象规则和具体示范
- 训练目标: **标准LM loss（next-token prediction）**
- **关键假设**: 显式规则描述 + 对偶示范，能让模型将MR内化为可泛化的推理能力，从而泛化到**未见过的MR**（这是S1和S2无法做到的）

---

#### 策略对比总结

| 策略 | 对偶结构 | 规则描述 | 核心机制 | 模型对MR的认知 | 预期MR泛化 |
|------|:-------:|:-------:|---------|:------------:|:---------:|
| **S1: MR-as-Data** | ✗ | ✗ | MR作为数据增强工具 | 完全不知道MR存在 | 无 |
| **S2: MR-as-Pair** | ✓ | ✗ | 对比中隐式发现规律 | 隐式感知变换模式 | 弱（模式匹配） |
| **S3: MR-as-Instruction** | ✓ | ✓ | 规则+示范，显式学习MR本身 | 理解MR逻辑规则 | **强**（规则内化） |

#### 核心研究问题（策略层面）

- **S1 vs S2**: 对偶结构（看到变换前后的对比）相比随机混入单样本，是否带来增益？——验证"对比结构"的价值
- **S2 vs S3**: 在已有对偶结构的基础上，显式添加MR规则描述，能带来多少额外提升？——这是**最关键**的比较，直接验证"MR规则是否可学习"
- **S1 vs S3**: 从完全不知道MR到显式学习MR规则，整体增益有多大？——验证整个MR微调范式的价值


### 4.3 实验三：MR泛化层级实验

**目标**：系统研究MR泛化的边界

**设计**：留一法(Leave-One-Out) + 按类别泛化

```
训练MR集合         测试MR集合              预期泛化程度
─────────────────────────────────────────────────
{negation_flip,     {negation_flip,          高（完全训练）
 antonym_sub,       antonym_sub,
 synonym_rep}       synonym_rep}

{negation_flip,     {adding_contradiction,   中-高（同类别）
 antonym_sub}       conditional_clause}
                    （同为flip类MR）

{all flip MRs}      {all preserve MRs}       低-中（跨类别）

{all NLI MRs}       {SA MRs, QA MRs}         低（跨任务）
                    （仅任务无关MR）

{lexical MRs}       {semantic MRs}           低（跨变换层次）
```

### 4.4 评估指标体系

| 指标类别 | 具体指标 | 计算方式 |
|----------|---------|---------|
| **MR一致性** | MRV (MR Violation rate) | 违反MR约束的样本占比 |
| | MRV降低率 | (MRV_pre - MRV_post) / MRV_pre |
| | MR泛化率 | 未见MR的MRV降低量 / 已见MR的MRV降低量 |
| **任务性能** | Accuracy | 标准分类准确率 |
| | Macro-F1 | 类别平衡的F1 |
| | OOD Accuracy | 跨域测试集准确率 |
| **学习质量** | MR知识保持率 | 微调后N天后MRV的保持程度 |
| | 灾难性遗忘程度 | 原始任务准确率下降幅度 |
| **效率** | 收敛速度 | 达到目标MRV所需训练步数 |
| | 数据效率 | 不同训练量下的MRV曲线 |

### 4.5 数据集规划

**训练数据**：

| 数据集 | 规模 | 用途 | 来源 |
|--------|------|------|------|
| SNLI | 570K | NLI训练（MR数据生成源） | 已有 |
| MNLI | 433K | NLI训练+OOD测试 | 已有 |
| RTE | 2.5K | NLI binary测试 | 已有 |
| SICK | 10K | NLI语义测试 | 已有 |
| SST-2 | 70K | SA训练（跨任务实验） | 待获取 |
| SQuAD v2 | 130K | QA训练（跨任务实验） | 待获取 |

**MR变换数据生成**：

当前项目已有9个MR应用于NLI数据。需扩展：
1. 使用LLMORPH框架（或其MR定义）生成NLI上的额外MR变换
2. 为SA和QA任务生成MR变换数据
3. 质量验证：人工抽样+自动一致性检查

---

## 5. 实施路线图

### Phase 1: 基础验证（2-3周）

**目标**：在最小可行设置下验证核心假设（MR可学习性）

**任务清单**：

1. **扩展MR数据生成** (3-4天)
   - [ ] 从LLMORPH论文提取可实现的MR定义（优先选取有明确变换规则的MR）
   - [ ] 编写 `scripts/generate_mr_data.py` —— 使用LLM API生成MR变换数据
   - [ ] 为NLI数据生成额外5-10个MR的变换数据
   - [ ] 质量检查：每个MR抽取50个样本人工验证变换正确性

2. **构建微调pipeline** (3-4天)
   - [ ] 创建 `scripts/finetune_mr.py` —— 基于`transformers`库的微调脚本
   - [ ] 支持LoRA/QLoRA（适配本地GPU，3-14B模型）
   - [ ] 实现S1 (MR-as-Data) 的标准SFT数据加载（纯数据增强，单样本随机混入）
   - [ ] 实现S2 (MR-as-Pair) 的对偶数据格式（源-跟随对，无规则描述）
   - [ ] 实现S3 (MR-as-Instruction) 的prompt模板和数据格式（S2对偶结构 + 显式MR规则）
   - [ ] 集成现有 `test_llm.py` + `evaluate.py` 作为评测后端

3. **小规模验证实验** (5-7天)
   - [ ] 使用Qwen2.5-7B（通过LM Studio或本地加载）作为基座模型
   - [ ] 在SNLI上使用5个MR进行微调（3个flip + 2个preserve）
   - [ ] 评估微调前后的MRV变化
   - [ ] 评估原始NLI准确率是否保持
   - [ ] **决策门**: 如果MRV下降<10%，重新评估假设；如果>30%，进入Phase 2

**交付物**：
- `scripts/generate_mr_data.py`
- `scripts/finetune_mr.py` (v0.1)
- Phase 1 实验报告（jupyter notebook）

### Phase 2: 系统实验（4-6周）

**目标**：完成全部四个实验，收集系统性的实验数据

**任务清单**：

1. **扩展MR覆盖** (1周)
   - [ ] 实现LLMORPH中所有36个可自动化MR的数据生成
   - [ ] 扩展到SA任务（SST-2）+ QA任务（SQuAD样本）
   - [ ] 构建统一的MR数据格式规范

2. **实验一：MR可学习性** (1周)
   - [ ] 3个模型规模 x 3个MR层级 x 3个数据量 = 27组实验
   - [ ] 优先跑Qwen2.5系列（3B, 7B, 14B）减少模型架构差异
   - [ ] 自动化批量实验脚本 `scripts/run_mr_experiments.sh`

3. **实验二：微调策略对比** (1周)
   - [ ] 实现完整的三种策略（S1-S3）的数据格式化
   - [ ] 3个策略 x 3个模型规模 = 9组实验
   - [ ] 每个策略的最佳超参数搜索

4. **实验三：泛化层级** (1周)
   - [ ] 留一法MR实验（~10组）
   - [ ] 跨类别泛化实验
   - [ ] 跨任务迁移实验

5. **实验四：鲁棒性深度分析** (1周)
   - [ ] 灾难性遗忘分析（多个checkpoint）
   - [ ] OOD鲁棒性测试
   - [ ] 错误分析：MRV下降但准确率不提升的原因

**交付物**：
- `scripts/finetune_mr.py` (v1.0, 支持全部三种策略)
- `scripts/run_mr_experiments.sh`
- 完整实验数据（wandb或tensorboard日志）
- Phase 2 实验报告

### Phase 3: 分析与写作（2-3周）

**目标**：深度分析实验结果，撰写论文

1. **数据分析** (1周)
   - [ ] 统计显著性检验（McNemar检验或paired bootstrap）
   - [ ] 错误案例分析（分类MRV来源）
   - [ ] MR嵌入空间可视化（t-SNE/UMAP of MR representations）
   - [ ] 注意力模式分析（模型对不同MR类型的注意力差异）

2. **论文撰写** (1-2周)
   - [ ] 目标会议：ACL, EMNLP, 或 ICSE (软件工程+AI交叉)
   - [ ] 论文大纲：Introduction → Related Work → Method → Experiments → Analysis → Conclusion
   - [ ] 与MetTrain论文形成互补（MetTrain关注NLI训练方法，本工作关注MR学习的泛化理论）

**交付物**：
- 完整实验数据分析报告
- 论文初稿

---

## 6. 可行性分析

### 6.1 基础设施评估

**当前可用资源**：

| 资源 | 规格 | 评估 |
|------|------|------|
| GPU | 本地GPU（推测为单卡，24-48GB VRAM）| 足以运行QLoRA微调14B模型 |
| 模型服务 | LM Studio (localhost:1234) | 可用于MR数据生成和评测 |
| 已测试模型 | Qwen2.5-14B, LLaMA 3.1-8B, LLaMA 3-8B, LLaMA 3.2-3B, Gemma 3-4B, Phi-4 | 丰富的baseline数据 |
| 代码基础 | test_llm.py, evaluate.py, mrv.py, run_all.sh | 成熟的评测pipeline |
| 数据 | SNLI/MNLI/RTE/SICK MR测试数据（9 MR x 5 datasets） | NLI部分的MR数据就绪 |

**关键技术路线**：

```
MR数据生成:
  Option 1: LM Studio本地模型（成本低，速度慢但可控）
  Option 2: 远程API（DeepSeek/Bailian，速度快但有成本）
  推荐: 混合使用——开发/调试用本地，大规模生成用API

微调:
  框架: transformers + peft (QLoRA 4-bit)
  训练器: TRL SFTTrainer（三种策略统一使用，均为标准SFT，区别仅在数据格式化）
  硬件需求: ~16GB VRAM (QLoRA 7B), ~24GB (QLoRA 14B)
  估算: 在SNLI 10K样本上微调7B模型约需2-4小时（单GPU）

评测:
  复用现有 test_llm.py pipeline
  扩展 evaluate.py 支持多任务评测
```

### 6.2 成本估算

| 阶段 | 活动 | 估算成本 |
|------|------|---------|
| Phase 1 | API调用生成MR数据（~50K样本） | ~$5-10 |
| Phase 1 | 小规模微调实验（~10次run） | 本地GPU，电费忽略 |
| Phase 2 | API调用生成跨任务MR数据（~200K样本） | ~$20-40 |
| Phase 2 | 系统微调实验（~50次run） | 本地GPU，电费忽略 |
| Phase 2 | 评测API调用（~100K次） | ~$10-20 |
| **总计** | | **$35-70** |

### 6.3 技术风险评估

| 风险 | 严重度 | 概率 | 缓解措施 |
|------|:------:|:----:|---------|
| MR变换质量不足以支撑训练 | 高 | 中 | 使用强LLM生成+人工验证+多MR一致性交叉检查 |
| QLoRA微调无法有效编码MR知识 | 中 | 低 | 逐步增加可训练参数比例（QLoRA→LoRA→全参数） |
| 3B模型容量不足 | 低 | 中 | 设计为预期结果而非blocker——这本身是有价值的发现 |
| 跨任务MR迁移效果微弱 | 中 | 中-高 | 优先验证任务内泛化，跨任务作为探索性实验 |

### 6.4 LLMORPH MR在NLI上的适用性映射

从LLMORPH的36个实现MR中，可适配NLI的MR如下：

| MR ID | 类别 | 可自动化？ | 实现难度 | 优先级 |
|-------|------|:---------:|:-------:|:-----:|
| MR-1 | 同义词替换 | Yes | 低 | P0 |
| MR-2 | 关键词替换 | Yes | 低 | P0 |
| MR-5 | 语态转换 | Yes | 低 | P0 |
| MR-6 | 句子重组 | Yes | 中 | P1 |
| MR-8 | 大小写变换 | Yes | 低 | P1 |
| MR-9 | 数字替换 | Yes | 中 | P1 |
| MR-10 | 实体替换 | Yes | 中 | P1 |
| MR-71 | 前提假设交换 | Yes | 低 | P0 |
| MR-72 | 否定翻转 | Yes | 低 | P0 |
| MR-73 | 矛盾条件添加 | Yes | 中 | P0 |
| MR-74 | 程度词修改 | Partially | 中 | P1 |
| MR-75 | 情感极性修改 | Yes | 中 | P1 |
| MR-102 | 否定插入 | Yes | 中 | P1 |

> 注：当前项目已覆盖的9个MR大致对应上述部分MR（如negation_flip ~ MR-72, synonym_replacement ~ MR-1, voice_switch ~ MR-5），但LLMORPH版本提供了更标准化的定义和跨任务适用性分析。

---

## 7. 风险与缓解策略

### 7.1 核心风险详细分析

#### 风险1: MR变换的质量瓶颈

**问题**：LLMORPH报告假阳性率约38%（=1-62%真阳性率），主要来源是输入变换质量和输出比较。如果训练数据的MR变换本身有38%的错误，模型可能学到的是噪声而非MR逻辑。

**缓解**：
- 多阶段质量过滤：LLM生成 → 规则验证 → 人工抽样（每MR 50样本）→ 一致性检查（多个LLM交叉验证）
- 优先选择**确定性MR**（如negation_flip, voice_switch），避免使用依赖LLM主观判断的MR
- 设计"MR保真度"指标：变换后的样本是否正确体现了MR意图

**质量检查脚本示例**：
```python
def validate_mr_transformation(original, transformed, mr_type):
    """检查MR变换是否保持了应保持的关系"""
    if mr_type == "negation_flip":
        # negation应该在hypothesis中
        has_negation = check_negation(transformed.hypothesis)
        # premise不应被改变
        premise_unchanged = (original.premise == transformed.premise)
        # 标签应该翻转
        label_flipped = check_label_flip(original.label, transformed.label)
        return has_negation and premise_unchanged and label_flipped
```

#### 风险2: 灾难性遗忘

**问题**：MR微调可能使模型过度关注MR约束而牺牲原始任务性能。

**缓解**：
- 混合训练：MR样本与原始任务样本按比例混合（建议初始比例1:3）
- 监控验证集上的原始任务性能，early stopping
- 使用较小的学习率（建议为预训练LR的1/10）
- 考虑使用replay buffer保留原始任务的重要样本

#### 风险3: 评估指标的局限性

**问题**：MRV降低可能并不代表模型真正"理解"了MR逻辑。例如，模型可能学会了对所有flip类变换输出contradiction（而非真正理解标签翻转逻辑）。

**缓解**：
- 设计**反事实测试**：使用MR定义之外的变换（如随机替换），检查模型是否仍表现出"一致性"
- 设计**MR组合测试**：连续应用两个MR（如negation_flip + synonym_replacement），检查模型是否跟踪累积的逻辑变化
- 使用**探测分类器**（probing classifier）：检查模型隐藏层是否编码了MR类型信息

#### 风险4: 实验规模与计算约束

**问题**：27组（实验一）+ 9组（实验二）+ 10组（实验三）= ~46组实验，计算量大。

**缓解**：
- Phase 1快速过滤：用小规模实验（100样本，1 epoch）快速筛选有希望的配置
- 使用QLoRA降低单次实验成本
- 优先在3B模型上做参数搜索，仅在最优配置上跑7B和14B
- 利用已有的5个模型baseline数据（已有MRV报告），减少基线评测需求

### 7.2 技术难点

| 难点 | 解决方案 |
|------|---------|
| MR-as-Instruction的prompt设计 | 参考FLAN的instruction模板；A/B测试不同表述；MR逻辑形式化（→Prolog-like规则） |
| 多任务MR数据格式统一 | 设计统一的JSON schema，字段：`task_type`, `mr_id`, `mr_category`, `source`, `target`, `transform_fn` |
| 微调与评测的pipeline自动化 | 扩展现有`run_all.sh`，增加`--mode finetune`模式 |
| 统计显著性检验 | 使用bootstrap resampling（1000次），报告置信区间而非仅点估计 |

---

## 8. 预期结果与解释框架

### 8.1 成功场景

**场景S1: 强MR学习 + 泛化**（最理想）

```
观察:
  - 微调后MRV从18%降至3-5%（~75%相对降低）
  - 同类别未见MR的MRV也显著下降（~50%相对降低）
  - 跨任务无关MR的MRV下降>20%
  - 原始任务准确率保持（±1%）
  - MR-as-Instruction策略在泛化上显著优于MR-as-Data

解释:
  - LLM确实能通过微调习得MR的逻辑结构
  - MR知识具有可迁移性，不限于特定任务
  - 指令编码促进了MR逻辑的抽象化
  → 强证据支持MR作为通用训练基础设施的可行性
  → 可投稿顶级会议（ACL/ICML/NeurIPS）

后续方向:
  - 扩展到全部24个任务的191个MR
  - 探索MR知识的层级结构（MR taxonomy learning）
  - 研究大规模预训练中嵌入MR知识的可能性
```

**场景S2: 中等MR学习 + 有限泛化**（有价值但受限）

```
观察:
  - 微调后MRV从18%降至8-10%（~50%相对降低）
  - 同类别未见MR的MRV下降微弱（~10%）
  - 跨任务迁移失败
  - 原始任务准确率轻微下降（-2%）
  - MR-as-Data和MR-as-Instruction效果相当

解释:
  - LLM可以学习特定的MR模式，但主要靠记忆而非理解
  - MR知识是任务绑定的，泛化能力有限
  → 仍然有用但价值受限：特定MR的鲁棒性提升是确定的
  → 可投稿NLP workshop或软件工程会议

后续方向:
  - 聚焦特定高价值MR（如negation_flip）的深度优化
  - 探索如何提升MR泛化（更多样的训练数据？更大的模型？）
```

**场景S3: MR学习失败或负面**（重要负面结果）

```
观察:
  - 微调后MRV下降<10%或反而上升
  - 原始任务准确率显著下降（>5%）
  - 模型出现"过度泛化"——对所有输入输出同一标签
  - 小模型完全无法学习

解释:
  - MR逻辑可能过于复杂，超出了当前规模LLM的微调可学习范围
  - MR变换数据中的噪声淹没了信号
  - 需要更基础的研究：MR的形式化表示、更高质量的数据等
  → 负面结果但有发表价值（"On the (Un)learnability of MRs"）
  → 可投稿到更偏分析的venue

后续方向:
  - 分析失败原因：是数据问题还是学习问题？
  - 转向更简单的MR形式（如二元约束vs多元约束）
  - 考虑是否需要专门的架构设计（而非通用LLM微调）
```

### 8.2 结果解释框架

```
                    MRV显著下降？
                   /            \
                 Yes              No
                /                  \
        泛化到未见MR？          原始任务性能保持？
       /           \            /            \
     Yes           No         Yes            No
      |             |          |              |
   场景S1        场景S2      场景S3         场景S3
  (强证据)     (中等证据)  (MR不可学)   (MR有负面作用)
```

### 8.3 与现有工作的预期对比

| 方法 | NLI Acc | MRV | 跨任务泛化 | 计算成本 |
|------|:------:|:---:|:---------:|:------:|
| Baseline (无微调) | ~85% | ~18% | N/A | 0 |
| MetTrain (原始, BERT分类器) | ~87% | ~12% | Limited | 中 |
| 本工作 S1: MR-as-Data | ~86% | ~10% | 待测 | 中 |
| 本工作 S2: MR-as-Pair | ~86% | ~7% | 待测 | 中 |
| 本工作 S3: MR-as-Instruction | ~85% | ~4% | 待测 | 中 |

*注：数值为基于MetTrain论文和LLMORPH论文的初步估计*

### 8.4 潜在贡献总结

无论实验结果是正面还是负面，本研究的贡献包括：

1. **首个MR可学习性的系统研究**：填补了蜕变测试与LLM微调交叉领域的空白
2. **MR微调策略的统一框架**：为后续研究提供方法学基础
3. **LLMORPH MR目录的训练化应用**：将MR从测试工具提升为训练基础设施
4. **MR泛化层级的实验证据**：揭示MR知识在模型中的组织方式
5. **开放资源**：MR训练数据、微调代码、评测benchmark

---

## 9. 参考文献

### 核心参考文献

1. **MetTrain**: *"MetTrain: A Metamorphic-Relation-Driven Training Framework for Natural Language Inference"* — 本项目的前置工作，首次将MR用于LLM训练信号。

2. **LLMORPH**: *"Metamorphic Testing of Large Language Models for Natural Language Processing"* (ICSME 2025) — 191个MR的系统目录，36个MR的自动化实现，561K次测试的实证研究。

3. **METAL**: He, Pinjia, et al. "METAL: Metamorphic testing for natural language processing." ICSE 2020. — 最早的NLP蜕变测试系统化工作之一。

4. **CheckList**: Ribeiro, Marco Tulio, et al. "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList." ACL 2020. — NLP模型的行为测试框架，与MR测试互补。

5. **R-Drop**: Wu, Lijun, et al. "R-Drop: Regularized Dropout for Neural Networks." NeurIPS 2021. — MetTrain使用的KL散度正则化方法。

### 相关方法论文献

6. **FLAN**: Wei, Jason, et al. "Finetuned Language Models Are Zero-Shot Learners." ICLR 2022. — 指令微调的经典工作。

7. **QLoRA**: Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023. — 本工作将使用的微调技术。

8. **FixMatch**: Sohn, Kihyuk, et al. "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." NeurIPS 2020. — 一致性正则化的代表性工作。

9. **UDA**: Xie, Qizhe, et al. "Unsupervised Data Augmentation for Consistency Training." NeurIPS 2020. — 数据增强一致性训练。

10. **Self-Training**: Xie, Qizhe, et al. "Self-Training With Noisy Student Improves ImageNet Classification." CVPR 2020. — 自训练的噪声鲁棒性视角。

### 蜕变测试基础文献

11. Chen, Tsong Yueh, et al. "Metamorphic Testing: A Review of Challenges and Opportunities." ACM Computing Surveys, 2018. — 蜕变测试的综述。

12. Segura, Sergio, et al. "A Survey on Metamorphic Testing." IEEE TSE, 2016. — 软件工程领域的蜕变测试综述。

### 评估相关

13. Wang, Alex, et al. "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding." ICLR 2019. — NLI评估基准。

14. Bowman, Samuel R., et al. "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference." EMNLP 2015. — SNLI数据集。

15. Williams, Adina, et al. "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference." NAACL 2018. — MNLI数据集。

---

## 附录

### A. 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 蜕变关系 | Metamorphic Relation (MR) | 描述程序多次执行间输入输出应满足的关系 |
| MR违反率 | MR Violation Rate (MRV) | 模型输出违反MR约束的样本占总样本的比例 |
| 源样本 | Source | MR变换的输入样本 |
| 跟随样本 | Follow-up | MR变换后的输出样本 |
| Flip类MR | Flip MR | 变换后标签应翻转的MR（如negation_flip） |
| Preserve类MR | Preserve MR | 变换后标签应保持的MR（如synonym_replacement） |
| MR可学习性 | MR Learnability | LLM通过微调降低MRV的能力 |
| MR泛化 | MR Generalization | 在训练MR上学到的知识向未见MR迁移的能力 |

### B. LLMORPH MR编号速查

本工作重点关注的LLMORPH MR（共14个，覆盖3个层级和跨任务特性）：

```
P0 (核心实验, 7个):
├── MR-1  : Synonym Substitution (词汇层, preserve)
├── MR-5  : Active-Passive Voice Switch (句法层, preserve)
├── MR-71 : Premise-Hypothesis Swap (句法层, NLI-specific, flip)
├── MR-72 : Negation Flip (语义层, NLI-specific, flip)
├── MR-73 : Contradiction Condition Addition (语义层, NLI-specific, flip)
├── MR-102: Negation Insertion (语义层, cross-task, flip)
└── MR-9  : Number Replacement (语义层, cross-task, preserve)

P1 (扩展实验, 7个):
├── MR-2  : Keyword Replacement (词汇层, preserve)
├── MR-6  : Sentence Reorganization (句法层, preserve)
├── MR-8  : Case Transformation (词汇层, preserve)
├── MR-10 : Named Entity Replacement (词汇层, preserve)
├── MR-74 : Degree Modifier Modification (语义层, NLI-specific, preserve)
├── MR-75 : Sentiment Polarity Modification (语义层, NLI-specific, flip)
└── MR-3  : Comparative Adjective Substitution (词汇层, preserve)
```

### C. 代码模块规划

```
LLMTrain/
├── scripts/
│   ├── test_llm.py              # [已有] LLM预测
│   ├── evaluate.py              # [已有] 评估报告
│   ├── mrv.py                   # [已有] MRV计算
│   ├── run_all.sh               # [已有] 批量运行
│   ├── generate_mr_data.py      # [新增] MR变换数据生成
│   │   ├── MRRegistry           # MR注册表（LLMORPH MR定义）
│   │   ├── TransformExecutor    # MR变换执行器（调用LLM API）
│   │   ├── QualityValidator     # 变换质量验证
│   │   └── DataExporter         # 统一格式导出
│   ├── finetune_mr.py           # [新增] MR微调（统一使用标准SFT）
│   │   ├── MRDataFormatter       # 数据格式化器（支持S1-S3三种策略的prompt模板）
│   │   ├── MRDataCollator        # 数据整理器
│   │   ├── MRTrainer             # 标准SFT训练器（基于TRL SFTTrainer）
│   │   └── MREvaluator           # 微调过程中的评测
│   └── run_mr_experiments.sh    # [新增] 批量实验脚本
├── data/
│   ├── nli/                     # [已有] NLI MR数据
│   ├── sa/                      # [新增] SA MR数据
│   └── qa/                      # [新增] QA MR数据
├── output/                       # [已有] 输出目录
├── configs/                      # [新增] 实验配置
│   ├── mr_registry.json         # MR定义配置
│   ├── finetune_strategies.yaml # 微调策略配置
│   └── experiment_grid.yaml     # 实验网格配置
├── .research/
│   └── llmorph_mr_finetuning_report.md  # [本文档]
└── requirements.txt             # [更新] 增加transformers, peft, etc.
```

---

> **文档版本**: v1.0
> **作者**: [项目研究者]
> **下次评审**: Phase 1 完成后（预计2026-06-15）
> **关联文档**: [MetTrain论文], [LLMORPH论文], [CLAUDE.md](../CLAUDE.md)
