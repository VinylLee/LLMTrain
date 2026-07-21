# MR-as-Instruction 微调实验整体规划

> **仓库**：https://github.com/VinylLee/LLMTrain  
> **参考论文**：已上传的 `LLMORPH.pdf`  
> **核心原则**：先保证实验可解释、无数据泄漏、可复现，再扩大实验规模。不要只实现一个 `--mr-instruction` 布尔开关后直接跑全量实验。

---

## 1. 研究目标与研究边界

### 1.1 核心研究问题

研究训练阶段提供蜕变关系（Metamorphic Relation, MR）信息，是否能够：

1. 提高微调模型在普通 NLI 测试提示下的准确率；
2. 提高模型在 MR 增强测试集上的鲁棒性；
3. 降低 source/follow-up 样本之间的 MR violation；
4. 提高学习速度或样本效率，即以更少训练 step、样本或 token 达到相同性能。

### 1.2 方法定位

该方法应表述为：

> **MR-aware instruction fine-tuning with privileged relational information**

它不是严格意义上的“训练阶段执行蜕变测试”。在 LLMORPH 的正式 MT 定义中，source 和 follow-up 的任务 prompt 应保持一致，输入变换只作用于任务文本。这里有意只在训练时向增强样本提供额外关系信息，因此属于一种利用 MR 的训练监督方法。

### 1.3 测试阶段不可改变的原则

测试阶段继续使用统一的普通 NLI prompt，不向模型提供：

- `mr_id`；
- MR 描述；
- 原始样本；
- 原始标签；
- 预期标签变化。

这样才能验证模型是否将训练阶段的 MR 信息内化，而不是依赖测试时提示。

---

## 2. 当前方案中必须先解决的风险

在添加实验配置前，先修复以下问题。它们会直接影响科研结论。

### 2.1 必须按 `pair_id` 分组拆分 train/validation

当前转换脚本是先逐行打乱，再按样本行数切分。这样可能把同一个 `pair_id` 的 source 和 follow-up 分到不同 split，造成：

- 训练／验证数据泄漏；
- 高度相似文本跨 split；
- 增强样本找不到原始对照；
- fallback 是否触发由随机拆分决定；
- baseline 与 MR-instruction 的验证集不可严格比较。

**要求：**

- 使用完整 pair group 作为最小拆分单元；
- 同一个 `pair_id` 的所有条目必须全部进入 train 或全部进入 validation；
- 多输入文件时，分组键优先使用 `(source_file, pair_id)`，避免不同文件中 ID 冲突；
- 没有 `pair_id` 的普通数据，每一行视为独立 group；
- 拆分后再分别调用转换函数；
- train 可按实验模式生成 instruction；
- validation 强制使用普通 instruction。

### 2.2 主实验不能直接暴露目标标签

原始提案中的以下信息会形成明显标签捷径：

```text
Original Label: entailment
Transformation: ... result in a neutral relation
```

或者：

```text
Transformation: composite_neutral
Transformation: adding_contradiction
```

问题包括：

- `Original Label + MR 类型` 可直接映射目标标签；
- `neutral`、`contradiction` 等词直接出现在 instruction；
- `preserve the original relation` 配合原始标签即可复制答案；
- 最终性能提升不能证明模型理解 current premise/hypothesis。

因此，**主实验不应包含原始标签，也不应直接陈述目标 NLI 标签**。强标签信息只能作为 upper-bound 对照实验。

### 2.3 当前样本不要在 instruction 中重复

Alpaca 的 `input` 已经包含当前 premise/hypothesis。不要再在 instruction 中放：

```text
Current: Premise: ... | Hypothesis: ...
```

否则：

- 当前样本被重复编码；
- baseline 与实验组不再只差 MR 信息；
- token 长度增加；
- `cutoff_len=512` 下更容易截断。

### 2.4 MR 类型的“操作描述”和“输出关系”必须分开

不要把以下两类信息混在一个字典中：

1. **Input transformation / operation**：做了什么文本变换；
2. **Output relation / effect**：标签应该保持、改变或变成 neutral。

代码层面必须使用两个独立结构，以支持消融实验。

### 2.5 不允许静默处理未知 `mr_id`

正式实验中，未知 MR、缺失 source、一个 pair 有多个 source 等情况应在转换前暴露出来，而不是悄悄 fallback。

---

## 3. 推荐的实验变量设计

不要继续只使用单个布尔字段：

```json
"mr_instruction": true
```

建议改成一个明确的枚举字段：

```json
"mr_instruction_mode": "pair_operation"
```

### 3.1 支持的模式

| 模式 | Reference P/H | Original label | MR operation | MR output effect | 作用 |
|---|---:|---:|---:|---:|---|
| `none` | 否 | 否 | 否 | 否 | 现有 baseline |
| `operation_only` | 否 | 否 | 是 | 否 | 检验 MR 操作描述本身 |
| `pair_only` | 是 | 否 | 否 | 否 | 检验原始对照本身 |
| `pair_operation` | 是 | 否 | 是 | 否 | **主实验** |
| `shuffled_operation` | 是 | 否 | 随机错误描述 | 否 | 负对照 |
| `relation_aware` | 是 | 否 | 是 | 是 | 输出关系信息上界 |
| `full_oracle` | 是 | 是 | 是 | 是 | 强标签信息上界 |

### 3.2 优先级

#### Tier 0：工程验证

先实现并验证：

- `none`
- `pair_operation`
- `full_oracle`

目的不是形成论文结论，而是确认整个 pipeline 正确。

#### Tier 1：核心论文实验

正式主实验至少包含：

- `none`
- `operation_only`
- `pair_only`
- `pair_operation`
- `shuffled_operation`

该矩阵能够区分：

- 提升来自额外 reference；
- 提升来自 MR 描述；
- 提升来自 reference 与 MR 的组合；
- 提升是否只是 instruction 变长或模板化文本造成。

#### Tier 2：信息上界

预算允许时增加：

- `relation_aware`
- `full_oracle`

这两组只能解释为 privileged/oracle information upper bound，不能作为“模型理解 MR”的主要证据。

---

## 4. MR 描述结构

### 4.1 安全的操作描述

主实验使用仅描述文本操作的字典，例如：

```python
MR_OPERATION_DESCRIPTIONS = {
    "adding_contradiction":
        "Additional information was inserted, and the inserted content is incompatible "
        "with part of the reference text.",

    "antonym_substitution":
        "One or more content words were replaced with context-appropriate antonyms.",

    "negation_flip":
        "A negation marker was added, removed, or reversed in one component.",

    "synonym_replacement":
        "One or more words were replaced with context-appropriate synonyms.",

    "pronoun_substitution":
        "A noun phrase or pronoun was replaced by a coreferential form.",

    "voice_switch":
        "A sentence was rewritten between active and passive voice.",

    "conditional_clause":
        "A conditional clause was added to one component.",

    "uninformative":
        "Additional content with little direct relevance was inserted.",

    "composite_inv":
        "Multiple controlled transformations were applied.",

    "composite_flip":
        "Multiple controlled transformations were applied.",

    "composite_neutral":
        "Multiple controlled transformations were applied.",

    "none": "",
}
```

注意：

- 主实验不要把 raw `mr_id` 输出到 instruction；
- 尤其不要显示 `composite_neutral`、`composite_flip` 等带标签暗示的名称；
- composite 数据若有实际组成 MR 字段，优先列出组成操作；
- 若没有组成信息，三个 composite 在主实验中使用相同的中性描述；
- `voice_switch` 作为 label-invariant 类型处理是合理的；
- `summary` 不属于当前 MR 集，不加入映射，但必须先统计真实数据中的所有 `mr_id`。

### 4.2 输出关系描述

另建结构，仅供 `relation_aware` 和 `full_oracle` 使用：

```python
MR_RELATION_EFFECTS = {
    "adding_contradiction": "...",
    "antonym_substitution": "...",
    "negation_flip": "...",
    "synonym_replacement": "...",
    "pronoun_substitution": "...",
    "voice_switch": "...",
    "conditional_clause": "...",
    "uninformative": "...",
    "composite_inv": "...",
    "composite_flip": "...",
    "composite_neutral": "...",
}
```

不要让主实验代码路径读取该字典。

### 4.3 无法完全消除的语义提示

某些 MR 的操作语义本身就与 NLI 标签高度相关，例如“添加不兼容信息”。这并不一定是错误，因为研究目标就是提供 MR 语义，但必须通过以下方式判断模型是否只学了 shortcut：

- `shuffled_operation` 负对照；
- 每 MR 分析；
- original test 与 MR test 同时评估；
- 可选的 instruction-only / input permutation 诊断；
- 不使用原始标签作为主实验输入。

---

## 5. Instruction 模板

### 5.1 普通 instruction

原始样本、validation 和所有 test 均保持：

```text
Determine the natural language inference relation between the premise and hypothesis.
Answer with exactly one label: entailment, neutral, or contradiction.
```

### 5.2 `operation_only`

```text
Transformation information:
{operation_description}

Determine the natural language inference relation between the current premise and hypothesis.
Answer with exactly one label: entailment, neutral, or contradiction.
```

### 5.3 `pair_only`

```text
Reference sample:
Premise: "{original_premise}"
Hypothesis: "{original_hypothesis}"

Determine the natural language inference relation between the current premise and hypothesis.
Answer with exactly one label: entailment, neutral, or contradiction.
```

### 5.4 `pair_operation`：主实验

```text
Reference sample:
Premise: "{original_premise}"
Hypothesis: "{original_hypothesis}"

Transformation applied to create the current sample:
{operation_description}

Determine the natural language inference relation between the current premise and hypothesis.
Answer with exactly one label: entailment, neutral, or contradiction.
```

### 5.5 `full_oracle`

仅用于上界：

```text
Reference sample:
Premise: "{original_premise}"
Hypothesis: "{original_hypothesis}"
Reference label: {original_label}

Transformation:
{operation_description}

Expected relation effect:
{relation_effect}

Determine the natural language inference relation between the current premise and hypothesis.
Answer with exactly one label: entailment, neutral, or contradiction.
```

### 5.6 Alpaca `input`

所有模式都只在 `input` 中放一次当前样本：

```text
Premise: "{current_premise}"
Hypothesis: "{current_hypothesis}"
```

---

## 6. 数据预检与规范化

在转换前增加统一 preflight。

### 6.1 `mr_id` 规范化

```python
def normalize_mr_id(value):
    if value is None:
        return "none"
    return str(value).strip().lower().replace("-", "_")
```

### 6.2 必须输出的统计

转换开始前打印并保存：

- 总样本数；
- pair group 数；
- 每个 `mr_id` 的样本数；
- 每个标签的样本数；
- 每个 MR × 标签的交叉分布；
- 每组样本数的 min / median / max；
- 缺失 `pair_id` 的数量；
- 没有 source 的增强组数量；
- 一个 group 中有多个 `mr_id=none` 的数量；
- 未知 `mr_id` 列表；
- fallback 数量和比例。

### 6.3 严格模式

新增：

```bash
--strict-pairing
```

正式 MetTrain 实验默认开启。以下情况直接失败：

- 未知 `mr_id`；
- 增强样本找不到 source；
- 一个 group 有多个不一致的 source；
- 同一个 group 被拆入 train 和 validation；
- 主实验使用了原始 label 或 relation-effect 文本。

兼容旧数据时可显式关闭 strict，但必须在日志中显示 warning。

---

## 7. Group-aware split 设计

### 7.1 推荐函数

增加类似函数：

```python
def build_pair_groups(samples, source_ids=None):
    ...

def split_pair_groups(
    groups,
    val_ratio,
    seed,
    stratify=False,
):
    ...
```

### 7.2 分组键

优先：

```python
group_key = (source_file_id, normalized_pair_id)
```

无 `pair_id`：

```python
group_key = (source_file_id, f"__row_{row_index}")
```

### 7.3 拆分流程

```python
groups = build_pair_groups(all_samples)
train_groups, val_groups = split_pair_groups(
    groups,
    val_ratio=args.val_ratio,
    seed=args.seed,
)

train_samples = flatten(train_groups)
val_samples = flatten(val_groups)

train_converted, _ = convert_to_alpaca(
    train_samples,
    instruction_text,
    label_names,
    mr_instruction_mode=args.mr_instruction_mode,
    strict_pairing=args.strict_pairing,
)

val_converted, _ = convert_to_alpaca(
    val_samples,
    instruction_text,
    label_names,
    mr_instruction_mode="none",
    strict_pairing=False,
)
```

### 7.4 共享 split manifest

为了让不同 instruction 变体严格使用相同样本，新增：

```bash
--split-manifest path/to/split_manifest.json
--write-split-manifest path/to/split_manifest.json
```

manifest 至少包含：

```json
{
  "seed": 42,
  "val_ratio": 0.05,
  "group_key_version": 1,
  "train_group_ids": [],
  "val_group_ids": [],
  "train_sample_count": 0,
  "val_sample_count": 0
}
```

**推荐流程：**

1. baseline 第一次转换时生成 manifest；
2. 同一 dataset/target/seed 下的所有 MR-instruction 变体复用它；
3. 若 sampled pair IDs 与 manifest 不一致，直接报错。

最低可接受方案是相同 seed 下确定性拆分并对 manifest hash 做断言，但不应只依赖“理论上随机结果相同”。

---

## 8. `convert_nli_to_ft.py` 修改计划

### Step 8.1：重构配置参数

新增：

```bash
--mr-instruction-mode {
  none,
  operation_only,
  pair_only,
  pair_operation,
  shuffled_operation,
  relation_aware,
  full_oracle
}
--strict-pairing
--split-manifest PATH
--write-split-manifest PATH
--report-token-lengths
```

可保留旧的 `--mr-instruction` 作为兼容 alias，但应映射到：

```text
pair_operation
```

并打印 deprecation warning。

### Step 8.2：拆分纯函数

至少拆成：

```python
normalize_mr_id()
validate_mr_samples()
build_pair_groups()
split_pair_groups()
build_original_map()
build_instruction_for_sample()
convert_to_alpaca()
save_split_manifest()
load_split_manifest()
```

避免把 MR 拼接、split 和文件写入全塞进 `main()`。

### Step 8.3：建立 source map

在 train 子集内部建立：

```python
group_key -> original_sample
```

正式 MetTrain + strict 模式下，增强样本 source 缺失应为错误，不应静默 fallback。

### Step 8.4：实现 shuffled negative control

`shuffled_operation` 必须：

- 使用固定 seed；
- 保持样本和标签不变；
- 只打乱操作描述；
- 确保尽可能不分配回原 MR；
- 保存 `true_mr_id` 与 `assigned_description_id` 到转换报告；
- 主训练 JSON 中可不保留这些元数据，避免影响 LLaMA Factory；
- 可选择全局打乱和同 MR-family 内打乱两种诊断，第一版先做全局打乱。

### Step 8.5：输出转换报告

每次转换额外保存：

```text
data/ft_datasets/<experiment>/conversion_report.json
data/ft_datasets/<experiment>/split_manifest.json
```

报告包含：

- mode；
- source input files；
- seed；
- sample/group counts；
- MR/label 分布；
- missing/unknown/fallback 统计；
- instruction 长度统计；
- tokenizer token 长度统计；
- 超过 `cutoff_len` 的比例；
- manifest hash。

---

## 9. Token 长度与截断控制

当前批量训练配置使用 `cutoff_len: 512`。加入 reference 后，MR 组会明显变长。

### 9.1 必须统计

使用实际训练 tokenizer，分别对 baseline 和各 MR 模式统计：

- P50；
- P90；
- P95；
- P99；
- max；
- `>512` 数量与比例；
- 被截断样本的 MR 分布；
- 被截断样本的标签分布。

### 9.2 处理规则

在看完统计前，不要直接提高 cutoff。

若截断比例明显：

1. 优先删除重复 current P/H；
2. 压缩模板文字；
3. 对 reference 做明确的长度策略；
4. 再评估是否将 cutoff 调到 768 或 1024；
5. 若改变 cutoff，baseline 与所有变体必须同步改变。

### 9.3 公平性报告

至少同时报告：

- optimizer steps；
- samples seen；
- tokens seen；
- wall-clock time；
- GPU-hours。

相同 epoch 不代表相同训练计算量。

---

## 10. `experiments_config.json` 设计

### 10.1 先清理 RTE

删除：

- `mettrain_rte_2490_gemma3_4b`
- `original_rte_2490_gemma3_4b`

同时检查：

- `run_batch_experiments.py` 顶层 `ALL_DS`；
- summary 函数内的局部数据集列表；
- 文件头部使用示例；
- “8 个实验／5 个测试集”等旧注释；
- `PROJECT_OVERVIEW.md` 中的推荐命令和当前配置描述。

数据集列表最好从配置的 `test_sets` 动态派生，不要维护多个硬编码列表。

### 10.2 主实验命名

保留现有 baseline 名称不变。

将 `_mrinstr` 明确定义为主实验 `pair_operation`：

```json
{
  "name": "mettrain_mnlim_4413_gemma3_4b_mrinstr",
  "train_data": "...",
  "target": 4413,
  "task_type": "nli",
  "mr_instruction_mode": "pair_operation",
  "split_group_key": "pair_id",
  "strict_pairing": true,
  "ft_params": {
    "lr": 0.0003,
    "epochs": 3.0,
    "rank": 8,
    "batch": 4,
    "grad_accum": 8
  }
}
```

为以下四个 MetTrain 实验增加 `_mrinstr`：

- `mettrain_mnlim_4413_gemma3_4b_mrinstr`
- `mettrain_sick_4439_gemma3_4b_mrinstr`
- `mettrain_sick_11176_gemma3_4b_mrinstr`
- `mettrain_snli_5340_gemma3_4b_mrinstr`

### 10.3 Ablation 命名

建议：

```text
..._mrop                 -> operation_only
..._paironly             -> pair_only
..._mrinstr              -> pair_operation
..._mrinstr_shuffled     -> shuffled_operation
..._mrinstr_relation     -> relation_aware
..._mrinstr_fulloracle   -> full_oracle
```

不要一次在所有数据集上复制全部模式。先在一个 pilot 数据集上筛选，再扩大。

### 10.4 推荐增加 cohort 概念

不同 instruction 变体共享相同数据 cohort：

```json
"cohort_id": "mettrain_mnlim_4413"
```

batch runner 根据 `cohort_id + seed` 复用：

- sampled data；
- selected pair IDs；
- split manifest。

这样避免每个变体独立采样、独立拆分。

---

## 11. `run_batch_experiments.py` 修改计划

### Step 11.1：数据集列表动态派生

移除所有硬编码 RTE 列表，统一从：

```python
config["test_sets"]["original"].keys()
config["test_sets"]["mr"].keys()
```

派生。

### Step 11.2：Step 2 convert 命令

根据 experiment 字段追加：

```bash
--mr-instruction-mode <mode>
--strict-pairing
--split-manifest <shared_manifest>
```

若 manifest 尚不存在，则第一个 cohort variant 使用：

```bash
--write-split-manifest <shared_manifest>
```

Dry-run 必须完整显示这些参数。

### Step 11.3：保存实验元信息

`experiment_meta.json` 增加：

```json
{
  "mr_instruction_mode": "pair_operation",
  "cohort_id": "mettrain_mnlim_4413",
  "sample_manifest": "...",
  "split_manifest": "...",
  "split_manifest_sha256": "...",
  "cutoff_len": 512,
  "conversion_report": "..."
}
```

### Step 11.4：训练配置可参数化

不要继续把以下值全部写死：

- `cutoff_len`
- `save_steps`
- `eval_strategy`
- `eval_steps`
- `save_total_limit`

至少允许 experiment 或全局配置覆盖。

### Step 11.5：resume 安全性

当 mode、manifest hash、cutoff 或转换模板版本改变时，不允许错误地复用旧 progress 状态。

在 progress key 或 meta 中加入：

```text
data_signature
instruction_template_version
```

检测不一致时要求重新执行 convert/finetune。

---

## 12. 测试输出与评估修改

### 12.1 测试 prompt 保持不变

`scripts/test_mettrain_experiment.py` 中普通三分类 prompt 不增加 MR 信息。

### 12.2 每条结果必须保存元数据

当前只保存：

```json
{"index": 1, "gold": "...", "pred": "...", "correct": true}
```

修改为至少：

```json
{
  "index": 1,
  "dataset": "snli",
  "test_type": "mr",
  "mr_type": "synonym_replacement",
  "pair_id": "...",
  "premise": "...",
  "hypothesis": "...",
  "gold": "entailment",
  "pred": "neutral",
  "correct": false
}
```

若原始记录还有以下字段，也应尽量保留：

- source/current 标记；
- original sample ID；
- transformation components；
- `_source` 文件名。

`collect_mr_samples()` 当前已经构造 `mr_types`，必须将其传入 `test_dataset()` 并逐条写出，不能丢弃。

### 12.3 核心指标

每个 experiment × seed 报告：

1. Original test accuracy；
2. MR test overall accuracy；
3. 每个 MR 的 accuracy；
4. macro-average over MRs；
5. original-to-MR robustness gap；
6. 每个标签的 precision / recall / F1；
7. 无法解析输出的比例；
8. 若 test 数据支持 source/follow-up 对齐：MR consistency / violation rate。

### 12.4 跨 seed 汇总

正式实验至少 3 个 seeds：

```text
42, 43, 44
```

报告：

- mean；
- standard deviation；
- 每 seed paired delta；
- 相对 baseline 的平均提升；
- 可选 paired bootstrap confidence interval。

不要只用一个 seed 得出科研结论。

---

## 13. “更快理解 MR”的实验定义

仅比较最终 epoch 结束后的结果，不能支持“学得更快”。

### 13.1 学习曲线 checkpoint

建议评估：

```text
10%, 25%, 50%, 75%, 100% training steps
```

或者根据总 step 选择固定 checkpoint。

### 13.2 曲线指标

绘制：

- Original accuracy vs optimizer steps；
- MR accuracy vs optimizer steps；
- per-MR macro accuracy vs optimizer steps；
- MR violation rate vs optimizer steps；
- accuracy vs cumulative training tokens。

### 13.3 速度定义

至少使用一个预先定义的指标：

- 达到 baseline 最终性能所需的 step；
- 达到某个固定 accuracy 阈值所需的 step；
- learning-curve AUC；
- 固定 step 下的性能；
- 固定 token budget 下的性能。

### 13.4 两种公平性协议

#### 协议 A：相同样本／step

保持：

- 相同 sampled groups；
- 相同 batch；
- 相同 optimizer steps；
- 相同超参数。

用于回答常规训练设置下是否提升。

#### 协议 B：相同 token budget

尽可能控制累计训练 token 一致，用于回答是否真正提高 token efficiency。

若第一阶段难以实现协议 B，至少记录 token 数并明确将其作为后续实验。

---

## 14. 单元测试与验收标准

新增 `tests/test_convert_nli_to_ft.py` 或等价测试。

### 14.1 Split 测试

- [ ] 同一 pair 不跨 train/validation；
- [ ] train group IDs 与 val group IDs 交集为空；
- [ ] 相同 seed 得到相同 split；
- [ ] 不同 instruction mode 复用同一 manifest；
- [ ] 无 pair_id 数据仍能正常拆分。

### 14.2 Instruction 测试

- [ ] `mr_id=none` 始终使用普通 instruction；
- [ ] validation 所有样本均为普通 instruction；
- [ ] `pair_operation` 有 reference 和 operation；
- [ ] `pair_operation` 不含 original label；
- [ ] `pair_operation` 不输出 raw `mr_id`；
- [ ] `pair_operation` 不含 relation-effect 模板；
- [ ] 当前 P/H 只出现在 `input` 一次；
- [ ] `full_oracle` 明确包含上界信息；
- [ ] shuffled 描述与真实 MR 不同。

### 14.3 数据完整性测试

- [ ] 未知 MR 在 strict 模式报错；
- [ ] 增强样本缺 source 在 strict 模式报错；
- [ ] 多 source group 在 strict 模式报错；
- [ ] conversion report 统计正确；
- [ ] fallback 在正式 MetTrain 数据中为 0。

### 14.4 Pipeline 测试

- [ ] dry-run convert 命令包含正确 mode；
- [ ] `_mrinstr` 指向 `pair_operation`；
- [ ] test prompt 没有 MR 信息；
- [ ] MR 逐条结果包含 `mr_type`；
- [ ] RTE 不再出现在 experiment、测试列表或 summary；
- [ ] resume 不会复用签名不匹配的旧产物。

---

## 15. 推荐的分阶段执行顺序

### 阶段 A：代码审计与数据审计

1. 检索当前真实 `mr_id` 集合；
2. 检查每个 pair 的 source 数量；
3. 检查 composite 是否有 component 信息；
4. 检查训练和 MR test 文件是否保存 `pair_id`；
5. 检查实际 token 长度；
6. 记录当前 baseline 产物和配置，避免覆盖。

**退出条件：** 数据 schema、MR 枚举和 pair 完整性均有报告。

### 阶段 B：安全重构

1. 清理 RTE；
2. 实现 group-aware split；
3. 实现 split manifest；
4. 重构 instruction mode；
5. 实现 preflight 和 strict pairing；
6. 增加 conversion report；
7. 增加测试输出元数据；
8. 补充单元测试。

**退出条件：** 所有验收测试通过。

### 阶段 C：最小 smoke test

使用：

```text
mettrain_mnlim_4413_gemma3_4b
seed = 42
```

运行：

- `none`
- `pair_operation`
- `full_oracle`

只跑一个 seed，用于确认：

- convert 文件格式正确；
- train/val instruction 正确；
- 训练能启动；
- test prompt 保持普通；
- per-MR 输出可统计；
- 无截断异常或 fallback。

**退出条件：** 工程流程无错误，不对效果下结论。

### 阶段 D：Pilot 消融

仍选择一个较小／成本可控的数据集，运行 3 seeds：

- `none`
- `operation_only`
- `pair_only`
- `pair_operation`
- `shuffled_operation`

根据 pilot 判断：

- `pair_operation` 是否稳定优于 baseline；
- shuffled control 是否明显更差；
- 是否存在 original accuracy 损失；
- 哪些 MR 受益或退化；
- 是否值得扩大到全部数据集。

### 阶段 E：全数据集核心实验

对以下 MetTrain 设置运行通过 pilot 的模式：

- MNLI 4413；
- SICK 4439；
- SICK 11176；
- SNLI 5340。

每组至少 3 seeds。原始数据实验保持不变，作为对照。

### 阶段 F：学习速度实验

只对最有代表性的 dataset × mode 做 checkpoint 曲线，避免一开始扩大计算量。

### 阶段 G：上界实验

最后补充：

- `relation_aware`
- `full_oracle`

将其明确标注为 upper bound，不与无标签泄漏主实验混为一谈。

---

## 16. 结果解释规则

### 16.1 可以支持“MR 信息有效”的证据

较强证据组合：

- `pair_operation > none`；
- `pair_operation > pair_only`；
- `pair_operation > operation_only`；
- `pair_operation > shuffled_operation`；
- 多 seed 稳定；
- original accuracy 不明显下降；
- 多个 MR 类型均受益，而不是单一标签驱动；
- 普通 test prompt 下仍提升。

### 16.2 不能直接支持“模型理解 MR”的情况

以下结果只能谨慎解释：

- 只有 `full_oracle` 提升；
- MR 指令直接包含目标标签；
- shuffled MR 与正确 MR 表现相同；
- 提升只发生在 `composite_neutral` 等带标签名称的类别；
- MR test 提升但 original test 大幅下降；
- 只有一个 seed；
- 训练集和验证集有 pair 泄漏；
- MR 组训练 token 显著更多但未控制或报告。

### 16.3 可能的负结果也有价值

若 `pair_operation` 不提升，仍应分析：

- token 截断；
- MR 描述质量；
- MR 类型噪声；
- source/follow-up 对齐错误；
- 不同 MR 的效果互相抵消；
- 模型容量；
- instruction 与 test prompt 的分布差异；
- 数据集规模是否过小；
- operation semantics 是否过于模板化。

---

## 17. Claude 重新规划时的具体要求

请 Claude 在真正改代码前完成以下动作：

1. 读取当前仓库相关文件，不依赖旧行号：
   - `experiments_config.json`
   - `scripts/convert_nli_to_ft.py`
   - `scripts/sample_mettrain_pairid.py`
   - `scripts/run_batch_experiments.py`
   - `scripts/test_mettrain_experiment.py`
   - `scripts/summarize_seed_experiments.py`
   - `PROJECT_OVERVIEW.md`
   - 相关 tests
2. 检查真实训练数据 schema 和所有 `mr_id`；
3. 明确哪些需求是代码修改、哪些是实验配置、哪些是评估修改；
4. 给出按 commit 或按阶段拆分的实施 plan；
5. 每个阶段列出：
   - 修改文件；
   - 新增函数／参数；
   - 兼容性影响；
   - 测试方法；
   - 验收标准；
6. 优先完成 group split、label shortcut 控制和 per-MR 输出；
7. 不要直接创建所有全量实验配置并开跑；
8. 先完成 smoke test，再决定是否扩展到 Tier 1/Tier 2；
9. 所有新行为需要更新 `PROJECT_OVERVIEW.md` 或新增专门实验说明；
10. 不读取或提交 `.env`、密钥、模型缓存和大体积训练产物。

---

## 18. 最终交付物清单

Claude 的实施计划最终应覆盖以下交付物：

- [ ] RTE 遗留清理；
- [ ] group-aware split；
- [ ] shared split manifest；
- [ ] MR preflight validation；
- [ ] `mr_instruction_mode` 枚举；
- [ ] operation/effect 分离；
- [ ] 主实验无 original label；
- [ ] 主实验不输出 raw label-bearing MR ID；
- [ ] shuffled negative control；
- [ ] validation/test 普通 instruction；
- [ ] token 长度报告；
- [ ] per-MR 测试元数据；
- [ ] 多 seed 汇总；
- [ ] learning-curve 预留；
- [ ] 单元测试；
- [ ] dry-run 验证；
- [ ] smoke test；
- [ ] 文档更新。

---

## 19. 建议的最小结论目标

第一轮工作的目标不是证明“MR 一定提高鲁棒性”，而是建立一个能够可靠回答以下问题的实验框架：

> 在不向测试阶段提供 MR 信息、并避免 source/follow-up 泄漏和直接标签捷径的条件下，训练阶段提供原始对照与 MR 操作描述，是否能稳定提升模型的 NLI 泛化、MR 鲁棒性和学习效率？

只有当该框架完成后，再扩大模型、数据集和 seed 数量。
