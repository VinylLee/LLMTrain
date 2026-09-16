# RQ2 指令设计总结（v4）

> **这是什么**：RQ2（MR information representation）当前状态的**唯一入口说明**，面向要修改本项目的 agent。
> 读完这一份即可了解：用哪个数据集、模板怎么组装、Pair / Operation / Relation / Label 各自怎么设计、
> 每个 MR 的 description 是什么、以及已知的泄漏与对照风险。
>
> 生成时间：2026-09-16 ｜ 对应代码版本：`instruction_template_version = 4`
> 权威代码：`scripts/mr_instruction_design.py`（唯一真源）+ `scripts/convert_nli_to_ft.py`（渲染 / 校验）
>
> 相关文档：
> `RQ2/RQ2_INSTRUCTION_DESIGN.md`（共享矩阵 + v3 措辞，v3 已冻结）、
> `RQ2/RQ2_INSTRUCTION_DESIGN_V4.md`（v4 改了什么）、
> `RQ2/MR_RELATION_AUDIT.md`（v3 关系证据）、
> `RQ2/MR_RELATION_AUDIT_V4.md`（v4 关系证据，含 `flip` 收窄为 `E→C` 的理由）

> ⚠️ **实现状态**：本文描述的是**已实现**的 v4（8 core + 3 control + 1 diagnostic = 12 个 mode）。
> 之后提出的两个**尚未实现**的对照——`pair_wrong_operation_relation_matched` 与
> `pair_wrong_relation`——见 `.research/LLMTrain/RQ2_v4_final_experimental_configurations_zh_code_preserved.md`
> （该文件位于 `.gitignore` 的 `.research/` 下，不进版本库，但可直接读取）。在它们落地之前，
> 不要假设 registry 里已经存在这两个 mode。

---

## 一、当前使用的是哪一个数据集

### 1.1 训练数据（正式 v4）

| 项 | 值 |
|---|---|
| 文件 | `data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json` |
| 任务 | SNLI 三分类 NLI（entailment / neutral / contradiction） |
| 总行数 | **5340** |
| source 行（`mr_id=none`） | **2033** |
| MR follow-up 行 | **3307** |
| composite 行 | **486**，`component_mrs` 覆盖 **100%**（全部 arity 2） |
| 标签分布 | entailment 1168 / neutral 2574 / contradiction 1598 |
| 每行字段 | `premise, hypothesis, label, pair_id, is_syn, weight, mr_id, mr_type`（composite 额外 `component_mrs`） |
| 生成端 | `/home/ubuntu/MTrain/augment_snli_all_label_mrs.py`，`PROMPT_VERSION = snli-pair-aware-all-label-mrs-v3.3` |

**为什么是它**：这是唯一一个 composite 行带**有序** `component_mrs` provenance 的 artifact。v4 的 Operation 必须展开真实 component 序列，没有 provenance 就无法满足设计要求。

### 1.2 对照：冻结的 v3 数据（不要混用）

| 项 | v3 | v4 |
|---|---|---|
| 文件 | `…/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json` | `…-v3_3/augmented_data_all_label_mrs_v3_3_full.json` |
| 总行数 / source | 5340 / 2033 | 5340 / 2033 |
| composite | 1949 | 486 |
| `component_mrs` 覆盖 | **0%** | **100%** |
| 标签分布 | 1273 / 1855 / 2212 | 1168 / 2574 / 1598 |

两者 **source 行完全相同（2033/2033 精确匹配）**，但 composite 行 **0 条精确重合**，是两次不同生成。**v3 与 v4 的结果不能逐例比较**。

> v3 cohort 的 composite 无法补 provenance：该次生成的 generation log 已不存在（13 个候选日志中 12 个与 cohort composite 文本零重合，唯一"命中"是一条 `status: rejected` 记录），且旧生成器用 `random.randint(3, len(invariant_mrs))` 随机组合、从不落盘。

### 1.3 测试数据

| 项 | 值 |
|---|---|
| 文件 | `data/nli/mr_test_data_merged/snli.jsonl` |
| 总行数 | **22548**（source 9824 + follow-up 12724） |
| MR 覆盖 | `none` 9824、`uninformative` 2601、`negation_flip` 2384、`synonym_replacement` 1856、`conditional_clause` 1781、`antonym_substitution` 1471、`pronoun_substitution` 1332、`adding_contradiction` 996、`voice_switch` 303 |
| 字段 | `premise, hypothesis, label, pair_id, mr_id, mr_type, idx, is_source` |
| 注意 | 测试集**不含** composite MR |

### 1.4 训练超参（v4 config）

```jsonc
// RQ2/configs/rq2_snli_config_v4.json
{
  "instruction_template_version": 4,
  "require_composite_provenance": true,
  "model":   { "hub_id": "google/gemma-3-4b-it", "template": "gemma" },
  "data":    { "target_samples": 5340, "val_ratio": 0.05, "cutoff_len": 512 },
  "seeds":   [42, 43, 44],
  "training":{ "learning_rate": 0.0003, "epochs": 3.0, "rank": 8,
               "batch": 4, "grad_accum": 8, "test_batch_size": 32, "max_new_tokens": 10 },
  "cohorts_dir":  "RQ2/data/cohorts_v4",
  "converted_dir":"RQ2/data/converted_v4",
  "output_root":  "RQ2/output/gemma3_4b_nli_grounded_v4"
}
```

同一 seed 下所有 mode 共享 **一个 cohort 文件 + 一个 split manifest**，因此训练样本完全一致。

---

## 二、模板是怎么设计的

### 2.1 核心思路：不写整段模板，只写"哪些 block 存在"

**没有**为任何 mode 手写完整模板。每个 mode 只是一条**数据**，说明四个 block 是否存在、以及 payload 从哪来：

```python
# scripts/mr_instruction_design.py: MODE_SPECS
"pair_relation": {
    "pair": True, "operation": False, "relation": True, "label_anchor": False,
    "pair_source": "correct", "operation_source": "none", "relation_source": "correct",
    "role": "core",
},
```

模板由注册表**派生**（`build_mode_template_v4`），顺序固定写死在代码里：

```
[PAIR_BLOCK]          仅当 P=1
[LABEL_ANCHOR_BLOCK]  仅当 L=1（只有 full_oracle）
[OPERATION_BLOCK]     仅当 O=1
[RELATION_BLOCK]      仅当 R=1
[NLI_TASK_BLOCK]      永远有，且所有 mode 逐字相同
```

block 之间用 `"\n\n"` 连接。**唯一随 mode 变化的只有 block 的有无**——措辞、顺序、分隔符、NLI task 文本全局唯一。新增 mode 只需加一条 registry 记录，不用复制模板。

### 2.2 五个 block 常量

| Block | v3（冻结） | v4（当前） |
|---|---|---|
| PAIR | `Paired source sample:` + `<source_premise>` / `<source_hypothesis>` | `Paired source input:`（同上 XML 边界） |
| LABEL_ANCHOR | `Source label:\n{original_label}` | **与 v3 逐字相同** |
| OPERATION | `Input-transformation specification:\n{op}` | `Input transformation:\n{op}` |
| RELATION | `Output-relation specification:\n{rel}` | `Output relation:\n{rel}` |
| NLI_TASK | `Determine the natural language inference relation between the premise and hypothesis. Answer with exactly one label: entailment, neutral, or contradiction.` | **与 v3 逐字相同** |

v4 换成 `Paired source input` 是因为 `sample` 容易被模型读成 few-shot exemplar；`source input` 明确表达"这是当前 follow-up 的 source grounding"。

### 2.3 mode 矩阵（v4 = v3，完全一致）

| mode | P | O | R | L | grounding | role | payload 来源 |
|---|:-:|:-:|:-:|:-:|---|---|---|
| `none` | 0 | 0 | 0 | 0 | ungrounded | core | — |
| `operation_only` | 0 | 1 | 0 | 0 | ungrounded | core | op=correct |
| `relation_only` | 0 | 0 | 1 | 0 | ungrounded | core | rel=correct |
| `operation_relation` | 0 | 1 | 1 | 0 | ungrounded | core | op/rel=correct |
| `pair_only` | 1 | 0 | 0 | 0 | source_grounded | core | pair=correct |
| `pair_operation` | 1 | 1 | 0 | 0 | source_grounded | core | pair/op=correct |
| `pair_relation` | 1 | 0 | 1 | 0 | source_grounded | core | pair/rel=correct |
| `full_specification` | 1 | 1 | 1 | 0 | source_grounded | core | 全部 correct |
| `pair_shuffled_operation` | 1 | 1 | 0 | 0 | source_grounded | control | op=**shuffled** |
| `pair_shuffled_relation` | 1 | 0 | 1 | 0 | source_grounded | control | rel=**shuffled** |
| `mismatched_pair` | 1 | 0 | 0 | 0 | source_grounded | control | pair=**mismatched** |
| `full_oracle` | 1 | 1 | 1 | **1** | source_grounded | diagnostic | 全部 correct + L=1 |

别名：`relation_aware` → `full_specification`；`shuffled_operation` → `pair_shuffled_operation`（精确别名，输出逐字节一致）。

### 2.4 渲染出的 instruction 形态（v4）

以 `composite_flip`（`component_mrs = ["pronoun_substitution","negation_flip"]`）为例：

```
none                 [0000]  (只有 NLI task block)

operation_only       [0100]  Input transformation:
                             The follow-up input was derived from its source input through the following
                             ordered transformation sequence:
                             1. A noun phrase or pronoun was replaced with a coreferential expression.
                             2. A negation marker was added, removed, or reversed.

relation_only        [0010]  Output relation:
                             For a valid source-follow-up pair to which this relation applies, the output
                             labels must satisfy the following constraint:
                             if the source label is entailment, the follow-up label must be contradiction.

pair_only            [1000]  Paired source input:
                             <source_premise>…</source_premise>
                             <source_hypothesis>…</source_hypothesis>

pair_operation       [1100]  [PAIR] + [OPERATION]
pair_relation        [1010]  [PAIR] + [RELATION]
full_specification   [1110]  [PAIR] + [OPERATION] + [RELATION]
full_oracle          [1111]  [PAIR] + [Source label: entailment] + [OPERATION] + [RELATION]
```

所有 mode 的 `output` 都是**当前 follow-up 样本自己的标签**；source 行在所有 mode 下都退化为普通 NLI instruction。

### 2.5 版本与校验

| | 说明 |
|---|---|
| `instruction_template_version` | **4**（当前）。2/3 冻结保留，可用 `--instruction-template-version` 选择 |
| block 顺序 | 全局唯一，任何 mode 都改不动 |
| 硬校验 | 核心 2×2×2 出现 source label → 直接报错 |
| strict 模式 | O=1 缺 operation、R=1 缺 relation、P=1 缺 source、L=1 缺 source label、shuffled 控制出现 identity collision → 直接 fail |
| 模板哈希 | 按版本分别计算（v3 报告仍记 v3 哈希），保证 v3 重新生成逐字节一致 |
| v3 冻结验证 | 重新生成 v3 的 **33 个产物全部 SHA-256 一致**；由 `tests/fixtures/rq2_v3_instruction_golden.json` 永久守卫 |

---

## 三、Pair / Operation / Relation / Label 各自的设计

三个信息通道被设计成回答三个**互不重叠**的问题：

| 通道 | 回答的问题 | 性质 |
|---|---|---|
| **Pair** | 这个 follow-up 绑定到哪个 source input？ | grounding（实例级绑定） |
| **Operation** | 输入经过了**哪些、按什么顺序**的变换？ | factual / provenance |
| **Relation** | valid source–follow-up pair 的**输出必须满足什么**？ | normative / constraint |
| **Label** | source 的 gold label 是什么？ | 只允许在 `full_oracle`（L=1）出现 |

### 3.1 P — Pair block（grounding factor）

```text
Paired source input:
<source_premise>
{original_premise}
</source_premise>
<source_hypothesis>
{original_hypothesis}
</source_hypothesis>
```

* 只提供 `x_source`，**绝不**包含：source label、Operation、Relation、当前 follow-up 的 target label。
* 功能只有一句话：**告诉模型当前样本与这个 source 属于同一个 source–follow-up pair**。
* 刻意不用 `Reference sample:` / `Example:` 这类措辞，避免被读成 few-shot 示范。

**P 是 grounding factor，不是 O/R 的对称伙伴**：

* `P=0` 时 O/R 是 **ungrounded** 的 metadata / provenance——模型知道"做了什么变换""输出该满足什么关系"，但不知道**是哪一个** source；
* `P=1` 时 O/R 才成为绑定到具体 source–follow-up pair 的 **source_grounded** specification。

这正是 RQ2 要研究的核心：不仅问"O/R 信息有没有用"，还要问"**O/R 的作用是否依赖它是否被 grounded 到明确的 pair**"。

### 3.2 O — Operation block（有序 transformation trace）

**v3（冻结）**：每个 MR 一句声明式 provenance：

```text
Input-transformation specification:
The current sample was derived from its source sample by replacing one or more words with context-appropriate synonyms.
```

- 统一前缀 `The current sample was derived from its source sample by …`，是**声明**不是命令（不写 `Replace …` / `Change …`）。
- composite 三者共用一句 generic fallback（`applying multiple input-side transformations`）——这既丢信息，也让 `shuffled_operation` 控制出现大量 identical text。

**v4（当前）**：改成**有序 transformation trace**，由 `component_mrs` 展开：

```text
Input transformation:
The follow-up input was derived from its source input through the following ordered transformation sequence:
1. {step_1}
2. {step_2}
```

- **atomic 与 composite 共用同一个渲染器**：atomic 就是 arity-1 的 trace（只有 `1. …`）。
- `component_mrs` 被当作**有序序列**：**绝不**排序、转 set、去重、按字母或按类型重排。数组里出现两次同一变换就渲染两步（因为那可能代表真的执行了两次）。
- composite 的 Operation **只由 `component_mrs` 决定**：若 `composite_inv` 与 `composite_flip` 恰好有相同的 component 序列，它们的 Operation block **逐字节相同**。relation class 只允许出现在 Relation block。
- 正式 v4 run 启用 `--require-composite-provenance`：composite 缺 provenance **直接 fail**，不允许 generic fallback 悄悄进入正式实验。

**Operation 的硬约束**（有测试强制）：

* 不得出现 `label` / `prediction` / `entailment` / `neutral` / `contradiction` / `output` / `relation is preserved` 等输出侧词汇；
* P=0 与 P=1 使用**完全相同**的文本（同一份 description 字典），grounding 与否只体现在 Pair block 的有无；
* 不因 mode 改变措辞。

### 3.3 R — Relation block（输出约束）

**v3（冻结）**：按 relation type 共享一段文本，但用了 totalized 的 3 分类映射：

| type | 文本 |
|---|---|
| `inv` | For valid source-follow-up pairs generated by this metamorphic relation, the NLI label is preserved: the follow-up label is the same as the source label. |
| `flip` | …the follow-up label is determined from the source label by a fixed label mapping: entailment maps to contradiction, **contradiction maps to entailment, and neutral maps to neutral**. |
| `neutral` | …the follow-up label is determined from the source label by a fixed label mapping: the follow-up label is neutral. |

**v4（当前）**：改成 `must` 形式的**适用性受限约束**：

| kind | 文本 |
|---|---|
| `invariance` | For a valid source–follow-up pair, the output labels must satisfy the following constraint:<br>the follow-up label must be the same as the source label. |
| `entailment_to_contradiction` | For a valid source–follow-up pair **to which this relation applies**, the output labels must satisfy the following constraint:<br>if the source label is entailment, the follow-up label must be contradiction. |
| `entailment_to_neutral` | For a valid source–follow-up pair **to which this relation applies**, the output labels must satisfy the following constraint:<br>if the source label is entailment, the follow-up label must be neutral. |

**为什么砍掉 `C→E` / `N→N`**（详见 `RQ2/MR_RELATION_AUDIT_V4.md`）：

1. 生成器 `OPERATION_SPECS` 的 `target="contradiction"`，且 eligibility 规则拒绝一切非 entailment 的 source —— 那两个分支**从未被实例化**；
2. 论文的形式定义本身就是 `R_{E→C} = { r | O_r(E)=C }`，只定义在 entailment source 上；
3. v3 的 3 分类映射来自 `scripts/metamorphic_metrics.py`——那是为了让"满足度计数"成为全函数而做的**度量实现**，不是 MR 定义；
4. 数据核验：928 条 `flip` 行**全部**是 `source=entailment → follow-up=contradiction`。

把这些分支写进 prompt 等于**断言一个生成器从未产生的映射**，所以 v4 不写。旧 `mr_type` 字段保持不动，由 resolver（`mr_type`/`mr_id` → v4 kind）做兼容。

**Relation 的硬约束**：

* 只用 `must`，不用 `likely` / `generally` / `may` / `tends to` / `is unlikely to`——对已通过 applicability filter 的 valid pair，这是 behavioral oracle 而不是概率趋势；
* 只描述抽象的 `source output → follow-up output` 映射，**绝不**写 "the correct answer for this sample is …"；
* Relation 文本按 kind 共享（而不是按 mr_id 各写一份），使文本本身无法反推是哪个 MR。

### 3.4 L — Label anchor（只属于 diagnostic）

```text
Source label:
{original_label}
```

* **不属于**核心 2×2×2，核心条件永远 `L=0`；
* 只有 `full_oracle` 设置 `L=1`，用来测 **source-label anchoring 是否让 Relation 更容易被执行**；
* `full_oracle − full_specification` 才是 label-anchoring 的增量；`full_oracle − pair_operation` **不是**纯 Relation effect（它同时加了 Relation 和 source label）。

---

## 四、各个 MR 的 description

### 4.1 总表

| mr_id | v4 cohort 行数 | v3 Operation（冻结） | v4 Operation trace | v3 relation | v4 relation kind | 数据中的 `mr_type` | 定义来源 |
|---|---:|---|---|---|---|---|---|
| `none` | 2033 | *(source 行，无)* | — | — | — | `inv` | S1 |
| `uninformative` | 1150 | …by inserting additional content that is not informative about the rest of the text. | Additional content that is semantically irrelevant to the affected component was inserted. | `neutral` | `entailment_to_neutral` | `neutral` | S1 `target="neutral"` |
| `conditional_clause` | 565 | …by adding a conditional clause to one component. | A conditional clause was added to one component of the input. | `neutral` | `entailment_to_neutral` | `neutral` | S1 |
| `negation_flip` | 480 | …by adding, removing, or reversing a negation marker. | A negation marker was added, removed, or reversed. | `flip` | `entailment_to_contradiction` | `flip` | S1 `target="contradiction"` |
| `pronoun_substitution` | 355 | …by replacing a noun phrase or pronoun with a coreferential form. | A noun phrase or pronoun was replaced with a coreferential expression. | `inv` | `invariance` | `inv` | S1 `target="source"` |
| `composite_flip` | 255 | composite fallback（generic） | **由 `component_mrs` 展开** | `flip` | `entailment_to_contradiction` | `flip` | S1 `COMPOSITE_STAGE_PLANS` |
| `composite_neutral` | 208 | composite fallback | **由 `component_mrs` 展开** | `neutral` | `entailment_to_neutral` | `neutral` | 同上 |
| `adding_contradiction` | 180 | …by inserting additional content that conflicts with the surrounding text. | Additional content that conflicts with a proposition in the affected component was inserted. | `flip` | `entailment_to_contradiction` | `flip` | S1 |
| `voice_switch` | 45 | …by rewriting a sentence between active and passive voice. | A sentence was rewritten between active and passive voice. | `inv` | `invariance` | `inv` | S1 |
| `synonym_replacement` | 33 | …by replacing one or more words with context-appropriate synonyms. | One or more words were replaced with context-appropriate synonyms. | `inv` | `invariance` | `inv` | S1 |
| `composite_inv` | 23 | composite fallback | **由 `component_mrs` 展开** | `inv` | `invariance` | `inv` | 同上 |
| `antonym_substitution` | 13 | …by replacing one or more content words with context-appropriate antonyms. | One or more content words were replaced with context-appropriate antonyms. | `flip` | `entailment_to_contradiction` | `flip` | S1 |
| `race_sensitive_transformation` | 0 | （定义在 `MR_OPERATION_EDITS`，`supported=False`） | — | **未定义** | **未定义** | — | 无 |
| `same_type_named_entity_substitution` | 0 | （同上） | — | **未定义** | **未定义** | — | 无 |
| `tense_shift` | 0 | （同上） | — | **未定义** | **未定义** | — | 无 |

> 定义来源简称：**S1** = `/home/ubuntu/MTrain/augment_snli_all_label_mrs.py::OPERATION_SPECS`（生成器契约）；**S2** = `MRAugment.py::NLIMRTool.MR_TYPE`；**S3** = 论文 `bare_jrnl_new_sample4.tex` 的 `R_inv` / `R_{E→C}` / `R_{E→N}`；**S4** = `scripts/metamorphic_metrics.py::RELATION_TYPES`；**S5** = 数据行自带的 `mr_type`（只做一致性核对）。
>
> 关系定义**不是**从当前数据反推的。数据只用来做一致性检查。

### 4.2 v4 relation-kind 分布（v4 cohort，3307 条 follow-up）

| relation kind | 行数 | 占比 |
|---|---:|---:|
| `entailment_to_neutral` | 1923 | 58.1% |
| `entailment_to_contradiction` | 928 | 28.1% |
| `invariance` | 456 | 13.8% |

### 4.3 composite 的有序 trace（v4 cohort）

9 条真实 plan（来自 S1 的 `COMPOSITE_STAGE_PLANS`），全部 arity 2：

| ordered trace | n | 所属 composite | relation kind |
|---|---:|---|---|
| `pronoun_substitution → negation_flip` | 248 | composite_flip | E→C |
| `pronoun_substitution → uninformative` | 177 | composite_neutral | E→N |
| `voice_switch → uninformative` | 20 | composite_neutral | E→N |
| `pronoun_substitution → synonym_replacement` | 18 | composite_inv | invariance |
| `synonym_replacement → uninformative` | 11 | composite_neutral | E→N |
| `synonym_replacement → pronoun_substitution` | 4 | composite_inv | invariance |
| `voice_switch → adding_contradiction` | 4 | composite_flip | E→C |
| `synonym_replacement → negation_flip` | 3 | composite_flip | E→C |
| `voice_switch → synonym_replacement` | 1 | composite_inv | invariance |

渲染示例（`pronoun_substitution → negation_flip`）：

```text
Input transformation:
The follow-up input was derived from its source input through the following ordered transformation sequence:
1. A noun phrase or pronoun was replaced with a coreferential expression.
2. A negation marker was added, removed, or reversed.
```

若顺序反了，渲染顺序也随之相反，instruction 不同——这是 v4 的关键测试点。

---

## 五、三个通道如何做到互不泄漏

| 规则 | 强制方式 |
|---|---|
| Operation 不含输出侧信息 | 禁用词表测试（`label` / `entailment` / `contradiction` / `neutral` / `output` / `predict` 等） |
| Relation 不含实例级答案 | 禁用 "this sample" / "the correct label"；只写抽象映射 |
| composite 的 Operation 只由 component 决定 | `composite_inv` / `composite_flip` / `composite_neutral` 用同一组 component 时 Operation block 逐字节相同 |
| P=0 与 P=1 的 O/R 文本完全一致 | 同一份 description 表；有测试比对 block payload 逐字节相等 |
| 核心 2×2×2 永不出现 source label | `convert_to_alpaca` 硬校验，`role == "core"` 出现 source label 直接 raise |
| target label 永远是当前样本标签 | 所有 mode 的 `output` 都取自当前 follow-up 样本 |

### 已知的、被记录而未"修复"的泄漏风险

* **ungrounded Relation 的 shortcut**：`E→C` / `E→N` 的文本在 `P=0` 时几乎等于直接说出 follow-up label。v4 cohort 中这类行有 **2851 / 3307（86%）**，由 `relation_shortcut_risk_count` 记录。因此 `relation_only` / `operation_relation` 只作为 **ungrounded metadata diagnostic**；Relation 的**主要结论必须**来自 `pair_relation` / `full_specification` / `pair_shuffled_relation`。把文本写模糊来"修复"会破坏 specification 本身的含义，所以选择如实上报。
* **`pair_shuffled_relation` 只能做到 ~84% derange**：`entailment_to_neutral` 占训练集 augmented 行的 58%，三分类下 Hall 条件不成立，理论上限 `2·max − total` = 504/3170（15.9% 的行保持自己的 kind）。marginals 保持匹配，但这段行的 Relation 控制是 inert 的，已在 `matched_control_audit` 中报告。

---

## 六、control 的设计

| control | 做法 | 回答什么问题 |
|---|---|---|
| `pair_shuffled_operation` | 按 `operation_trace_id` 做 **matched derangement**：先同 arity、再尽量同 relation kind、trace 必须不同、seed 决定顺序 | 模型是否用到了**正确的** grounded Operation 语义 |
| `pair_shuffled_relation` | 按 relation **kind** 重排，保持 kind marginals | 正确的 Relation 语义内容是否有价值 |
| `mismatched_pair` | source 换成**另一个 group** 的 source；双输入格式与 token 预算保持可比 | 正确的 source–follow-up 对应关系本身是否重要（而不只是"多了一个输入"） |

`pair_shuffled_operation` 的实测质量（v4 cohort，seed 42）：

| 指标 | 值 |
|---|---|
| `operation_trace_identity_collision_count` | **0** |
| `operation_text_unchanged_count` | **0**（v3 是 833/3155） |
| `shuffled_operation_same_arity_rate` | **0.994**（3150/3170） |
| `shuffled_operation_cross_arity_fallback_count` | 20 |
| `shuffled_operation_no_valid_candidate_count` | 0 |
| `shuffled_operation_same_relation_kind_rate` | 0.285 |

---

## 七、一句话总结

* **模板** = 5 个固定 block 常量 + 一张 `MODE_SPECS` 注册表 + 按固定顺序拼装的派生函数；没有任何 mode 拥有自己的整段模板。
* **Pair** = 只给 `x_source` 的 grounding，唯一的功能是建立 source–follow-up 绑定。
* **Operation** = 由 `component_mrs` 展开的**有序** trace（`1. … 2. …`），只回答"输入被怎么改了"，绝不回答"输出该是什么"。
* **Relation** = `must` 形式的**输出约束**，只回答"valid pair 的输出必须满足什么"，按 kind 共享文本。
* **Label** = 只出现在 `full_oracle`，用于诊断 source-label anchoring。
* **数据集** = 训练用 `…-v3_3/augmented_data_all_label_mrs_v3_3_full.json`（5340 行，486 composite 100% 带有序 provenance），测试用 `data/nli/mr_test_data_merged/snli.jsonl`（22548 行）。

---

## 附：常用命令

```bash
# 人工核对某个真实 pair 在全部 mode 下的 instruction（无需 GPU）
python scripts/inspect_rq2_instructions.py \
  --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json \
  --pair-id 0 --mr-id composite_flip
# 加 --json 得到机器可读输出；加 --template-version 3 对比冻结的 v3

# 审计 composite provenance / trace 分布（无需 GPU）
python scripts/audit_rq2_v4_provenance.py \
  --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json

# 正式 v4 实验（8 核心条件 x 3 seeds）
python RQ2/run_rq2_snli.py --config RQ2/configs/rq2_snli_config_v4.json \
  --seeds 42 43 44 \
  --modes none operation_only relation_only operation_relation \
          pair_only pair_operation pair_relation full_specification
```
