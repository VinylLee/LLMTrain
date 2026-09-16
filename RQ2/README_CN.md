# RQ2：SNLI MR-information representation

在固定 standard SFT 目标、相同 metamorphic examples 与匹配训练预算下，研究不同形式的
MR information 如何影响 **persistent MR-compliant behavior**。

**入口文档 → [`RQ2_V4_DESIGN_SUMMARY.md`](RQ2_V4_DESIGN_SUMMARY.md)**
这一份涵盖当前全部现状：用哪个数据集、模板如何组装、Pair / Operation / Relation / Label
各自的设计、每个 MR 的 description、以及已知的泄漏与对照风险。**修改本项目前先读它。**

## 当前版本

| 项 | 值 |
|---|---|
| `instruction_template_version` | **4**（当前）；2、3 冻结保留且可复现 |
| mode 数量 | 8 个核心（grounding-aware 2×2×2）+ 3 个 control + 1 个 diagnostic = **12** |
| 训练数据 | `data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json` |
| 基座模型 | `google/gemma-3-4b-it`，LoRA rank 8，3 epochs，lr 3e-4，batch 4 × grad_accum 8 |
| seeds | 42 / 43 / 44；同一 seed 下所有 mode 复用一个 cohort + 一个 split manifest |
| 配置 | [`configs/rq2_snli_config_v4.json`](configs/rq2_snli_config_v4.json) |
| 指标 | source accuracy、follow-up accuracy、MSR、joint correctness |
| merged test | `data/nli/mr_test_data_merged/snli.jsonl`（22548 行；推理统一用普通三分类 NLI prompt，不提供 MR specification） |

> v4 使用与 v3 **不同的 cohort 文件**：它是唯一一个 composite 行带**有序** `component_mrs`
> provenance 的 artifact。两者的 source 行完全相同，但 metamorphic examples 不同，
> 因此 **v3 与 v4 的结果不能逐例比较**。
>
> v3 cohort 的 composite 无法补 provenance——那次生成的 generation log 已不存在，
> 旧生成器用 `random.randint(3, len(invariant_mrs))` 随机组合且从不落盘。

## 文档

| 文件 | 内容 |
|---|---|
| [`RQ2_V4_DESIGN_SUMMARY.md`](RQ2_V4_DESIGN_SUMMARY.md) | **入口**：数据集、模板、P/O/R/L 设计、MR description 表、风险 |
| [`RQ2_INSTRUCTION_DESIGN_V4.md`](RQ2_INSTRUCTION_DESIGN_V4.md) | v4 相对 v3 改了什么、对照质量、已知限制 |
| [`MR_RELATION_AUDIT_V4.md`](MR_RELATION_AUDIT_V4.md) | 逐 MR 关系证据；`flip` 收窄为 `E→C` 的理由 |
| [`RQ2_INSTRUCTION_DESIGN.md`](RQ2_INSTRUCTION_DESIGN.md) | 共享 2×2×2 矩阵 + v3 措辞（v3 已冻结） |
| [`MR_RELATION_AUDIT.md`](MR_RELATION_AUDIT.md) | v3 文本表的关系证据 |

## 运行

```bash
conda activate llmtrain310

# 8 个核心条件 x seeds 42/43/44
python RQ2/run_rq2_snli.py --config RQ2/configs/rq2_snli_config_v4.json \
  --seeds 42 43 44 \
  --modes none operation_only relation_only operation_relation \
          pair_only pair_operation pair_relation full_specification

# 全部 12 个条件（3 seeds = 33 次 fine-tune）
python RQ2/run_rq2_snli.py --config RQ2/configs/rq2_snli_config_v4.json --seeds 42 43 44
```

阶段为 `audit → sample → convert → finetune → test → summarize`，可用 `--steps` 分阶段执行。
`--smoke` 只把训练限制在 20 steps，用于 pipeline health check，**不是效果结论**。

> 注意：`--smoke` 的 20 steps 只用于 pipeline health check。MR-as-Instruction 的数据质量门槛与
> Human Validation 状态在审计完成前都不是确认性通过，因此本目录结果在审计完成前只能作为
> exploratory evidence。

## 训练前先人工核对

```bash
# 渲染某个真实 pair 在全部 mode 下的 instruction（无需模型 / GPU）
python scripts/inspect_rq2_instructions.py \
  --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json \
  --pair-id 0 --mr-id composite_flip

# composite provenance / trace 审计
python scripts/audit_rq2_v4_provenance.py --input <同上文件>
```

## 目录约定

```
RQ2/data/cohorts/      # 冻结的 v3 cohort（不要覆盖）
RQ2/data/cohorts_v4/   # v4 cohort
RQ2/data/converted/    # 冻结的 v2 转换结果
RQ2/data/converted_v3/ # 冻结的 v3 转换结果
RQ2/data/converted_v4/ # v4 转换结果
RQ2/output/            # adapter、测试、汇总（已被 .gitignore 排除）
```
