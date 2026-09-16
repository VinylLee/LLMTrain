# LLMTrain 项目记忆

本文件供后续进入仓库的编码代理自动读取。开始工作前先读根目录 `PROJECT_OVERVIEW.md`；它是项目结构、数据流和脚本职责的详细说明。已有 `CLAUDE.md` 主要描述 API 推理流程，可作为补充，但没有完整覆盖当前三种子 LoRA 实验。

## 项目目标

项目研究 MetTrain：把蜕变关系（Metamorphic Relations, MR）用于自然语言推断（NLI）的数据增强、半监督训练和鲁棒性测试。目前核心任务包括 SNLI、MNLI matched（`mnlim`）、MNLI mismatched（`mnlimm`）、SICK 与二分类 RTE。

## 当前两条主流程

1. 通用 LLM API 推理：`scripts/test_llm.py` + `scripts/run_all.sh`，随后由 `scripts/evaluate.py`、`scripts/mrv.py`、`scripts/mrv_excel.py` 评估。
2. 系统化 LoRA 实验：Gemma 使用 `experiments/configs/experiments_config.json`，Llama 3.2 使用 `experiments/configs/experiments_config_llama32_3b.json`，均由 `scripts/run_batch_experiments.py` 编排采样、转换、微调、Original/MR 测试和汇总。

## 当前实验状态（2026-07-23）

- RQ1 merged 测试已增加 `joint correctness`：每条具有唯一 source 的 follow-up 为一个评价单位，仅 source 与 follow-up 均预测正确时计为正确；与 source accuracy、follow-up accuracy、MSR 一起进入单/多 seed、zero-shot、JSON 与训练方法 Δ 汇总。共享计算位于 `scripts/metamorphic_metrics.py`，历史 merged 逐条预测可直接重算，无需 GPU。
- 已新增 RQ1 一般合成数据对照生成器 `scripts/generate_generic_llm_aug.py`（`generic-LLM-Aug`）：默认 Gemma-3-4B，只使用原始 NLI 示例、目标标签和普通场景类别提示生成全新 premise–hypothesis，不向模型提供 MR specification 或配对结构；当前正式训练文件中的 prompt version 为 `generic_nli_v6`，输出带可复现 provenance 与审计字段，并按 Z-Aug/DISCO 相同的普通 NLI SFT 格式转换（内部兼容标记为 `mr_instruction_mode=none`，不提供 MR instruction）。SNLI 的 seed 42/43/44、每个 5340 条对比训练文件已完成 sample/convert；三 seed GPU 微调与 merged 评测已完成，结果位于 `RQ1/output/generic_llm_aug_gemma3_4b/`，12 个测试 JSONL 已通过行数与 `correct` 字段核验。训练未读取 `data/nli/generic_llm_aug/training/other/`。

- 模型：`google/gemma-3-4b-it`，LoRA rank 8，学习率 3e-4，3 epochs，batch 4，gradient accumulation 8。
- 已完成 7 组 × seed 42/43/44：MetTrain MNLI、MetTrain SICK-4439、MetTrain SICK-11176、MetTrain SNLI、Original MNLI、Original SICK、Original SNLI。
- 每个已完成种子有 4 个 Original 与 4 个 MR 测试文件。
- Gemma 结果已整理到 `output/experiments/gemma3_4b_nli/`；其中 `result.md` 是从逐行 `correct` 字段重算的逐种子结果。
- Llama-3.2-3B-Instruct 已完成全部 7 组 × seed 42/43/44 微调和测试（2026-07-20 ~ 2026-07-21）。采样与 Gemma 逐字节一致，确保公平对照。168 个测试 JSONL 全部生成并通过完整性校验。
- Llama 请求模型为 `meta-llama/Llama-3.2-3B-Instruct`；因 HF 账号无 gated 权限，实际使用并记录为非量化 BF16 镜像 `unsloth/Llama-3.2-3B-Instruct`。
- 结果文件：`output/experiments/llama32_3b_nli/RESULTS.md`（三种子汇总），`COMPARISON_vs_Gemma.md`（配对差值表）。
- `experiments/configs/experiments_config.json` 还定义了 MetTrain/Original RTE，但当前系统实验目录没有对应三种子结果；不要把它写成已完成。
- MR-as-Instruction Pilot 的 Stage 2 conversion/工程硬门槛修补已完成，125/125 自动检查通过；adding_contradiction、composite_flip、conditional_clause、pronoun_substitution 的 1114 条数据质量风险仍使 Stage 2 为 **FAIL**、原 training gate 为 **BLOCKED**，Human Validation 未通过。`MR_QUICK_SAMPLE_AUDIT_40.md` 的快速单人抽查发现 17/40 标签明确不同意、11/40 不确定、21/40 变换无效；它只支持探索性决策，不能标记 Human Validation PASS。用户已批准以 **exploratory pilot** 名义进入受限 Stage 3：仅 seed 42 的 `none`、`pair_operation`、`shuffled_operation`，首轮强制 `max_steps=20`，独立输出根为 `output/experiments/gemma3_4b_mrinstr_exploratory_smoke_v1/`，证据哈希、seed/mode/训练时长/输出目录白名单和 `confirmatory_use_allowed=false` 由 runner 强制校验。三个模式均已完成 20/20 steps、eval、checkpoint 和 adapter 保存；`none` adapter 的 8 条 MNLI 推理在 batch 1/32 下逐字节一致。迁移中发现的 dataset registry 远端绝对路径、Windows CRLF 哈希和 Transformers 5 `BatchEncoding` 单条推理兼容问题均已修复；旧失败现场保留。ignored 报告见 `artifacts/mrinstr_validation/stage3_exploratory_smoke_report.{json,md}`。这些 loss/8 条准确率仅是 pipeline health signal，不是效果结论。Stage 4、正式结论和确认性报告仍须完整 Human Validation/裁决后才能进行。双人复核模板仍保留在 ignored 的 `artifacts/mrinstr_annotation/`，但当前可跳过全量填写；不得把模板或快速抽查写成正式验证通过。
- 独立 full seed-42 exploratory 配置已完成 `none`、`pair_operation`、`shuffled_operation` 三组 3 epochs（各 393/393 steps），输出根为 `output/experiments/gemma3_4b_mrinstr_exploratory_full_seed42_v1/`，最终 eval loss 分别为 0.09847、0.16866、0.13062。`none` 曾因外层监控超时中断，runner 从完整 `checkpoint-300` 恢复后完成；其 `train_results.json` 仅描述恢复段，不得与另两组 runtime/train loss 直接比较。三组中期最佳 eval checkpoint 都因 `save_total_limit=2` 被滚动删除，当前只把最终 3-epoch adapter 视为可加载产物。旧 MR-as-Instruction Stage 4 仍未启动，且不得把它写成 Human Validation PASS；用户现已确认当前主机就是 GPU 服务器，并明确授权执行新的 SNLI v3.3 / `gemma-3-4b-it` / `none` 三 seed 实验。该 generic-LLM-Aug 对比实验现已完成 GPU 训练与 merged 评测；MR-as-Instruction Stage 4 仍保持暂停。
- 运行入口已完成 Windows/Linux 跨平台改造：项目根目录由脚本位置推导，持久化相对路径统一为 POSIX 格式，子进程使用参数列表和独立 `env`，CUDA 默认设备统一为 0。Gemma 缓存可用 `scripts/cache_hf_model.py` 下载并离线验证；gated 仓库必须先在当前环境完成 Hugging Face 授权。

- RQ2 MR-information 实验已重构为 **grounding-aware 2×2×2**（2026-09-15）：`P`(pair/source grounding) × `O`(input-transformation specification) × `R`(output-relation specification)，`L`(source-label anchoring) 只属于 diagnostic 条件 `full_oracle`，不在核心 2×2×2 内。八个核心条件为 `none` / `operation_only` / `relation_only` / `operation_relation` / `pair_only` / `pair_operation` / `pair_relation` / `full_specification`；controls 为 `pair_shuffled_operation` / `pair_shuffled_relation` / `mismatched_pair`；diagnostic 为 `full_oracle`。唯一真源是 `scripts/mr_instruction_design.py`；`relation_aware`→`full_specification`、`shuffled_operation`→`pair_shuffled_operation` 是精确别名。`instruction_template_version=3` 是当前正式版本，v2 旧模板冻结保留（历史 `experiments_config_mrinstr_*.json` 仍可复现），但 v2 与 v3 的 instruction 措辞/顺序不同，**两边的已转换数据与 adapter 不可直接比较**。RQ2 v3 的转换与输出分别写入 `RQ2/data/converted_v3/` 与 `RQ2/output/gemma3_4b_nli_grounded_v3/`，历史 v2 产物保持不动。设计文档 `RQ2/RQ2_INSTRUCTION_DESIGN.md`，关系审计 `RQ2/MR_RELATION_AUDIT.md`。启动正式训练前先用 `scripts/inspect_rq2_instructions.py` 人工核对指令。

## 目录约定

- `data/nli/original_dataset/`：规范 Original 数据。
- `data/nli/mettrain/`：不同数据集和生成模型的 MetTrain 数据；文件多且包含大量历史变体。
- `data/nli/MR_testing/`：按数据集组织的 9 类 MR 测试数据。
- `data/ft_datasets/`：按实验和种子生成的微调中间数据。
- `output/experiments/gemma3_4b_nli/`：已完成的 Gemma 三种子结果。
- `output/experiments/llama32_3b_nli/`：Llama 独立进度、adapter、测试、日志、汇总和 Gemma 对比表。
- `output/旧版结果/`、`backup_output/`、`output/experiments/系统化nli任务之前的数据（仍然值得参考）/`：历史结果；除非用户明确要求，不和当前系统实验混算。
- `paper/`：参考论文 PDF；论文主稿在根目录 `bare_jrnl_new_sample4.tex`。

## 工作规则

- 不读取、输出或提交 `.env` 中的密钥。
- 不扫描或修改 `output/**/model/` 下的大型权重，除非任务明确需要。
- 不删除、移动或重命名历史数据；中文目录名是人为设置的历史边界。
- 汇总当前实验时以逐行结果的 `correct` 为准，保留 `正确数/总数`，并明确 Original/MR、数据集和 seed。
- `mnlim` 是 MNLI matched，`mnlimm` 是 MNLI mismatched；RTE 是二分类，其标签不能和三分类 NLI 直接混算。
- 修改实验编排前检查对应 `experiments/configs/experiments_config*.json` 与模型输出根目录下的 `_progress.json`；批量脚本支持断点状态，避免误重跑昂贵训练。
- `requirements.txt` 只覆盖 API 推理的最小依赖。LoRA 流程还依赖项目环境中的 PyTorch、Transformers、PEFT、LLaMA Factory 等。
- 工作区含大量实验产物和可能属于用户的未提交内容。只修改任务要求涉及的文件，先检查再处理。
- 架构、实验范围或目录约定发生变化时，同步更新 `PROJECT_OVERVIEW.md` 和本文件的”当前实验状态”。
- 研究实验规划见 `.research/` 目录。涉及新实验设计、MR instruction 模式、数据泄漏防范或评估协议时，先查看该目录下的相关计划文档再动手修改代码。
