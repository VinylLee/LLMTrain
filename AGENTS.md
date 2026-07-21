# LLMTrain 项目记忆

本文件供后续进入仓库的编码代理自动读取。开始工作前先读根目录 `PROJECT_OVERVIEW.md`；它是项目结构、数据流和脚本职责的详细说明。已有 `CLAUDE.md` 主要描述 API 推理流程，可作为补充，但没有完整覆盖当前三种子 LoRA 实验。

## 项目目标

项目研究 MetTrain：把蜕变关系（Metamorphic Relations, MR）用于自然语言推断（NLI）的数据增强、半监督训练和鲁棒性测试。目前核心任务包括 SNLI、MNLI matched（`mnlim`）、MNLI mismatched（`mnlimm`）、SICK 与二分类 RTE。

## 当前两条主流程

1. 通用 LLM API 推理：`scripts/test_llm.py` + `scripts/run_all.sh`，随后由 `scripts/evaluate.py`、`scripts/mrv.py`、`scripts/mrv_excel.py` 评估。
2. 系统化 LoRA 实验：Gemma 使用 `experiments_config.json`，Llama 3.2 使用 `experiments_config_llama32_3b.json`，均由 `scripts/run_batch_experiments.py` 编排采样、转换、微调、Original/MR 测试和汇总。

## 当前实验状态（2026-07-21）

- 模型：`google/gemma-3-4b-it`，LoRA rank 8，学习率 3e-4，3 epochs，batch 4，gradient accumulation 8。
- 已完成 7 组 × seed 42/43/44：MetTrain MNLI、MetTrain SICK-4439、MetTrain SICK-11176、MetTrain SNLI、Original MNLI、Original SICK、Original SNLI。
- 每个已完成种子有 4 个 Original 与 4 个 MR 测试文件。
- Gemma 结果已整理到 `output/experiments/gemma3_4b_nli/`；其中 `result.md` 是从逐行 `correct` 字段重算的逐种子结果。
- Llama-3.2-3B-Instruct 已完成全部 7 组 × seed 42/43/44 微调和测试（2026-07-20 ~ 2026-07-21）。采样与 Gemma 逐字节一致，确保公平对照。168 个测试 JSONL 全部生成并通过完整性校验。
- Llama 请求模型为 `meta-llama/Llama-3.2-3B-Instruct`；因 HF 账号无 gated 权限，实际使用并记录为非量化 BF16 镜像 `unsloth/Llama-3.2-3B-Instruct`。
- 结果文件：`output/experiments/llama32_3b_nli/RESULTS.md`（三种子汇总），`COMPARISON_vs_Gemma.md`（配对差值表）。
- `experiments_config.json` 还定义了 MetTrain/Original RTE，但当前系统实验目录没有对应三种子结果；不要把它写成已完成。

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
- 修改实验编排前检查对应 `experiments_config*.json` 与模型输出根目录下的 `_progress.json`；批量脚本支持断点状态，避免误重跑昂贵训练。
- `requirements.txt` 只覆盖 API 推理的最小依赖。LoRA 流程还依赖项目环境中的 PyTorch、Transformers、PEFT、LLaMA Factory 等。
- 工作区含大量实验产物和可能属于用户的未提交内容。只修改任务要求涉及的文件，先检查再处理。
- 架构、实验范围或目录约定发生变化时，同步更新 `PROJECT_OVERVIEW.md` 和本文件的”当前实验状态”。
- 研究实验规划见 `.research/` 目录。涉及新实验设计、MR instruction 模式、数据泄漏防范或评估协议时，先查看该目录下的相关计划文档再动手修改代码。
