# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MetTrain**: A metamorphic-relation (MR) driven semi-supervised learning framework for Natural Language Inference (NLI). The project explores using metamorphic relations both for data generation and as input encoding during LLM fine-tuning.

Primary paper: `bare_jrnl_new_sample4.tex` (IEEEtran format).

## Code Architecture

Only one source script exists:

- **`scripts/test_deepseek.py`** — Sends MR-transformed NLI examples to a DeepSeek LLM API and saves responses. Key components:
  - `extract_nli_fields()` — Parses NLI examples (premise, hypothesis, label) from various JSON formats
  - `build_nli_prompt()` — Constructs the prompt template: "Premise: ... Hypothesis: ... Is the hypothesis entailed by the premise?"
  - `call_deepseek()` — Makes the HTTP POST to the configured API endpoint
  - `parse_dataset_mr()` — Extracts dataset name and MR type from file path for organizing output
  - `process_file()` — Iterates over a JSONL file, calls the API, writes results to `output/<run_id>/<llm>/<dataset>/<mr>.jsonl`

## Data Structure

Data lives in `data/{dataset}_test_MR/` directories. Each file is **JSONL** (one JSON object per line), not a JSON array.

**Datasets** (5): `snli_test_MR`, `mnlim_test_MR`, `mnlimm_test_MR`, `rte_test_MR`, `sick_test_MR`

**Metamorphic Relations** (9): `adding_contradiction`, `antonym_substitution`, `conditional_clause`, `negation_flip`, `pronoun_substitution`, `summary`, `synonym_replacement`, `uninformative`, `voice_switch`

Each line has fields: `premise`, `hypothesis`, `label` (int: 0/1/2 mapping to entailment/neutral/contradiction per dataset convention), `mr_type`, `mr_category`.

## Scripts

### `scripts/test_deepseek.py` — 调用 LLM API 预测 NLI

```bash
# 激活环境（重要）
conda activate LLMTrain3.9

# 对 data/ 下所有 MR 数据跑预测
python scripts/test_deepseek.py --data-dir data --output-dir output --delay 0.5 --llm deepseek-chat

# 指定 .env 路径
python scripts/test_deepseek.py --env .env --data-dir data --output-dir output

# 指定 run-id（不指定则自动生成时间戳）
python scripts/test_deepseek.py --data-dir data --run-id my_experiment_1

# 输出: output/<run_id>/<llm>/<dataset>/<mr>.jsonl（每行一个 JSON 对象）
```

**自动任务识别**:
- RTE 数据集 → 自动使用二分类（entailment / not_entailment）prompt + 标签归一化
- 其他数据集（SNLI/MNLI/SICK） → 三分类（entailment / neutral / contradiction）
- 输出 `meta.task` 字段标记 `nli` 或 `nli-binary`

**配置** (`.env`):
- `DEEPSEEK_KEY` / `DEEPSEEK_API_KEY` — API 密钥
- `DEEPSEEK_OPENAI_BASE_URL` / `DEEPSEEK_API_URL` / `DEEPSEEK_URL` — API 地址（自动补 `/v1/chat/completions`）
- `LLM_NAME` — 默认模型名

### `scripts/evaluate.py` — 评估预测结果并生成报告

```bash
# 输出指标到控制台
python scripts/evaluate.py output/results.jsonl

# 按字段分组（数据集、MR 类型等）
python scripts/evaluate.py output/results.jsonl --group-by _source mr_type

# 指定报告输出路径（默认 output/<同名>_report.txt）
python scripts/evaluate.py output/results.jsonl --report my_report.txt
```

**报告包含**: API 状态 / 总体指标(acc/precision/recall/F1) / 混淆矩阵 / 错误用例 / 分组统计 / Token 用量与费用估算

**自动检测**: 根据 `meta.task` 字段自动区分二分类（nli-binary）和三分类（nli），使用对应标签集和评估逻辑。

### Dependencies

```bash
pip install -r requirements.txt    # requests, python-dotenv
```

### `scripts/run_all.sh` — 批量跑全量数据

```bash
# 依次跑完 data/ 下所有数据集，每跑完一个就出评估报告，最后全量汇总
bash scripts/run_all.sh

# 自定义模型和延时
bash scripts/run_all.sh --llm deepseek-chat --delay 0.3

# 输出目录结构:
#   output/<run_id>/<llm>/<dataset>/<mr>.jsonl       # 每个 MR 文件的预测结果
#   output/<run_id>/<llm>/<dataset>/<ds>_all.jsonl    # 单个数据集合并
#   output/<run_id>/<llm>/<dataset>/<ds>_report.txt   # 单个数据集评估报告
#   output/<run_id>/<llm>/all_combined.jsonl           # 全量合并
#   output/<run_id>/<llm>/all_report.txt               # 全量评估报告（按 _source, mr_type 分组）
```

## Research Context

- **Published work**: MetTrain (MR-driven semi-supervised NLI training) using 14 MRs, R-Drop regularization
- **Active exploration**: MR-as-Instruction fine-tuning — encoding MR logic as structured input during LLM fine-tuning rather than just data augmentation
- **Key hypothesis**: MR encoding helps LLMs learn logical constraints and generalize to unseen MRs
