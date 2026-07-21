# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

> Current repository-wide architecture, the systematic Gemma LoRA experiment
> pipeline, experiment status, and result-source conventions are documented in
> `PROJECT_OVERVIEW.md`. Read it before relying on the older API-focused notes
> below.

## Project Overview

**MetTrain**: A metamorphic-relation (MR) driven semi-supervised learning framework for Natural Language Inference (NLI). The project explores using metamorphic relations both for data generation and as input encoding during LLM fine-tuning.

Primary paper: `bare_jrnl_new_sample4.tex` (IEEEtran format).

**Expanding scope**: The framework is being extended beyond NLI to other NLP tasks, and beyond DeepSeek to any LLM provider (OpenAI-compatible APIs, local models via LM Studio, etc.).

## Code Architecture

### Core scripts

- **`scripts/test_llm.py`** — Sends examples to any LLM API (OpenAI-compatible) and saves responses. Key components:
  - `extract_nli_fields()` — Parses NLI examples (premise, hypothesis, label) from various JSON formats
  - `build_nli_prompt()` — Constructs the NLI prompt template
  - `call_llm()` — Makes the HTTP POST to the configured API endpoint
  - `parse_dataset_mr()` — Extracts dataset name and MR type from file path for organizing output
  - `process_file()` — Iterates over a JSONL file, calls the API, writes results to `output/<llm>/<dataset>/<mr>.jsonl`

- **`scripts/run_all.sh`** — Batch runner: iterate all MR files in a task directory → predict → evaluate → report
- **`scripts/evaluate.py`** — Evaluate prediction results and generate accuracy/F1/confusion-matrix reports
- **`scripts/mrv.py`** — Compute MRV (Mutation Rate Value = error rate per MR and dataset)
- **`scripts/mrv_excel.py`** — Export MRV report to Excel

## Data Structure

### Task-based layout

```
data/
  nli/                          # NLI task data
    snli_test_MR/
    mnlim_test_MR/
    mnlimm_test_MR/
    rte_test_MR/
    sick_test_MR/
  ...future tasks go here...
```

Each file is **JSONL** (one JSON object per line), not a JSON array.

**Metamorphic Relations** (9): `adding_contradiction`, `antonym_substitution`, `conditional_clause`, `negation_flip`, `pronoun_substitution`, `summary`, `synonym_replacement`, `uninformative`, `voice_switch`

Each line has fields: `premise`, `hypothesis`, `label` (int: 0/1/2), `mr_type`, `mr_category`.

## Output Structure

```
output/
  <llm_name>/                   # one directory per model
    <dataset>/
      <mr>.jsonl                # per-MR predictions
      <dataset>_all.jsonl       # dataset-level merged results
      <dataset>_report.txt      # per-dataset evaluation
    all_combined.jsonl           # all datasets merged
    all_report.txt               # full evaluation report (grouped by _source, mr_type)
  _archive/                      # incomplete / legacy timestamp-based runs
```

## Scripts

### `scripts/test_llm.py` — Run LLM inference on datasets

```bash
conda activate LLMTrain3.9

# Basic usage on NLI data
python scripts/test_llm.py --data-dir data/nli --llm phi-4 --delay 0.5

# Custom run-id (default: auto-generated timestamp)
python scripts/test_llm.py --data-dir data/nli --run-id my_experiment_1

# Override API URL / key from .env
python scripts/test_llm.py --data-dir data/nli --api-url-env LMSTUDIO_BASE_URL --api-key-env LMSTUDIO_KEY
```

**自动任务识别**:
- RTE 数据集 → 自动使用二分类（entailment / not_entailment）prompt + 标签归一化
- 其他数据集（SNLI/MNLI/SICK） → 三分类（entailment / neutral / contradiction）
- 输出 `meta.task` 字段标记 `nli` 或 `nli-binary`

**配置** (`.env`):
- `DEEPSEEK_KEY` / `DEEPSEEK_API_KEY` — API 密钥
- `DEEPSEEK_OPENAI_BASE_URL` / `DEEPSEEK_API_URL` / `DEEPSEEK_URL` — API 地址
- `LLM_NAME` — 默认模型名
- `LMSTUDIO_BASE_URL=http://localhost:1234` — 本地 LM Studio
- `LMSTUDIO_KEY=` — LM Studio 不需密钥
- `BAILIAN_BASE_URL` / `BAILIAN_API_KEY` — 百炼平台

### 本地模型（LM Studio）

LM Studio 运行在 `localhost:1234`，OpenAI 兼容接口。

```bash
# 使用 .env 中的 LM Studio 配置
python scripts/test_llm.py --llm phi-4 --api-url-env LMSTUDIO_BASE_URL --api-key-env LMSTUDIO_KEY

# 或环境变量覆盖
DEEPSEEK_API_URL=http://localhost:1234/v1/chat/completions \
  DEEPSEEK_API_KEY="" \
  python scripts/test_llm.py --llm phi-4
```

### `scripts/run_all.sh` — 批量跑全量数据

```bash
# NLI 任务全量跑
bash scripts/run_all.sh --llm phi-4 --delay 0.5

# 指定任务目录
bash scripts/run_all.sh --task nli --llm deepseek-chat --delay 0.3

# 使用 LM Studio
bash scripts/run_all.sh --llm phi-4 --delay 0.5 --api-url-env LMSTUDIO_BASE_URL --api-key-env LMSTUDIO_KEY

# 输出目录结构:
#   output/<llm>/<dataset>/<mr>.jsonl          # 每个 MR 文件的预测结果
#   output/<llm>/<dataset>/<ds>_all.jsonl       # 单个数据集合并
#   output/<llm>/<dataset>/<ds>_report.txt      # 单个数据集评估报告
#   output/<llm>/all_combined.jsonl              # 全量合并
#   output/<llm>/all_report.txt                  # 全量评估报告（按 _source, mr_type 分组）
```

### `scripts/evaluate.py` — 评估预测结果并生成报告

```bash
python scripts/evaluate.py output/<llm>/all_combined.jsonl
python scripts/evaluate.py output/<llm>/all_combined.jsonl --group-by _source mr_type
```

### `scripts/mrv.py` — MRV 指标

```bash
python scripts/mrv.py output/<llm>/
```

### Dependencies

```bash
pip install -r requirements.txt    # requests, python-dotenv
```

## Research Context

- **Published work**: MetTrain (MR-driven semi-supervised NLI training) using 14 MRs, R-Drop regularization
- **Active exploration**: MR-as-Instruction fine-tuning — encoding MR logic as structured input during LLM fine-tuning rather than just data augmentation
- **Key hypothesis**: MR encoding helps LLMs learn logical constraints and generalize to unseen MRs

## Research Plans

- `.research/MR_AS_INSTRUCTION_PLAN.md` — Full experimental design for MR-as-Instruction, including variable definitions, instruction templates, evaluation protocols, and staged execution plan. Read this before implementing any MR-instruction features or modifying the convert/train pipeline for experiments.
