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

- **`scripts/test_llm.py`** — Sends examples to any LLM API (OpenAI-compatible) and saves **per-example** results. Key components:
  - `extract_nli_fields()` — Parses NLI examples (premise, hypothesis, label) from various JSON formats
  - `build_nli_prompt()` — Constructs the NLI prompt template
  - `call_llm()` — Makes the HTTP POST to the configured API endpoint
  - `parse_dataset_mr()` — Returns `(dataset, mr_type, test_type)` from a file path; `test_type` ∈ `original` / `mr` / `skip`
  - `process_file()` — Iterates over a JSONL file, calls the API, writes one JSON record per test case (with parsed `pred`/`correct`)
  - `--test-type original|mr` → nested layout `output/<llm>/<OFFICIAL_DATASET>/<test_type>/…`; `auto` → legacy flat layout

- **`scripts/organize_nli_experiment.py`** — 整理 NLI 实验结果：规范化每条用例（缺失的 `pred`/`correct` 从 response 回填）、把 MR 各文件合并成 `<DS>_MR_all.jsonl`、生成逐数据集准确率的 `SUMMARY.md`
- **`scripts/run_all.sh`** — 实验入口：对指定 LLM 跑 SNLI/MNLIm/MNLImm/SICK 的 Original + MR 测试 → 整理 → 摘要
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

**Metamorphic Relations** (8 个数据文件；SICK 只有 7 个，无 `negation_flip`): `adding_contradiction`, `antonym_substitution`, `conditional_clause`, `negation_flip`, `pronoun_substitution`, `synonym_replacement`, `uninformative`, `voice_switch`

`summary` **不是真实 MR** —— 它是每个 MR 目录里的元数据文件 `{ds}_test_summary_*.json`（无样本行），会被 `test_llm.py` 自动跳过。

Each line has fields: `premise`, `hypothesis`, `label` (int: 0/1/2), `mr_type`, `mr_category`.

## Output Structure

两种布局：

**实验布局**（`test_llm.py --test-type original|mr`，由 `run_all.sh` 使用）：

```
output/
  <llm_name>/                   # 模型文件夹名，如 DeepSeek-v4-flash-0731
    <OFFICIAL_DATASET>/         # 正式数据集名：SNLI / MNLIm / MNLImm / SICK
      original/
        <DS>_original.jsonl     # 每行=一个用例（含 gold/pred/correct）
      MR/
        <mr>.jsonl              # 每个 MR 一个文件
        <DS>_MR_all.jsonl       # MR 合并文件（每条含 uid）
    SUMMARY.md                  # 逐数据集×测试类型 + 逐MR 准确率摘要
```

**legacy 布局**（`test_llm.py` 默认 `--test-type auto`，旧 wrapper 使用）：

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

### `scripts/test_llm.py` — Run LLM inference on NLI datasets (original + MR)

```bash
conda activate llmtrain310

# 实验布局：Original 测试（4 个数据集，每个 test.json，一行=一个用例）
python scripts/test_llm.py --data-dir data/nli/original_dataset --test-type original \
    --llm DeepSeek-v4-flash-0731 --api-model deepseek-v4-flash

# 实验布局：MR 测试
python scripts/test_llm.py --data-dir data/nli/MR_testing --test-type mr \
    --llm DeepSeek-v4-flash-0731 --api-model deepseek-v4-flash

# 只看会发现哪些文件、写到哪（不调 API）
python scripts/test_llm.py --data-dir data/nli/MR_testing --test-type mr --dry-run

# 冒烟测试：每个文件只处理前 N 条
python scripts/test_llm.py --data-dir data/nli/original_dataset --test-type original \
    --datasets sick --max-samples 10 --llm DeepSeek-v4-flash-0731

# 指定数据集子集
python scripts/test_llm.py --data-dir data/nli/MR_testing --test-type mr --datasets snli,sick ...

# legacy 模式（旧布局，旧 wrapper 使用）
python scripts/test_llm.py --data-dir data/nli --llm phi-4 --delay 0.5
python scripts/test_llm.py --data-dir data/nli --run-id my_experiment_1

# Override API URL / key from .env
python scripts/test_llm.py --data-dir data/nli --llm phi-4 --api-url-env LMSTUDIO_BASE_URL --api-key-env LMSTUDIO_KEY
```

**说明**:
- `--llm` 是输出文件夹名；`--api-model` 是真正发给 API 的模型 id（默认等于 `--llm`）。例如文件夹 `DeepSeek-v4-flash-0731` + API id `deepseek-v4-flash`。
- `--run-id` 默认不再自动生成时间戳目录（默认 None → `--llm` 成为顶层目录）。
- 每条记录（逐用例）含 `premise/hypothesis/gold/pred/correct/mr_type/mr_category/idx/status/response` 等字段。

**自动任务识别**:
- RTE 数据集 → 自动使用二分类（entailment / not_entailment）prompt + 标签归一化
- 其他数据集（SNLI/MNLI/SICK） → 三分类（entailment / neutral / contradiction）
- 输出 `meta.task` 字段标记 `nli` 或 `nli-binary`

**配置** (`.env`):
- `DEEPSEEK_KEY` / `DEEPSEEK_API_KEY` — API 密钥
- `DEEPSEEK_OPENAI_BASE_URL` / `DEEPSEEK_API_URL` / `DEEPSEEK_URL` — API 地址
- `DEEPSEEK_V4_FLASH` — DeepSeek v4 flash 的 API 模型 id（当前 `deepseek-v4-flash`）
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

### `scripts/run_all.sh` — 批量实验入口（original + MR + 摘要）

```bash
# DeepSeek-v4-flash 实验（4 数据集 Original + MR + SUMMARY.md）
bash scripts/run_all.sh --llm DeepSeek-v4-flash-0731 --api-model deepseek-v4-flash --delay 0.1

# 只测部分数据集
bash scripts/run_all.sh --datasets snli,sick

# 冒烟测试（每文件 N 条，走完整流程含 SUMMARY）
bash scripts/run_all.sh --datasets sick --max-samples 2 --api-model deepseek-v4-flash

# 输出到指定目录（便于隔离冒烟结果）
bash scripts/run_all.sh --datasets sick --max-samples 2 --output-dir /tmp/nli_smoke

# 使用 LM Studio / 其它 OpenAI 兼容端点
bash scripts/run_all.sh --llm phi-4 --api-url-env LMSTUDIO_BASE_URL --api-key-env LMSTUDIO_KEY

# 输出目录结构（实验布局）:
#   output/<llm>/<OFFICIAL_DATASET>/original/<DS>_original.jsonl   # 每行=一个用例
#   output/<llm>/<OFFICIAL_DATASET>/MR/<mr>.jsonl                  # 每个 MR 一个文件
#   output/<llm>/<OFFICIAL_DATASET>/MR/<DS>_MR_all.jsonl           # MR 合并文件
#   output/<llm>/SUMMARY.md                                        # 逐数据集/逐MR 准确率摘要
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
