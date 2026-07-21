# LLMTrain / MetTrain 项目说明

更新日期：2026-07-21

## 1. 项目定位

本项目研究基于蜕变关系（Metamorphic Relations, MR）的自然语言推断训练与测试。核心问题是：由 MR 产生或编码的逻辑约束，能否改善大语言模型在 NLI 原始分布、跨数据集分布和蜕变测试上的准确率与鲁棒性。

当前仓库同时保留两代工作流：

- 通用 OpenAI 兼容 API 的模型推理、评估与 MRV 报告流程；
- 以 Gemma-3-4B 和 Llama-3.2-3B-Instruct 为基座、通过 LLaMA Factory/PEFT 进行 LoRA 微调的系统化三种子实验流程。

研究任务目前聚焦三分类 NLI（entailment、neutral、contradiction）。RTE（二分类）实验已从主配置中移除。

## 2. 总体数据流

### 2.1 当前系统化 LoRA 实验

```text
experiments_config*.json
        │
        ▼
scripts/run_batch_experiments.py
        │
        ├─ 1. sample_mettrain_pairid.py
        │     按 pair_id 成组采样，生成每个实验/种子的 sampled 数据
        │
        ├─ 2. convert_nli_to_ft.py
        │     转成 instruction/input/output 微调格式并登记数据集
        │
        ├─ 3. 动态生成 LoRA YAML → LLaMA Factory 微调
        │     模型与 checkpoint 写入 output/experiments/<实验_seed>/model/
        │
        ├─ 4. test_mettrain_experiment.py
        │     分别测试 Original 与 MR 数据
        │
        └─ 5. <output_root>/_progress.json / SUMMARY.md / RESULTS.md / tests/**/*.jsonl
              记录进度、单种子汇总、多种子聚合和逐条预测
```

批量编排的阶段名是 `sample`、`convert`、`finetune`、`test_original`、`test_mr`。配置可用 `output_root` 隔离模型系列，也可用 `--steps` 分阶段执行；每个输出根目录自己的 `_progress.json` 保存断点状态。

### 2.2 通用 API 推理与 MR 评估

```text
data/nli/MR_testing/<dataset>_test_MR/*
        │
        ▼
scripts/test_llm.py  ← .env 中的 OpenAI 兼容端点/密钥
        │
        ▼
output/<model>/<dataset>/<mr>.jsonl
        │
        ├─ scripts/evaluate.py → accuracy / F1 / confusion matrix / 分组报告
        ├─ scripts/mrv.py      → 文本版 MRV（按 MR 的错误率）
        └─ scripts/mrv_excel.py→ Excel 版 MRV
```

`scripts/run_all.sh` 封装全量发现、推理、合并和评估。`scripts/test_llm.py` 会根据路径自动识别 RTE 二分类，其他 NLI 数据按三分类处理，并把任务类型写入输出的 `meta.task`。

## 3. 顶层目录与文件

| 路径 | 作用 | 当前定位 |
|---|---|---|
| `scripts/` | 数据转换、微调、推理、评估、批量实验脚本 | 核心代码 |
| `tests/` | 标签归一化单元测试和 API 连接测试 | 测试较少，尚未覆盖训练主流程 |
| `data/nli/original_dataset/` | SNLI、MNLI、SICK、RTE 原始/规范化数据 | 当前 Original 来源 |
| `data/nli/mettrain/` | 各数据集、生成模型和筛选版本的 MetTrain 数据 | 当前训练源 + 大量历史变体 |
| `data/nli/MR_testing/` | 9 种 MR 的跨数据集测试集 | 当前 MR 测试来源 |
| `data/ft_datasets/` | 按实验和 seed 生成的采样、训练、验证数据 | 当前中间产物 |
| `output/experiments/gemma3_4b_nli/` | 已完成的 Gemma adapter、测试与汇总 | Gemma 当前结果 |
| `output/experiments/llama32_3b_nli/` | Llama 独立进度、adapter、测试与汇总 | Llama 已完成三种子结果 |
| `output/旧版结果/` | API 模型及旧微调实验结果 | 历史，不默认混算 |
| `backup_output/` | 另一份旧 API 输出备份 | 历史备份 |
| `test_only_MR/` | MR 文件的单独副本 | 辅助/历史输入 |
| `paper/` | LLMORPH、MetTrain 参考 PDF | 研究背景 |
| `bare_jrnl_new_sample4.tex` | IEEEtran 格式论文主稿 | 当前论文源文件 |
| `experiments_config.json` | 系统实验、模型、训练集和测试集路径 | 当前批量实验入口 |
| `ft_config*.yaml` | 单次 LLaMA Factory LoRA 配置样例 | 手动/兼容入口 |
| `output/experiments/gemma3_4b_nli/result.md` | Gemma 逐种子结果与三种子统计 | Gemma 当前结果汇总 |
| `scripts/summarize_seed_experiments.py` | 从逐行 `correct` 重算结果并可生成 Gemma/Llama 对比 | 可复现汇总入口 |
| `CLAUDE.md` | 既有项目说明 | 主要覆盖 API 流程 |
| `LM_STUDIO_GUIDE.md` | 本地 LM Studio 使用说明 | 本地 API 辅助文档 |
| `.env` | API 端点与密钥 | 敏感，禁止读取/提交 |

## 4. 核心脚本逻辑

### 4.1 系统实验编排

#### `scripts/run_batch_experiments.py`

当前最重要的总入口。职责包括：

- 读取 `experiments_config.json`；
- 将基础实验名扩展成 `<实验名>_seed<seed>`；
- 在配置的 `output_root` 下读写 `_progress.json`，按阶段支持 `--resume`；
- 为实验和 seed 组合建立目录及 `experiment_meta.json`；
- 调用采样、格式转换、LoRA 微调和两类测试；
- 临时生成根目录 `ft_config_<实验_seed>.yaml`，训练结束后删除；
- 强制 Hugging Face/Transformers 离线模式，使用本地模型缓存；
- 支持 `--steps`、`--output-root`、`--progress-file`、`--result-file`，并保证 `--dry-run` 不写目录、元数据或临时 YAML；
- 单 seed 时生成 `SUMMARY.md`，多 seed 时生成 `RESULTS.md`。

当前推荐命令：

```bash
# 先检查将执行什么
python scripts/run_batch_experiments.py \
  --config experiments_config.json \
  --seeds 42 43 44 \
  --dry-run

# 三种子断点续跑
python scripts/run_batch_experiments.py \
  --config experiments_config.json \
  --seeds 42 43 44 \
  --resume

# 仅运行指定实验
python scripts/run_batch_experiments.py \
  --config experiments_config.json \
  --only mettrain_rte_2490_gemma3_4b original_rte_2490_gemma3_4b \
  --seeds 42 43 44
```

可用参数还包括 `--stratify`、`--model`、`--template`、`--cuda`、`--batch-size` 和 `--skip-summary`。

注意：脚本把工作区写死为 `/home/ubuntu/LLMTrain/LLMTrain`，迁移仓库后必须修改；训练通过 `subprocess.run(..., shell=True)` 调用，配置文件应视为可信输入。

#### `scripts/sample_mettrain_pairid.py`

将输入样本按 `pair_id` 分组，使原样本及其 MR 变体在采样时保持为一个整体：

- 没有 `pair_id` 的数据会获得逐行唯一的自动 ID；
- 默认随机打乱组，累加完整组直到达到目标条数；
- `--stratify` 时按组内多数标签分层，并把目标数近似均分到各层；
- 因为不拆散组，实际条数可能略大于 `target`；
- 输出 `sampled.json` 与 `sampled.pair_ids.json`，后者记录 seed、target 和被选 pair ID。

这种分组是实验可比性的关键，不能改成逐行独立随机采样。

#### `scripts/convert_nli_to_ft.py`

负责：

- 从逐行 JSON 读取 `premise`、`hypothesis`、`label`、`pair_id`、`mr_id` 等字段；
- 支持多种 `--mr-instruction-mode`：`none`、`operation_only`、`pair_only`、`pair_operation`（主实验）、`shuffled_operation`（负对照）、`relation_aware`、`full_oracle`；
- MR 操作描述与关系效果使用独立的字典结构，主实验不读取关系效果；
- 验证集始终使用普通 NLI instruction（mode=none），避免标签泄漏；
- **按 `pair_id` group 拆分** train/validation，同一 pair 的所有样本不会跨 split；
- 支持 `--split-manifest` / `--write-split-manifest`：同一 cohort 的不同 instruction 变体复用相同的 train/validation 划分；
- `--strict-pairing` 模式：未知 `mr_id`、增强样本缺 source、多 source group 等直接报错；
- 输出 `conversion_report.json`，包含样本数、group 数、MR 分布、标签分布、fallback 统计等；
- 自动判断二分类的 `--binary` 参数保留但标记为已弃用（RTE 已从主实验移除）。

转换结果格式：

```json
{
  "instruction": "Reference sample:\nPremise: \"...\"\nHypothesis: \"...\"\n\nTransformation applied...",
  "input": "Premise: ...\nHypothesis: ...",
  "output": "entailment"
}
```

当前 P/H 只出现在 `input` 字段，不出现在 `instruction`。主实验（`pair_operation`）的 instruction 不包含原始标签、不包含 `mr_id` 原始名称、不包含 relation effect 文本。

#### `scripts/run_finetune.py`

用于单次 LoRA 微调的包装入口，展示模型/数据/任务配置并调用训练命令。`ft_config.yaml`、`ft_config_gemma.yaml`、`ft_config_snli_original.yaml` 是手动实验样例；系统批量实验主要由批量脚本动态生成配置。

#### `scripts/run_experiment.py`

较轻量的单实验流水线包装器，用于串联数据转换、训练与测试。保留作手动调试/兼容入口；正式三种子结果以 `run_batch_experiments.py` 的目录和进度记录为准。

### 4.2 LoRA 模型测试

#### `scripts/test_mettrain_experiment.py`

当前系统实验测试入口：

- 用 Transformers 加载基座模型，用 PEFT 加载 LoRA adapter；
- 支持逐条或 batch generation，批量脚本默认测试 batch size 为 32；
- 构造 NLI prompt，并归一化三分类或二分类生成标签；
- 二分类模型测试三分类数据时，将 neutral/contradiction 折叠为 not_entailment；
- Original 测试加载单一规范数据文件；
- MR 测试将某数据集目录下所有 `.json` 文件合并后整体评估；
- 输出每条样本的丰富元数据，包括 `dataset`、`test_type`、`mr_type`、`pair_id`、`premise`、`hypothesis`、`gold`、`pred` 和 `correct`。

结果位置：

```text
output/experiments/<experiment>_seed<seed>/tests/
  original/{mnlim,mnlimm,sick,snli}.jsonl
  mr/{mnlim,mnlimm,sick,snli}.jsonl
```

每条结果示例：

```json
{
  “index”: 1,
  “dataset”: “snli”,
  “test_type”: “mr”,
  “mr_type”: “synonym_replacement”,
  “pair_id”: “...”,
  “premise”: “...”,
  “hypothesis”: “...”,
  “gold”: “entailment”,
  “pred”: “neutral”,
  “correct”: false
}
```

#### `scripts/test_ft_model.py`

较早的 LoRA 测试入口，默认每个数据集只测有限样本。其部分数据路径仍是旧目录布局，正式结果不要用它代替当前测试脚本。

#### `scripts/test_lora_finetune.py`

LoRA 环境烟雾测试，会临时创建小数据集和 YAML，使用小模型启动短训练，最后清理临时文件。脚本内部分路径仍指向旧布局，使用前需要检查。

### 4.3 通用 API 推理

#### `scripts/test_llm.py`

通用 OpenAI 兼容推理入口。主要逻辑：

- 从 `.env` 或命令行指定的环境变量读取 API URL、key 和模型名；
- 递归发现数据文件；
- 兼容多种 NLI 字段形式并提取 premise、hypothesis、label；
- 从路径解析数据集与 MR 类型；
- 自动识别 RTE 二分类；
- 请求 chat completions，包含重试与延迟；
- 按 `output/<llm>/<dataset>/<mr>.jsonl` 组织逐条结果。

常用示例：

```bash
python scripts/test_llm.py --data-dir data/nli --llm phi-4 --delay 0.5
bash scripts/run_all.sh --task nli --llm phi-4 --delay 0.5
```

#### `scripts/test_deepseek.py`

`test_llm.py` 的 DeepSeek 时代兼容版本，结构和函数几乎一致。新增提供商优先接入 `test_llm.py`，不要继续复制一份提供商专用核心逻辑。

#### 提供商/本地包装脚本

- `scripts/run_lmstudio_qwen.sh`：LM Studio 本地 Qwen 调用包装；
- `scripts/run_qwen35_flash.sh`：Qwen Flash 运行包装；
- `scripts/prepare_gemma.sh`：Gemma 环境/模型准备辅助；
- `scripts/test_original_data.py`：较早的本地服务切模与 Original 数据测试流程。

### 4.4 评估与报告

#### `scripts/evaluate.py`

统一处理三分类和二分类预测：

- 从输出条目读取 gold 与生成内容；
- 对大小写、解释文本和常见标签变体做归一化；
- 计算 accuracy、各类 precision/recall/F1、混淆矩阵；
- 可按 `_source`、`mr_type` 等字段分组；
- 汇总 token usage 和解析失败项；
- 输出文本报告。

示例：

```bash
python scripts/evaluate.py output/<llm>/all_combined.jsonl
python scripts/evaluate.py output/<llm>/all_combined.jsonl --group-by _source mr_type
```

#### `scripts/mrv.py` 与 `scripts/mrv_excel.py`

MRV 在当前代码中按数据集/MR 的错误率统计，用于观察不同变换对模型的破坏程度。前者输出文本表，后者生成 Excel 报告。

### 4.5 测试代码

- `tests/test_evaluate.py`：使用 pytest 检查三分类/二分类预测归一化，尤其防止 `not_entailment` 被错误识别成 `entailment`；
- `tests/run_unit_tests.py`：轻量测试启动器；
- `tests/test.py`：DeepSeek/API 连接型测试，会访问外部服务，不属于纯单元测试。

## 5. 数据组织与格式

### 5.1 数据集命名

| 名称 | 含义 | 任务类型 |
|---|---|---|
| `snli` | Stanford NLI | 三分类 |
| `mnlim` | MultiNLI matched | 三分类 |
| `mnlimm` | MultiNLI mismatched | 三分类 |
| `sick` | SICK entailment | 三分类 |
| `rte` | Recognizing Textual Entailment | 二分类 |

### 5.2 MR 类型

当前测试目录包含 9 类关系：

`adding_contradiction`、`antonym_substitution`、`conditional_clause`、`negation_flip`、`pronoun_substitution`、`summary`、`synonym_replacement`、`uninformative`、`voice_switch`。

部分输入虽然扩展名是 `.json`，实际由脚本按逐行 JSON 读取。处理前应检查文件内容，不应只凭扩展名选择解析器。

### 5.3 当前实验配置

`experiments_config.json` 的 Gemma 公共设置：

- base model：`google/gemma-3-4b-it`；
- template：`gemma`；
- CUDA 配置值：`2`；
- LoRA：rank 8、learning rate 3e-4、3 epochs、batch size 4、gradient accumulation 8；
- 测试集：当前配置只列 mnlim、mnlimm、sick、snli，未列 RTE。

已得到完整三种子结果的实验：

| 训练来源 | 目标样本数 | 状态 |
|---|---:|---|
| MetTrain MNLI | 4413 | seed 42/43/44 完成 |
| MetTrain SICK | 4439 | seed 42/43/44 完成 |
| MetTrain SICK | 11176 | seed 42/43/44 完成 |
| MetTrain SNLI | 5340 | seed 42/43/44 完成 |
| Original MNLI | 4413 | seed 42/43/44 完成 |
| Original SICK | 4439 | seed 42/43/44 完成 |
| Original SNLI | 5340 | seed 42/43/44 完成 |
| MetTrain RTE | 2490 | 已配置，当前系统输出未完成 |
| Original RTE | 2490 | 已配置，当前系统输出未完成 |

Gemma 具体准确率见 `output/experiments/gemma3_4b_nli/RESULTS.md`；Llama 汇总见 `output/experiments/llama32_3b_nli/RESULTS.md`；两者同 seed 配对对比见 `output/experiments/llama32_3b_nli/COMPARISON_vs_Gemma.md`。

`experiments_config_llama32_3b.json` 定义相同 7 组三分类实验和 seeds 42/43/44：

- 请求模型：`meta-llama/Llama-3.2-3B-Instruct`；实际来源：`unsloth/Llama-3.2-3B-Instruct` BF16 非量化镜像；
- template：`llama3`；训练超参数和测试集与 Gemma 保持一致；
- 独立输出：`output/experiments/llama32_3b_nli/`；
- 21 组全部完成微调和测试（2026-07-20 ~ 2026-07-21），168 个测试 JSONL 文件生成并验证；
- 数据采样与 Gemma 对应文件逐字节一致，确保"仅换基座模型"的公平对照；

## 6. 结果文件的权威性

当前系统实验有多个同名或近似汇总文件，使用时遵循：

1. 最原始证据：各模型输出根目录下 `<实验_seed>/tests/{original,mr}/<dataset>.jsonl`；
2. Gemma 当前三种子汇总：`output/experiments/gemma3_4b_nli/RESULTS.md`；
3. Llama 三种子汇总：`output/experiments/llama32_3b_nli/RESULTS.md`；Gemma vs Llama 配对对比：`output/experiments/llama32_3b_nli/COMPARISON_vs_Gemma.md`；
4. 各输出根目录的 `_progress.json` 只说明阶段状态，不代表统计正确；
5. `SUMMARY.md` 是单 seed 兼容汇总，不能替代完整三种子结果；
6. `RESULTS.md` 由批量脚本生成多种子均值矩阵；当前版本使用样本标准差；
7. 中文历史目录中的材料引用前必须核对来源和实验口径；
8. `output/旧版结果/` 与 `backup_output/` 不默认并入当前系统实验。

重算准确率时使用：

```text
accuracy = correct 为 true 的条目数 / 非空且可解析结果条目总数
```

当前 `result.md` 与新版批量 `RESULTS.md` 均使用样本标准差（分母 n-1）。跨种子报告保留每个种子的 `正确数/总数（准确率）`，再给 `mean ± standard deviation`。

## 7. 环境与运行注意事项

`requirements.txt` 当前只有 `requests` 与 `python-dotenv`，只足够通用 API 推理流程。系统 LoRA 实验还从环境使用：

- PyTorch；
- Transformers；
- PEFT；
- LLaMA Factory 及其训练依赖；
- 报告、测试所需的 pytest 和 Excel 相关依赖。

因此，重新建环境时不能只执行 `pip install -r requirements.txt` 就期待训练流程可运行。当前 Llama 实验指定使用 `llmtrain310`；CUDA 映射仍需在运行时确认。

批量脚本明确设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。Gemma 与 Llama 均须在启动前完成缓存；Llama 当前镜像 snapshot 为 `006f5dcd1393c3add266de40994ba96225e9689d`。

`.env` 含 API 凭证和端点，任何文档、日志或提交都不得泄露其内容。

## 8. 已知技术债与整理边界

以下项目值得后续单独处理，但本次没有擅自删除或迁移：

- `CLAUDE.md` 仍以 API 推理流程为主，和当前系统实验说明存在时间差；
- `requirements.txt` 未覆盖训练依赖；
- `.gitignore` 有“Secrets/Data/Output”标题，但当前可见内容没有对应的明确忽略规则，需要先决定哪些研究数据和结果应版本化；
- `test_llm.py` 与 `test_deepseek.py` 重复度高；多个旧测试/微调入口与当前批量入口并存；
- 多个脚本硬编码 `/home/ubuntu/LLMTrain/LLMTrain`，仓库不可直接迁移；
- 批量脚本头部示例和注释仍写”8 个实验、5 个测试集”，实际配置是 9 个实验、4 个测试集；
- 采样以完整 `pair_id` 组为单位，因此实际训练条数可超过配置 target；
- `convert_nli_to_ft.py` 的 `--shuffle` 使用 `store_true` 且默认已经为 true，当前 CLI 无法关闭打乱；
- 数据目录包含大量带模型名、阈值、随机后缀和 `old` 的实验变体；
- `output/experiments/` 有 `SUMMARY.md`、`result.md`、`RESULTS.md` 多份汇总，口径不完全一致；
- 根目录存在名为 `claude --resume b0643a76-674f-4aa7-912a-947998144148` 的可疑文件，可能是误创建，但删除前需用户确认；
- 自动化测试主要覆盖标签归一化，尚未覆盖采样分组、数据转换、进度恢复和结果聚合。

整理原则是：先记录边界和权威来源，再在明确授权下清理。历史数据可能与论文复现实验有关，不能仅凭文件名删除。

## 9. 已完成工作

- ✅ Llama-3.2-3B-Instruct 7 组 × 3 seeds 微调和测试（2026-07-20 ~ 2026-07-21，21/21 实验全通过）；
- ✅ Llama 与 Gemma 同 seed 配对差值表已生成（见 `COMPARISON_vs_Gemma.md`）；
- ✅ `scripts/run_batch_experiments.py` 增强：支持 `--result-file`、`--progress-file`、按阶段执行、零写入 dry-run；
- ✅ `scripts/summarize_seed_experiments.py` 新增：从逐行 `correct` 重算三种子聚合、标准差、模型间对比；
- ✅ `scripts/convert_nli_to_ft.py` 重构：纯函数拆分、group-aware split、7 种 MR instruction mode、strict pairing、conversion report、split manifest；
- ✅ `scripts/test_mettrain_experiment.py` 增强：输出每条样本的 `mr_type`、`pair_id`、`premise`、`hypothesis` 等丰富元数据；
- ✅ `scripts/run_batch_experiments.py` 更新：MR mode 传播、cohort-based manifest 共享、experiment meta 扩展；
- ✅ RTE 已从实验配置和测试脚本中移除；
- ✅ `tests/test_convert_nli_to_ft.py` 新增 23 个单元测试覆盖 split、instruction、strict 模式。

## 10. 后续工作建议

建议按以下优先级继续：

1. ~~决定 RTE 的测试口径~~ ✅ RTE 已从实验配置和测试脚本中移除；
2. ~~完成或明确取消 RTE 三种子系统实验~~ ✅ RTE 实验已删除；
3. ~~在系统测试输出中保留 `mr_type`、源文件和原始样本标识~~ ✅ 测试输出已包含完整元数据；
4. 合并或标记 `output/experiments/` 中三份汇总文件的用途，并统一标准差口径；
5. ~~为采样、转换、断点恢复和结果汇总补单元测试~~ ✅ `tests/test_convert_nli_to_ft.py` 已覆盖转换流程；
6. 补齐训练环境锁定文件，或把 API 与训练依赖分成两个 requirements；
7. 经确认后清理误创建文件、缓存和明确无用的重复入口；
8. 论文中报告三种子结果时，同时说明 Original/MR 测试集规模与标准差口径。

## 11. 文档维护约定

后续进入本项目时：

1. 先读 `AGENTS.md` 和本文件；
2. 只在相关任务需要时再查看大型数据、模型 checkpoint 和历史输出；
3. 新实验完成后更新“当前实验配置”“结果文件的权威性”与 `AGENTS.md` 状态；
4. 核心入口、目录语义或标签映射发生变化时同步更新数据流和脚本职责；
5. 文档事实优先从配置、`experiment_meta.json`、`_progress.json` 和逐条测试结果交叉验证。
