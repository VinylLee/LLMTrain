# RQ1 NLI Experiment Pipeline

RQ1 研究问题：**MR-guided fine-tuning 能否在不显著损害 standard task performance 的前提下改善 MR compliance？**

本脚本在统一 standard SFT objective 下，系统比较 **Original 训练** 与 **MR-guided 训练** 对 NLI 模型在 4 个测试集上的表现。

## 快速开始

```bash
conda activate llmtrain310

# 最小示例：Gemma-3-4B，单 seed，两种实验类型，3 个训练集
python RQ1/run_rq1_nli.py --model gemma-3-4b-it

# 多 seed 统计（推荐）
python RQ1/run_rq1_nli.py --model gemma-3-4b-it --seeds 42 43 44

# 只跑 Original 实验
python RQ1/run_rq1_nli.py --model gemma-3-4b-it --experiment original

# 只跑 MR 实验，仅 SNLI 训练数据
python RQ1/run_rq1_nli.py --model gemma-3-4b-it --experiment mr --train-data snli

# Dry-run 预览所有命令
python RQ1/run_rq1_nli.py --model gemma-3-4b-it --dry-run
```

## CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--model` | str | **必填** | 模型 key（见下方支持列表） |
| `--experiment` | choice[] | `original mr` | 实验类型：`original` / `mr` / `both` |
| `--train-data` | str[] | snli,mnlim,sick | 训练数据集 |
| `--seeds` | int[] | `[42]` | 随机种子列表 |
| `--mr-instruction-mode` | choice | `pair_operation` | MR instruction 模式（仅 MR 实验生效） |
| `--target` | int | 配置中的默认值 | 覆盖训练样本目标数 |
| `--lr` | float | 3e-4 | 覆盖 learning rate |
| `--epochs` | float | 3.0 | 覆盖 epochs |
| `--rank` | int | 8 | 覆盖 LoRA rank |
| `--batch` | int | 4 | 覆盖 per-device batch size |
| `--grad-accum` | int | 8 | 覆盖 gradient accumulation |
| `--test-batch-size` | int | 32 | 推理 batch size |
| `--max-samples` | int | None | 测试时每数据集最大样本数（冒烟测试） |
| `--dry-run` | flag | False | 仅打印命令，不执行 |
| `--resume` | flag | False | 跳过已完成的阶段 |
| `--no-summary` | flag | False | 跳过结果汇总 |
| `--output-root` | path | `RQ1/output/<model>` | 输出根目录 |
| `--config` | path | `RQ1/configs/rq1_nli_config.json` | 配置文件 |
| `--steps` | choice[] | 全部 5 阶段 | 流水线阶段子集 |
| `--test-datasets` | str[] | snli,mnlim,mnlimm,sick | 测试数据集子集 |
| `--cuda` | str | 配置中的值 | 覆盖 CUDA 设备 |
| `--only` | str[] | None | 只运行指定实验名 |

## 流水线阶段

```
sample → convert → finetune → test_original → test_mr
```

- **sample**: 从训练数据中按 pair_id 分组采样（`sample_mettrain_pairid.py`）
- **convert**: 转为 Alpaca 微调格式，拆分 train/val（`convert_nli_to_ft.py`）
- **finetune**: LLaMA-Factory LoRA SFT（`llamafactory.cli train`）
- **test_original**: 在 4 个 NLI 数据集的 original test 上评估
- **test_mr**: 在 4 个 NLI 数据集的 MR test 上评估

可用 `--steps` 只运行部分阶段（如跳过已完成的前几步）。

## 支持的模型

| Key | Hub ID | Template |
|---|---|---|
| `gemma-3-4b-it` | `google/gemma-3-4b-it` | `gemma` |
| `llama-3.2-3b` | `unsloth/Llama-3.2-3B-Instruct` | `llama3` |

添加新模型：编辑 `RQ1/configs/rq1_nli_config.json` 的 `models` 字段。

## 训练数据集与目标数

| 数据集 | 原始数据量 | Gemma MetTrain | Llama MetTrain |
|---|---|---|---|
| snli | 550K train | 5,340 (target) | 4,149 (target) |
| mnlim | 392K train | 4,413 (target) | 3,969 (target) |
| mnlimm | 392K train | 4,413 (target) | 3,993 (target) |
| sick | 4,439 train | 11,176 (全量) / 4,439 (target) | 8,669 (全量) / 4,439 (target) |

默认 target 匹配 Original 训练数据量以实现公平比较。sick MR 全量数据更大，可用 `--target 11176` 使用全部 MR 数据。

默认训练数据集为 `snli,mnlim,sick`（3 个），测试数据集为 `snli,mnlim,mnlimm,sick`（4 个）。

## MR Instruction 模式

| 模式 | 说明 | 适用场景 |
|---|---|---|
| `none` | 纯 NLI instruction | Original 实验默认 |
| `pair_operation` | reference sample + 操作描述 | MR 实验默认（RQ1 baseline） |
| `operation_only` | 仅操作描述 | RQ2 ablation |
| `pair_only` | 仅 reference sample | RQ2 ablation |
| `shuffled_operation` | 随机打乱的操作描述 | RQ2 负对照 |
| `relation_aware` | 操作 + 关系效果 | RQ2 ablation |
| `full_oracle` | reference + 标签 + 操作 + 关系 | RQ2 上界 |

## 输出结构

```
RQ1/output/<model_key>/
  rq1_original_snli_seed42/
    experiment_meta.json          # 实验元信息
    model/                        # LoRA adapter checkpoint
    tests/
      original/                   # 逐条 JSONL 预测结果
        snli.jsonl
        mnlim.jsonl
        mnlimm.jsonl
        sick.jsonl
      mr/                         # MR 测试逐条结果
        snli.jsonl
        mnlim.jsonl
        mnlimm.jsonl
        sick.jsonl
  rq1_mr_snli_seed42/
    ... (同上)
  _progress.json                  # 断点续跑状态
  SUMMARY.md                      # 单 seed 结果汇总
  RESULTS.md                      # 多 seed 均值±标准差
  rq1_report.json                 # 机器可读报告
```

每条测试 JSONL 包含字段：`index, dataset, test_type, premise, hypothesis, gold, pred, correct`。
MR 测试额外包含：`mr_type, mr_category, pair_id`。

## 断点续跑

```bash
# 中断后恢复（自动跳过已完成阶段）
python RQ1/run_rq1_nli.py --model gemma-3-4b-it --seeds 42 43 44 --resume

# 只重跑 failed 的实验
python RQ1/run_rq1_nli.py --model gemma-3-4b-it --resume --only rq1_mr_snli_seed43
```

## 自定义输出目录

```bash
# 隔离 exploratory 运行
python RQ1/run_rq1_nli.py --model gemma-3-4b-it \
    --output-root RQ1/output/exploratory_v1

# 冒烟测试输出到 /tmp
python RQ1/run_rq1_nli.py --model gemma-3-4b-it \
    --experiment original --train-data sick --max-samples 10 \
    --output-root /tmp/rq1_smoke
```

## 结果解读

**RESULTS.md** 包含三张关键表格：

1. **Original Test Sets** — 各实验在 4 个数据集的 original 测试上的准确率（standard task performance）
2. **MR Test Sets** — 各实验在 4 个数据集的 MR 测试上的准确率（MR compliance）
3. **RQ1 Key Comparison** — Original vs MR 训练的 Δ 值。正 Δ = MR 训练改善了该测试类型上的表现

RQ1 的理想结论：MR 训练提高 MR compliance（MR test Δ > 0），同时基本保持 standard task performance（Original test Δ ≈ 0 或微正）。

## 烟雾测试

`RQ1/test_data/` 目录包含一套最小 NLI 测试数据集（24 条 train + 5 条/数据集 test + 25 条 MR train），用于快速验证流水线各阶段是否正常工作。

### 生成测试数据

```bash
# 一次性生成（已生成过则跳过）
python RQ1/test_data/generate_data.py
```

生成的数据：

```
RQ1/test_data/
  original/
    sick/train.json              # 24 条三分类 NLI 训练样本
    {snli,mnlim,mnlimm,sick}/test.json   # 各 5 条测试样本
  mr/
    sick_train.json              # 25 条 MR 训练数据（5 pair_id × 5 variants：none + 4 MR）
    {snli,mnlim,mnlimm,sick}_test_MR/    # 各含 2 种 MR × 5 条 + summary
```

### 运行烟雾测试

```bash
# Step 1: 预览所有命令
python RQ1/run_rq1_nli.py --config RQ1/test_data/config_smoke.json \
    --model gemma-3-4b-it --dry-run

# Step 2: 仅运行 sample + convert（无需 GPU，约 30 秒）
python RQ1/run_rq1_nli.py --config RQ1/test_data/config_smoke.json \
    --model gemma-3-4b-it --seeds 99 --target 20 \
    --steps sample convert --output-root /tmp/rq1_smoke

# Step 3: 检查转换结果
head -1 data/ft_datasets/rq1_original_sick_seed99/full_train.json | python -m json.tool
head -1 data/ft_datasets/rq1_mr_sick_seed99/full_train.json | python -m json.tool

# Step 4: 运行完整流水线（需要 GPU + LLaMA-Factory）
python RQ1/run_rq1_nli.py --config RQ1/test_data/config_smoke.json \
    --model gemma-3-4b-it --seeds 99 --target 20 --max-steps 10 \
    --test-batch-size 4 --output-root /tmp/rq1_smoke

# Step 5: 查看结果
cat /tmp/rq1_smoke/SUMMARY.md
cat /tmp/rq1_smoke/_progress.json
```

### 烟雾测试验证要点

| 检查项 | 预期结果 |
|---|---|
| Original 转换格式 | `instruction` 为纯 NLI prompt |
| MR 转换格式 | `instruction` 含 `<reference_premise>` + `Transformation applied` 块 |
| Finetune 完成 | `_progress.json` 中 `finetune: "completed"` |
| 测试输出字段 | JSONL 包含 `premise/hypothesis/gold/pred/correct` |
| SUMMARY.md 表格 | 含 Original Test Sets / MR Test Sets / RQ1 Key Comparison 三张表 |

## 注意事项

### 测试数据路径约定

`test_mettrain_experiment.py` 从以下硬编码路径读取测试数据：

```
data/nli/original_dataset/<ds>/test.json
data/nli/MR_testing/<ds>_test_MR/
```

而非从 config 的 `test_sets` 字段读取。**生产环境中**，RQ1 config 中的 `test_sets` 路径与上述 `data/nli/` 路径一致，不存在问题。**烟雾测试中**若使用自定义数据路径，需要将测试文件放到 `data/nli/` 对应位置，或创建符号链接。当前烟雾配置 (`config_smoke.json`) 中 `test_sets` 指向 `RQ1/test_data/`，仅用于流程验证——sample/convert/finetune 阶段可完全使用自定义数据，test 阶段会自动使用 `data/nli/` 下的真实数据。

### 中间产物

- `data/ft_datasets/rq1_*` — 采样和转换后的中间文件，用于 LLaMA-Factory 训练。由 `data/` 的 gitignore 规则排除。
- `ft_config_*.yaml` — 临时训练的 YAML 配置文件，训练结束后自动删除。若残留（如训练被 kill），已加入 `.gitignore`。
- `data/dataset_info.json` — LLaMA-Factory 数据集注册表，`convert` 阶段会自动追加条目。

### GPU 内存

- Gemma-3-4B (LoRA rank=8, bf16) 约占用 **~8GB** GPU 显存（训练时）
- 推理时加载 base model + LoRA adapter，约占用 **~8GB**
- 使用 `--cuda` 选择特定 GPU 设备

## 环境要求

- Conda 环境：`llmtrain310`
- LLaMA-Factory（含 `llamafactory.cli train`）
- PyTorch, Transformers, PEFT
- 模型需预先缓存到本地（`HF_HUB_OFFLINE=1`）
- Gemma-3-4B 是 gated 仓库，需 `hf auth login` 后下载
