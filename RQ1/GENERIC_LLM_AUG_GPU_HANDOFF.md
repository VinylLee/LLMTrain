# Generic-LLM-Aug RQ1 对比实验 GPU 执行记录

## 当前本地状态

本实验已完成 seed 42/43/44 的 `sample`、`convert`、LoRA 微调和 merged 测试（3/3 成功）。
三个 seed 共生成 12 个完整 merged JSONL；最终结果见
`RQ1/output/generic_llm_aug_gemma3_4b/RESULTS.md` 和 `rq1_report.json`。

输出根：

```text
RQ1/output/generic_llm_aug_gemma3_4b/
```

实验目录：

```text
rq1_generic_llm_aug_snli_seed42
rq1_generic_llm_aug_snli_seed43
rq1_generic_llm_aug_snli_seed44
```

每个 seed 当前均为 5,073 train / 267 validation，转换模式为普通 NLI SFT（内部兼容字段为 `mr_instruction_mode=none`，不含 MR instruction）。

训练配置：`google/gemma-3-4b-it`，LoRA rank 8，learning rate `3e-4`，3 epochs，
per-device batch 4，gradient accumulation 8，bf16；每个 seed 477 个优化 steps。
测试使用 merged source+MR 单次推理，batch size 32，覆盖 SNLI、MNLIm、MNLImm、SICK。

## 必须同步的文件

- 当前源码及 `RQ1/configs/rq1_nli_config.json`；
- `models/google/gemma-3-4b-it/`，或远程已验证的等价 Gemma 缓存；
- `data/nli/generic_llm_aug/training/snli_generic_llm_aug_5340_seed42.jsonl`；
- `data/nli/generic_llm_aug/training/snli_generic_llm_aug_5340_seed43.jsonl`；
- `data/nli/generic_llm_aug/training/snli_generic_llm_aug_5340_seed44.jsonl`；
- `data/ft_datasets/rq1_generic_llm_aug_snli_seed42/`；
- `data/ft_datasets/rq1_generic_llm_aug_snli_seed43/`；
- `data/ft_datasets/rq1_generic_llm_aug_snli_seed44/`；
- `data/dataset_info.json` 中三个 `rq1_generic_llm_aug_snli_seed*` 注册项；
- `RQ1/output/generic_llm_aug_gemma3_4b/_progress.json`。

不要同步或读取 `data/nli/generic_llm_aug/training/other/` 作为训练输入。

## 输入文件 SHA-256

```text
seed42  35332adf7cded55bbb91facf0537f145063cb017b4dd14ecd95b1494bdcc133f
seed43  125059b834e2cd5a5d752ee1f38d314463245784dc68d1dcb3be0e1f94fe7faf
seed44  79ae5c14acff72cd75192726e1543929ba320432dc01576bee05e190ba29eed3
```

## 远程执行命令

如果已经同步本地 `sample/convert` 产物，执行：

```bash
python RQ1/run_rq1_nli.py \
  --config RQ1/configs/rq1_nli_config.json \
  --model gemma-3-4b-it \
  --experiment generic_llm_aug \
  --train-data snli \
  --seeds 42 43 44 \
  --output-root RQ1/output/generic_llm_aug_gemma3_4b \
  --steps finetune test_merged \
  --resume
```

不要传入 `--mr-instruction-mode`、`--lr`、`--epochs`、`--rank`、`--batch`、`--grad-accum` 或 `--max-steps` 覆盖配置。

如果远程没有同步中间转换产物，才使用完整流水线命令：

```bash
python RQ1/run_rq1_nli.py \
  --config RQ1/configs/rq1_nli_config.json \
  --model gemma-3-4b-it \
  --experiment generic_llm_aug \
  --train-data snli \
  --seeds 42 43 44 \
  --output-root RQ1/output/generic_llm_aug_gemma3_4b \
  --resume
```

## 预期产物与检查

## 已完成检查（2026-09-11）

- `_progress.json` 中三个实验的 `sample`、`convert`、`finetune`、`test_merged` 均为 `completed`；
- 三个 `model/` 目录均含可加载 LoRA adapter；
- 12 个 merged JSONL 行数均与标准 merged 测试集一致（22,548 / 22,423 / 22,416 / 10,655）；
- 逐行 `correct` 字段已直接重算，12 个文件均无缺失/非布尔字段；
- 未读取或使用 `data/nli/generic_llm_aug/training/other/`，旧 `RQ1/output/gemma-3-4b-it/` 未修改。


每个 seed 应生成一个 LoRA adapter 和四个 merged 测试 JSONL（SNLI、MNLIm、MNLImm、SICK）。完成后检查：

- `_progress.json` 中三个实验的 `finetune` 和 `test_merged` 均为 `completed`；
- 三个 `model/` 目录存在且可加载；
- 共 12 个 merged JSONL；
- 每条结果含 `gold`、`pred`、`correct`，并从逐行 `correct` 重算结果；
- 汇总 Source Accuracy、Follow-up Accuracy、MSR 和 Joint Correctness；
- 不修改 `RQ1/output/gemma-3-4b-it/` 或其中已有 `RESULTS.md`、`rq1_report.json`。
