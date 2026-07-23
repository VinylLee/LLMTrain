# MR-as-Instruction GPU 实验执行交接

更新时间：2026-07-23 18:37（Asia/Shanghai）

## 1. 交接边界

这不是整个项目迁移。远程服务器只承担需要 GPU 的训练、推理与评测；本机仍是源码、研究审计、结果分析和实验决策的主工作区。

当前暂停点：受限 Stage 3 exploratory full seed-42 训练已完成；Stage 4 尚未启动。不要在远程服务器自行开始下一条 GPU 命令。先在本机分析现有训练曲线和实验设计，确定下一轮 GPU 工作后，再生成明确命令。

研究口径保持不变：Stage 2=`FAIL`，原 training gate=`BLOCKED`，Human Validation=`QUICK_AUDIT_ONLY_NOT_PASSED`，`confirmatory_use_allowed=false`。所有现有结果仅可作 exploratory evidence，不得写成 Human Validation PASS 或确认性结论。

## 2. 源码与配置身份

- 分支：`fix/mrinstr-pilot-v2`
- 当前 HEAD：`d0ccf2e2ef6745c5e17d1f712576c76860022f18`
- 当前还有未提交修改，远程执行前必须先通过后续 commit/push 或等价补丁同步：
  - `scripts/run_batch_experiments.py`
  - `tests/test_run_batch_experiments.py`
  - `experiments_config_mrinstr_exploratory_full_seed42.json`
  - `PROJECT_OVERVIEW.md`
  - `AGENTS.md`
  - `GPU_EXECUTION_HANDOFF.md`
- full 配置 SHA-256：`cb782620d2697658f9856fe10bcb19786d4ee2daf0a6ced4c25224e1c53f214d`
- Stage 2 报告 SHA-256：`8bc96a9b8fd1bfd8ae63e073535ba1ff523194ab4a649e5956a3d57ced12e984`
- quick audit SHA-256：`7394a1b79870d241d01cac7bfd1ab9c8001944d18562abec0dda4ca7cad8b12c`
- CUDA 默认设备：`0`
- 本机环境名：`llmtrain310`
- 本地 Gemma 缓存：`models/google/gemma-3-4b-it/`，约 8.05 GiB；远程可使用同一路径结构或重新缓存。

## 3. 已完成 GPU 训练

统一配置：Gemma 3 4B IT、seed 42、train 4174 / validation 239、LoRA rank 8、LR 3e-4、batch 4、gradient accumulation 8、3 epochs、393 steps。

| mode | 状态 | 最终 eval loss | 最终 adapter SHA-256 | run signature |
|---|---:|---:|---|---|
| `none` | completed | 0.0984746963 | `4229549eab494f3ca7cbdd53904311ae2e2786d8edde786a379cf0f87a90e60e` | `6e77015fb24480d3c35e2e7f5848a6ed2770d624a094ce1ddc529f0fd2147717` |
| `pair_operation` | completed | 0.1686632782 | `944646e0db2894ca2b08e5ad8643dcaaed154fb868b503897bb1b324fdba4589` | `28201deead8946c01899a46b9a3ba3e8c9ad475e3c146965215c25438945a404` |
| `shuffled_operation` | completed | 0.1306167543 | `390690be57e0a52df7e85a1d061ae9b10e61231e13e509bbaacb714533d867fe` | `81b4662dc6d5aae575d38342fd57308326ea35e0f27d48a201166c5088bf8459` |

输出根：`output/experiments/gemma3_4b_mrinstr_exploratory_full_seed42_v1/`。`_progress.json` 的三个 `finetune` 均为 `completed`。

`none` 首次运行在日志 340 步后被外层 2 小时监控上限终止；完整 checkpoint 只到 300。runner 随后从 `checkpoint-300` 恢复 optimizer、scheduler 和 RNG 状态，并完成 393 步。因此其最终 adapter 和 eval 有效，但根目录 `train_results.json` 的 train runtime/loss 只覆盖恢复段，不可与另外两组直接比较。

三组均出现中期 eval 优于最终 eval：`none` 日志最佳约为 step 200 / 0.07275，`pair_operation` 为 step 200 / 0.10652，`shuffled_operation` 为 step 150 / 0.09253。但 `save_total_limit=2` 已删除这些中期 checkpoint，目前每组仅保留 `checkpoint-350` 和 `checkpoint-393`。这些日志最佳值不是当前可加载模型，后续不能假装能直接评测“最佳 checkpoint”。

## 4. 远程 GPU 工作的最小传输集

下一轮具体任务由本机分析后决定。若下一轮是评测当前最终 adapter，远程至少需要：

1. 已同步的源码与最终实验配置；
2. `models/google/gemma-3-4b-it/`，或远程已验证的等价 Hugging Face 缓存；
3. 三个实验目录中除 `checkpoint-*` 外的最终 adapter 与元数据；每组约 88.8 MiB；
4. `data/nli/original_dataset/mnlim/test.json` 与 `data/nli/MR_testing/mnlim_test_MR/`；
5. `data/dataset_info.json`；
6. 每组的 `experiment_meta.json`、`run_signature.json`、`stage2_gate_decision.json`、`model/train_results.json`、`model/eval_results.json` 和 `model/trainer_log.jsonl`；
7. 输出根 `_progress.json`。

若远程需要断点续训或完整训练状态，才额外复制 `checkpoint-*`。每组含两个 checkpoint 时总目录约 494.5 MiB；仅做最终 adapter 推理无需复制 checkpoint。

远程 GPU 产物完成后，应按原相对路径带回本机，同时保留 run signature、gate decision、命令/配置、日志和逐行预测文件；本机再进行汇总、统计分析与下一轮决策。

## 5. 当前明确禁止/暂停的动作

- 不启动 Stage 4 Original/MR 全量评测；
- 不把 quick audit 或未填写双人模板写成 Human Validation PASS；
- 不把 exploratory loss 当成模式优劣结论；
- 不在没有本机分析决策的情况下自行扩展 seed、mode、epoch 或输出根；
- 不覆盖现有输出目录，也不删除旧失败现场或 checkpoint。

## 6. 本机下一步（不需要 GPU）

等待用户下一次指令后，本机应先分析：训练/验证曲线、3-epoch 过拟合、最终 adapter 与已删除中期最佳 checkpoint 的影响，以及 Stage 4 应评测最终 adapter还是重新训练并启用 `load_best_model_at_end`/更安全的 checkpoint 保留策略。只有决策完成后，才向远程服务器下发下一条 GPU 实验命令。
