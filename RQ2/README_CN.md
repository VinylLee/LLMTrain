# RQ2：SNLI MR-information representation

本目录保存 RQ2 的 SNLI 受控实验。第一批条件为 `none`、`operation_only`、`pair_only`、`pair_operation` 和 `shuffled_operation`。

## 固定设计

- 基座模型：`google/gemma-3-4b-it`
- 训练源：与 RQ1 对齐的 SNLI MetTrain 数据
- 每个 seed 只冻结一个 pair-aware cohort，所有 mode 复用该 cohort 和 split manifest
- merged test：`RQ2/data/test/mr_test_data_merged/snli.jsonl`
- 推理统一使用普通三分类 NLI prompt，不提供 MR specification
- 指标：source accuracy、MR/follow-up accuracy、MSR、joint correctness

## 运行顺序

```bash
python RQ2/run_rq2_snli.py --seeds 42 --steps audit sample convert
python RQ2/run_rq2_snli.py --seeds 42 --steps finetune test summarize --smoke --resume
python RQ2/run_rq2_snli.py --seeds 42 --steps finetune test summarize --resume
```

`--smoke` 的 20 steps 仅用于 pipeline health check。当前项目的 MR-as-Instruction 数据质量门槛和 Human Validation 状态仍不是确认性通过，因此本目录结果在审计完成前只能作为 exploratory evidence。
