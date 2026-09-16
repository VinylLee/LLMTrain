# SA 数据集导入 + 8 个 MR 实现 + 质量 pilot — 交接说明

对应 `.research/LLMTrain/RQ1_情感分析任务路线_v3(1).md` 的 **Phase 1（导入五域数据、冻结 split）+ Phase 4（MR catalog 与 pilot 质量门）**。
本阶段**不训练模型、不做 AutoCAD/PairCFR、不产出冻结的正式 MR test**。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `scripts/sa_lexicon.py` | 小型透明极性词典：导入期 label 合理性检查、MR 适用性规则、离线测试 |
| `scripts/sa_dataset_registry.py` | 五域声明式 registry：解析器、label 映射、stable ID、split 组装（纯函数，无 IO） |
| `scripts/download_sa_datasets.py` | 下载 → 规范化 → canonical JSONL + hash manifest |
| `scripts/sa_mr_catalog.py` | 8 个 MR 的声明式定义、prompt 模板、适用性规则、盲验证 prompt、catalog hash |
| `scripts/sa_validators.py` | 可插拔极性校验器：`GeneratorSelfValidator` / `LocalClassifierValidator` / `LexiconValidator` / `ScriptedValidator` |
| `scripts/generate_sa_mr.py` | 生成引擎：结构校验 + 关系硬门 + 重试反馈 + resume + 报告 |
| `scripts/sa_relation_metrics.py` | SA 关系级指标（binary flip map），`metamorphic_metrics.py` 零改动 |
| `scripts/run_sa_mr_pilot.py` | pilot 编排：共享 source pool、双校验器、报告与人工复核材料 |
| `scripts/sa_pilot_report.py` | 无 GPU 重算报告、gate risk、Cohen's κ、复核材料生成 |

测试：`tests/test_sa_*.py`、`tests/test_generate_sa_mr.py`（约 175 个用例，全部离线、无网络、无模型）。

## 2. 数据产物

```
data/sa/original_dataset/
  _raw/<dataset>/…                        # 原始下载缓存（含 sha256）
  imdb/{train_source_pool,common_dev,standard_test,mr_test_source_pool}.jsonl
  imdb/human_cad_pairs.jsonl              # 人工 (source, counterfactual) 配对
  sst2/{standard_test,mr_test_source_pool}.jsonl
  amazon|yelp|twitter/{standard_test,mr_test_source_pool}.jsonl
data/sa/registry_manifest.json            # 全部 hash / 计数 / 泄漏检查 / deviations
```

实测规模：IMDb CAD train 1707 / dev 245 / test 488；SST-2 test 1821（切 400 作 MR pool）；
Amazon 106740；Yelp 38000；Twitter 61998。

**注意**：`data/` 被 `.gitignore` 忽略，manifest 默认不受版本控制，而 hash 可追溯性是路线文档的硬要求。
二选一：给 `.gitignore` 加 `!data/sa/registry_manifest.json` 反例，或把冻结 manifest 复制到 `RQ1/configs/sa_manifests/`。

## 3. 8 个 MR

| mr_id | family | relation | 机制 |
|---|---|---|---|
| `sa_uninformative_context` | irrelevant_context_invariance | preserve | LLM |
| `sa_pronoun_substitution` | reference_preserving_invariance | preserve | LLM |
| `sa_voice_switch` | structural_invariance | preserve | LLM |
| `sa_synonym_nonpolar` | lexical_semantic_equivalence | preserve | LLM |
| `sa_tense_shift` | structural_invariance | preserve | LLM |
| `sa_case_reversal` | surface_form_invariance | preserve | **确定性**（无 prompt） |
| `sa_controlled_negation_flip` | task_specific_label_transition | flip | LLM |
| `sa_sentiment_antonym_flip` | task_specific_label_transition | flip | LLM |

`mr_type` 桥接：`preserve→inv`、`flip→flip`，SA 不产生 `neutral`，因此 `pair_id`/`is_source`/`mr_type`
契约与现有 `metamorphic_metrics.py` 兼容。

## 4. 关键设计

- **flip 硬门**：`follow-up 判为 opposite(source_label)` **且** `source 判为 source_label`。
  后一项（source prior）是承重项——缺了它，"source 与 follow-up 都被判 negative" 会被错误接受为
  一个 positive source 的成功翻转。
- **盲验证**：只喂 follow-up 文本，不含 label / source 文本 / MR 名称 / 期望关系，**temperature 0**。
  `TemperatureOverrideGenerator` 复用同一份已加载权重，避免为改一个采样参数而二次占显存。
- **`sa_case_reversal` 特判**：它是唯一规范化文本必须**相等**而原文必须不同的 MR，
  通用 diff / 相似度门对它反向，引擎单独处理，测试钉住该行为。
- **`generator_declined` 记 skip 不记 failure**：模型如实报告"该编辑做不到"是适用性结论，
  不是生成失败；报告里单列计数，避免"模型全部拒绝"伪装成"低接受率"。

## 5. 怎么跑

```bash
conda activate llmtrain310
cd /home/ubuntu/LLMTrain/LLMTrain

# 1 数据导入（无 GPU）。hf-mirror 已内置；GitHub raw 直连
python scripts/download_sa_datasets.py --datasets imdb,sst2,amazon,yelp,twitter \
  --verify-hashes --report data/sa/registry_manifest.json

# 2 单测（无网络/无 GPU/无模型）
python -m pytest tests/ -q

# 3 prompt 预览，不加载模型
python scripts/run_sa_mr_pilot.py --datasets sst2 --sources-per-domain 4 --dry-run

# 4 单 MR 真实 smoke
python scripts/generate_sa_mr.py --dataset imdb \
  --input data/sa/original_dataset/imdb/mr_test_source_pool.jsonl \
  --mr sa_tense_shift --num-samples 6 --seed 42 --offline --device cuda:1 \
  --output data/sa/MR_testing/_smoke/imdb/tense_shift.jsonl

# 5 pilot（共享 source pool，8 个 MR 跑同一套 source）
python scripts/run_sa_mr_pilot.py --datasets imdb,sst2 --sources-per-domain 200 \
  --offline --device cuda:0 --classifier-device cuda:0 \
  --validator-mode generator,classifier --preserve-gate both --flip-gate both \
  --pilot-root data/sa/MR_testing/_pilot

# 6 无 GPU 重算报告 + 人工复核材料
python scripts/sa_pilot_report.py --pilot-root data/sa/MR_testing/_pilot \
  --review-per-mr 50
```

## 6. 成本、批处理与调参（重要）

单次 Gemma-3-4b bf16 生成（SST-2 短句，输出约 15 token）在 **GPU 被其他任务占满**时约 **40 秒**
（首次调用含 CUDA kernel 编译约 118 秒），模型加载约 30 秒、bf16 显存约 9 GB。
batch=1 时每个 group 要 1 次生成 + 1–3 次盲验证，实测 **5–6 分钟/group**——
`200 × 8 × 2 = 3200 groups` 就是数百 GPU 小时。本机 4 张 A40 长期被占至 95–100%。

### 已实现的批处理

`--batch-size N`（默认 8，1 关闭）。`BatchedGenerator` 复用同一份已加载权重与 tokenizer，
把整批 chat prompt 左填充后一次前向，生成与盲验证**都**走批量；
`collect_verdicts_batched` 会按 `(validator, text)` **去重探测**——
flip 类 MR 每个 group 都要回判 source，而这些 source 在 8 个 MR 间重复，去重收益很大。

引擎主循环已改为**按轮（round）批处理**：每轮把该轮所有待处理 group 一次生成、一次验证，
再逐个过门；重试进入下一轮。几个后果要记住：

- **结果按轮落盘**，不再逐组出现。一个 96 group 的运行，第一轮跑完前 `accepted.jsonl` 是空的。
  监控时不要误判为卡死——看 `nvidia-smi` 显存是否在涨。
- **批解码等最慢的序列**。`--max-new-tokens` 要给该域合理上界：短文本域（SST-2）
  别沿用 IMDb 的长文本值，否则一条跑飞的序列会拖住整批。建议 SST-2 用 128–192、IMDb 用 640。
- **采样种子**：一批用该批第一条的 seed。盲验证是贪心解码、不受影响；只有 MR 改写本身
  损失一点逐条种子独立性，已记录在报告 `generation_parameters.batch_size`。
- 确定性 MR（`sa_case_reversal`）不重试：它没有 prompt，门未通过即终局失败
  （这个 bug 已修，并有回归测试钉住）。

### 其它提速手段

1. 等 GPU 空闲（空闲时单条可到 1.5–2 分钟/group）。
2. 缩小 `--sources-per-domain`。
3. 分开跑域：OOD 三个域文本更短，比 IMDb 便宜得多。

## 7. 已验证 / 待验证

已验证：

- 五域导入、split 冻结、hash manifest、跨域泄漏扫描；
- `sa_case_reversal` 确定性输出可逐字符断言；
- 全部 8 个 MR 的结构校验与关系门（含 flip source-prior 反例）在注入式测试下行为正确；
- 真实 Gemma 端到端 smoke：模型加载 → 生成 → 结构校验 → 盲验证 → 写行全部跑通；
- 模型输出带 markdown 代码围栏时 `extract_json_object` 能正确解析；
- NLI 回归：`tests/` 317 passed。

待验证（需要 pilot 跑完）：

- 每个 MR 在 IMDb / SST-2 上的实际接受率与适用率；
- **gate risk**：生成器门与独立分类器的一致程度，即"只用生成器"能否被信任；
- 转换保真度——**自动检查测不到这一项**。smoke 中 `sa_tense_shift` 出现了
  "This is → This was"（正确）与 "I will make fun → I make fun"（时态没改对）、
  "took away → took apart"（掉词）并存的情况；极性保持所以门放行，但转换本身不完美。
  这正是 `samples_for_review.md` + `review_sheet.csv` 人工复核存在的理由，
  也说明 `transformation_validity` 一列不能靠自动检查替代。

## 8. 两个已知偏差

1. **IMDb standard_test 与 mr_test_source_pool 是同一批 CAD test 488 条**
   （用户明确要求，偏离路线文档 §5.2/§5.4）。已写入 machine-readable `deviations` 块，
   下游所有报告必须打印该警告：两者**不是**独立证据。需要干净切分时，
   `stanfordnlp/imdb` 的 25k test 是现成修复路径。
2. **`load_records` 修复**：`scripts/generate_generic_llm_aug.py` 中，单行 JSONL 会被
   `json.loads` 解析成单个对象而被拒绝。该函数文档承诺"JSON 数组或 JSONL 均可"，
   故按单条记录接受。此修复只会把报错变成正确解析，NLI 测试全绿。
