# SA MR 数据管线 —— 交接文档（给接手的 agent）

> 最后更新：2026-09-17
> 上游设计文档：`.research/LLMTrain/RQ1_情感分析任务路线_v3(1).md`
> 覆盖范围：该文档 §15 的 **Phase 1（导入五域、冻结 split）+ Phase 4（MR catalog 与 pilot 质量门）**
> 工作仓库：`/home/ubuntu/LLMTrain/LLMTrain`（实验端）

---

## 0. 先读这一节：当前结论与卡点

**做完了什么**：五域数据导入并冻结；8 个 SA MR 全部实现、有测试；IMDb + SST-2 的 pilot 跑完两轮（90 个 accepted group）；**两名标注者的人工复核也已完成**。

**最重要的结论：MR catalog 不能按现状冻结。** 人工复核显示 **8 个 MR 里有 6 个的 transformation validity 是 0%，远低于预注册的 ≥0.95**。

| MR | 复核者 A | 复核者 B | 判定 |
|---|---:|---:|---|
| `sa_uninformative_context` | **100%** (8/8) | **100%** (8/8) | ✅ 唯一干净通过 |
| `sa_sentiment_antonym_flip` | 88% (7/8) | 43% (3/7) | ⚠️ 分歧大，需重判 |
| `sa_tense_shift` | 38% (3/8) | 25% (2/8) | ❌ |
| `sa_case_reversal` | 12% (1/8) | 100% (1/1 已判) | ⚠️ **是规范缺陷，不是数据缺陷**（见 §4.2） |
| `sa_controlled_negation_flip` | 0% (0/6) | 0% (0/6) | ❌ |
| `sa_pronoun_substitution` | 0% (0/6) | 0% (0/6) | ❌ |
| `sa_synonym_nonpolar` | 0% (0/7) | 0% (0/7) | ❌ |
| `sa_voice_switch` | 0% (0/8) | 0% (0/8) | ❌ |

**总体 transformation validity：A 19/59 = 32.2%，B 14/51 = 27.5%。** 门槛 95%。

**根因（两名复核者独立得出同一结论）**：**Gemma-3-4b 不做受控最小编辑，它做自由改写。** 它不是"加一个否定词"，而是直接把 `overstuffed` 换成 `remarkably insightful`；不是"只换非情感词"，而是把核心评价词 `underwritten` 换成 `inadequately supported`。请复核者原话：

> 「没有进行受控否定操作；直接把负向 "overstuffed" 重写成正向 "remarkably insightful"。」
> 「underwritten 是核心负向评价词，却被替换为 inadequately supported；既越过 nonpolar 限制，也把人物/剧本写作不足换成支持不足的义项。」

**关键对照**：`relation_valid` 是 **59/59 和 57/57 全 yes**。也就是说 **极性关系全都成立，失败的纯粹是"有没有真的执行那个操作"**。这正好印证了当初把「结构 + 极性」和「转换保真度」分成两道门的判断——自动检查能过全部，人一票否决。

**因此下一步不是改数据，而是改生成方式**（见 §6）。

---

## 1. 本阶段不做什么

- **不训练模型**、不接 AutoCAD / PairCFR、不产出冻结的正式 MR test。
- MR catalog 冻结发生在质量门通过之后。**现在没通过，所以后续所有训练都还不能启动。**

---

## 2. 新增文件

### 2.1 脚本（`scripts/`，全部由本次工作新增）

| 文件 | 职责 |
|---|---|
| `sa_lexicon.py` | 小型透明极性词典：导入期 label 合理性检查、MR 适用性规则、离线测试 |
| `sa_dataset_registry.py` | 五域**声明式** registry：解析器、label 映射、stable ID、split 组装。**纯函数，无 IO** |
| `download_sa_datasets.py` | 唯一 IO 层：下载 → 规范化 → canonical JSONL + hash manifest |
| `sa_mr_catalog.py` | 8 个 MR 的**声明式**定义、prompt 模板、适用性规则、盲验证 prompt、`catalog_sha256()` |
| `sa_validators.py` | 可插拔极性校验器：`GeneratorSelfValidator` / `LocalClassifierValidator` / `LexiconValidator` / `ScriptedValidator` |
| `generate_sa_mr.py` | 生成引擎：结构校验 + 关系硬门 + 重试 + resume + **批量生成** + 报告 |
| `sa_relation_metrics.py` | SA 关系级指标（binary flip map）。`metamorphic_metrics.py` **零改动** |
| `run_sa_mr_pilot.py` | pilot 编排：共享 source pool、双校验器、报告与复核材料 |
| `sa_pilot_report.py` | **无 GPU** 重算报告、gate risk、Cohen's κ、复核材料生成 |

**架构要点**：MR 知识**全在 `sa_mr_catalog.py`，引擎不含任何具体 MR 逻辑**。
加一个新 MR = 在 catalog 加一条 `MRDefinition`，不用动引擎。改 prompt 则必须 bump `PROMPT_VERSION`
（`catalog_sha256()` 会让数据失效可检测）。

### 2.2 测试（`tests/`，全部离线、无网络、无模型、CPU）

`test_sa_dataset_registry.py` / `test_sa_mr_catalog.py` / `test_sa_validators.py` /
`test_generate_sa_mr.py` / `test_sa_relation_metrics.py` / `test_sa_pilot_report.py` / `test_sa_batching.py`

`python -m pytest tests/ -q` → **510 passed, 1 failed**。
那 1 个 failed 是**既有的、与本次工作无关**：`test_rq1_back_translation` 要求
`data/nli/back_translation/*.jsonl`，该文件在本机不存在。改动前就失败，已用 `git show HEAD:` 版本验证过。

---

## 3. 数据与 pilot 产物

```
data/sa/original_dataset/
  _raw/<dataset>/…                       # 原始下载缓存
  imdb/{train_source_pool,common_dev,standard_test,mr_test_source_pool}.jsonl
  imdb/human_cad_pairs.jsonl             # 1707 组人工 (source, counterfactual)
  sst2/{standard_test,mr_test_source_pool}.jsonl
  amazon|yelp|twitter/{standard_test,mr_test_source_pool}.jsonl
data/sa/registry_manifest.json           # hash / 计数 / 泄漏检查 / deviations

data/sa/MR_testing/_pilot_v4/            # pilot 结果（下面全部结论的来源）
  pilot_report.{json,md}                 # 两域合并报告
  samples_for_review.md                  # 59 块复核材料（source/follow-up/自动检查裁决）
  review_sheet.csv                       # 空白模板
  review_sheet_completed_1.csv           # ← 复核者 A 已填
  review_sheet_completed_2.csv           # ← 复核者 B 已填
  codebook.md                            # 预注册阈值
  {sst2,imdb}/accepted.jsonl             # 90 个 accepted group
  {sst2,imdb}/accepted.jsonl.rejects.jsonl
  imdb/accepted.jsonl.rejects.jsonl.crashed   # 一次崩溃的残留，忽略
```

规模：IMDb CAD **1707 / 245 / 488**；SST-2 test 1821（切 400 作 MR pool）；
Amazon 106740；Yelp 38000；Twitter 61998。

---

## 4. Pilot 结果

### 4.1 自动指标

共 **90 个 accepted group**（SST-2 58 + IMDb 32），12 source × 8 MR × 2 域。

**Gate risk**（「只用生成器」这个门，独立分类器会不同意多少）：

| scope | generator 门接受 | 分类器不同意 | false-accept 估计 |
|---|---:|---:|---:|
| overall | 105 | 15 | **14.3%** |
| **imdb** | 32 | 0 | **0.0%** |
| **sst2** | 73 | 15 | **20.5%** |

结论分域：生成器自校验在主训练域（IMDb）可信，在短文本 OOD 域（SST-2）不可信。
注意 0/32 的 95% 置信上界约 9%，不是字面零。分类器自身精度：IMDb **12/12**、SST-2 **10/12**。

**`dataset × MR` applicability / acceptance**：

| MR | IMDb | SST-2 |
|---|---|---|
| `sa_uninformative_context` | 75.0% / 77.8% | 91.7% / 81.8% |
| `sa_pronoun_substitution` | **100% / 50.0%** | **41.7% / 40.0%** |
| `sa_voice_switch` | 100% / **25.0%** | 75.0% / **88.9%** |
| `sa_synonym_nonpolar` | 100% / **0.0%** | 91.7% / **81.8%** |
| `sa_tense_shift` | 100% / 33.3% | 75.0% / 66.7% |
| `sa_case_reversal` | **66.7%** / 75.0% | **100%** / 75.0% |
| `sa_controlled_negation_flip` | 41.7% / 60.0% | 100% / 50.0% |
| `sa_sentiment_antonym_flip` | 41.7% / 60.0% | 100% / 75.0% |

**适用率在两个域之间几乎互换**——代词替换在长评论 100%、短句 41.7%；flip 类反过来。
不能拿一个域的结果推另一个域（路线文档 §6.6 的预期）。

### 4.2 人工复核（关键，且发现了一个规范缺陷）

**标注者间一致性**：

| 问题 | observed agreement | Cohen's κ |
|---|---:|---:|
| `transformation_valid` | 0.80 | **0.59** ← 低于门槛 0.80 |
| `source_label_correct` | 1.00 | n/a（无方差） |
| `followup_label_correct` | 0.97 | 0.00（样本几乎全 yes，κ 不稳定，无意义） |
| `relation_valid` | 0.97 | 0.00（同上） |

**`sa_case_reversal` 的低分是规范缺陷，不是数据问题。** 复核者 B 在 notes 里写得很清楚：

> 「附件只列 case_reversal 名称，**未说明允许 upper/lower 还是要求逐字符 swapcase**；此例原 A 及 Catholic 的 C 未反转。前者可判 yes，后者应判 no，故操作记 ?。」

复核者 A 按**严格的逐字符 swapcase** 判，于是 8 组里 7 组判 no；复核者 B 因为定义缺失，7 组记 `?`。
而实际实现（`sa_mr_catalog.apply_case_reversal`）是**按 `sha256(source_id)` 在 `lower/upper/swap` 三选一**——
**两者都没错，是 codebook 没写清楚。**

→ **修法：在 codebook 里写明三模式规则，然后让复核者重判这 8 组。** 这 8 组很可能大部分翻成 yes。
这也是 κ=0.59 的主要来源，重判后 κ 大概率回升。

### 4.3 每个 MR 失败的具体原因（复核者原话摘录）

这些直接决定 prompt 该怎么改：

- **`sa_controlled_negation_flip`（0%）**：模型用**换评价词**来翻转，而不是加/删否定。
  「没有进行受控否定操作；直接把负向 overstuffed 重写成正向 remarkably insightful。」
  → 该 MR 与 `sa_sentiment_antonym_flip` **塌缩成了同一个操作**，因为模型自选最省力的翻转机制。
- **`sa_synonym_nonpolar`（0%）**：模型替换了**情感承载词**。「underwritten 被换成 inadequately supported，
  既越过 nonpolar 限制……按语境识别情感承载短语，而非只看词典正负词表。」
- **`sa_voice_switch`（0%）**：只做局部被动化，其余大量改写。
  「仅 columns pass off→were passed off 等局部被动化；MST3K gave 等可转换主句未改。」
- **`sa_pronoun_substitution`（0%）**：没做代词替换，而是删旁白、缩写人名。
  「Howie Long→Long 是姓名缩写，不是代词替换；还删除负向修饰语 long-suffering。」
- **`sa_tense_shift`（38%/25%）**：时好时坏。「would be rolling→would roll 主要改变进行体，
  并未完成时态转换；It's obvious、columns pass off 等基本未变。」
- **`sa_uninformative_context`（100%）**：唯一稳定通过。加一句无关中性句是个"加法"操作，
  模型不需要做外科式编辑。

**共同模式**：凡是要求**受控最小编辑**（只改一个语法特征、只换某一类词）的 MR 全部失败；
唯一成功的是**加法式**操作。这是 4B 模型的能力边界，不是 prompt 微调能完全解决的。

---

## 5. 已修 bug（勿重新引入）

**其中 #5/#6/#7 都属于「一个异常杀掉整轮实验」类**，在长任务里代价极高。

| # | 问题 | 位置 | 状态 |
|---|---|---|---|
| 1 | 单行 JSONL 被 `load_records` 拒绝（解析成单个对象） | `generate_generic_llm_aug.py`（**共享 NLI 代码**） | 已修 |
| 2 | 盲验证继承了生成器的 `temperature=0.8`，给两个方向的判定都注入噪声 | `sa_validators.py` + 引擎 | 已修 |
| 3 | NLI 指标返回**百分数**、SA 返回**小数**，混在同一份报告里 | `sa_relation_metrics.py` 边界归一 | 已修 |
| 4 | orphan follow-up 被误计入 ambiguous source | `sa_relation_metrics.py` | 已修 |
| 5 | 确定性 MR 门失败后进入重试循环 → `build_mr_prompt` 抛错 → **整轮中止** | `generate_sa_mr.py` | 已修 |
| 6 | `torch._dynamo` `cache_size_limit`(8) 被批处理的多种 batch shape 撑爆 → **整轮中止** | `relax_dynamo_cache_limit()` | 已修 |
| 7 | `extract_json_object` 抛**基类** `GenerationError`，循环只捕获子类 → 一次格式错误输出 **整轮中止** | `validate_mr_output` 包一层 | 已修 |
| 8 | `meta_leak` 误杀原文自带短语（`here is the` / `I cannot`），5 组被错拒 | `run_structural_checks` 改为对 source 取差集 | 已修 |
| 9 | `--agreement` 模式下 `--pilot-root` 仍是必填，导致复核一致性命令**根本跑不起来** | `sa_pilot_report.py` 改为按需校验 | 已修 |
| 10 | 复核表带 UTF-8 BOM，表头变成 `﻿group_id` → `KeyError` | `_read_csv` 改用 `utf-8-sig` + 明确报错 | 已修 |

每个都有回归测试钉住。**只有 #1 动了共享 NLI 代码**，且只会把报错变成正确解析，NLI 测试全绿。

> #9 / #10 是在写本文档、逐条验证"怎么跑"里的命令时发现的 —— 也就是说 §7 第 6 条命令
> 在修复前**对任何人都会失败**。接手时请照 §7 实际跑一遍，不要假设命令可用。

---

## 6. 建议的下一步（按优先级）

### 6.1 先做（便宜、能立刻解锁几个 MR）

1. **修 codebook 的 `sa_case_reversal` 定义**，写明 `lower/upper/swap` 三模式，让复核者重判那 8 组。
   预期能救回一个 MR，并把 κ 从 0.59 拉回门槛附近。
2. **记录被拒的模型输出**。现在 reject 记录只存 reason token、不存 response 文本。
   这次诊断 `sa_synonym_nonpolar` 在 IMDb 上 0/12 时，只能从 similarity 数值（0.014–0.116）反推
   "模型没做最小编辑"，看不到它到底写了什么。加一个字段即可。
3. **单独跑 `sa_pilot_report.py` 时报告头部 `Validators` / `Pilot gate` 为空**
   （那两个字段只有 pilot 主程序写进 `run_manifest`）。纯观感问题，不影响指标。

### 6.2 核心工作：重做生成方式

证据指向同一个结论：**指望 4B 模型"自由发挥地做受控编辑"是错的**。建议方向：

- **改成显式定位 + 受控替换**：让模型先输出"要改哪个 span"，再确定性地替换，
  而不是让它自由重写全文。这类"先抽取再应用"的两段式，比单次生成可控得多。
- **收紧 prompt 的输出契约**：要求回显被修改的片段与原文片段，自动比对是否只动了该动的地方。
- **或者换更强的生成器**（更大模型 / API），但这会改变 §5 的可复现性论证，需要重新记录 provenance。
- **加法式 MR 保留**：`sa_uninformative_context` 已证明可行，直接留。
- **`negation_flip` 与 `antonym_flip` 要考虑合并或重新定义**——现在模型无法区分两者。

### 6.3 扩容 pilot

现在每域只有 **12 个 source**，复核样本最多的 MR 才 7 组，远低于路线文档 §8.2 要求的
「每个 MR 分层抽查 ≥50 组」。**这批只能当方向性信号 + 流程演练**，不足以支撑正式质量门。
改完生成方式后需要把 pool 扩到 ≥60 source/域再复查。

### 6.4 尚未开始的

- Amazon / Yelp / Twitter 三域**还没做 pilot**（数据已导入，MR pool 已预切 400）。
- 正式 MR test 生成、token/step/label 预算匹配、baseline（AutoCAD / PairCFR）接入。
- **`data/` 被 `.gitignore` 忽略**，`registry_manifest.json` 与 pilot 报告都不受版本控制，
  而 hash 可追溯性是路线文档硬要求。二选一：加 `!data/sa/**/*.manifest.json` 反例，
  或把冻结 manifest 复制到受控的 `RQ1/configs/sa_manifests/`。

---

## 7. 怎么跑

```bash
conda activate llmtrain310
cd /home/ubuntu/LLMTrain/LLMTrain

# 1 数据导入（无 GPU）。先 dry-run 看计划
python scripts/download_sa_datasets.py --datasets imdb,sst2 --dry-run
python scripts/download_sa_datasets.py --datasets imdb,sst2,amazon,yelp,twitter \
  --verify-hashes --report data/sa/registry_manifest.json

# 2 单测（无网络 / 无 GPU / 无模型）
python -m pytest tests/ -q

# 3 prompt 预览，不加载模型、不写文件
python scripts/run_sa_mr_pilot.py --datasets sst2 --sources-per-domain 4 --dry-run

# 4 真实 pilot。分域跑，因为两域的 max-new-tokens 需求差很多
python scripts/run_sa_mr_pilot.py --datasets sst2 --sources-per-domain 12 --seed 42 \
  --offline --device cuda:1 --classifier-device cpu \
  --validator-mode generator,classifier --preserve-gate both --flip-gate both \
  --max-new-tokens 192 --max-retries 1 --batch-size 8 --review-per-mr 8 \
  --pilot-root data/sa/MR_testing/_pilot_v4

# 5 无 GPU 重算报告 + 复核材料
python scripts/sa_pilot_report.py --pilot-root data/sa/MR_testing/_pilot_v4 \
  --datasets sst2,imdb --review-per-mr 8

# 6 标注者一致性（读已完成的复核表）
python scripts/sa_pilot_report.py --agreement \
  data/sa/MR_testing/_pilot_v4/review_sheet_completed_1.csv \
  data/sa/MR_testing/_pilot_v4/review_sheet_completed_2.csv \
  --agreement-column transformation_valid
```

---

## 8. 环境陷阱（踩过的，很重要）

- **HF 下载**：`huggingface.co` 无 DNS，v2rayA 代理隧道对 HF 会 TLS 握手后超时。
  必须 `export HF_ENDPOINT=https://hf-mirror.com`；`resolve/main/...` 会 302 到
  `cas-bridge.xethub.hf.co`，**要跟随重定向**。GitHub raw 直连可用但吞吐不稳
  （同一个仓库，`original17/train.tsv` 秒下完，`revised17/train.tsv` 只有 4.6 KB/s）。
- **GPU 长期被占满**：4×A40 常年 90–100% 利用率，显存被占 20–37GB。
  用 `nvidia-smi --query-compute-apps` 看不到自己的 PID 是正常的（namespace 问题），
  **用显存增量判断进程有没有真的上卡**。
- **Gemma-3-4b 成本**：bf16 约 9GB；加载约 30s；首次 generate（含 CUDA kernel 编译）约 118s；
  之后 batch=1 单条约 **40s**。classifier 可以放 CPU（`--classifier-device cpu`）省显存。
- **批解码等最慢的序列**：`--max-new-tokens` 要给该域的合理上界。短文本域（SST-2）
  别沿用 IMDb 的长文本值，否则一条跑飞的序列拖住整批。建议 SST-2 128–192、IMDb 640。
- **结果按轮落盘**，不是逐组。一个 96 组的运行，第一轮跑完前 `accepted.jsonl` 是空的——
  **别误判为卡死**，看 `nvidia-smi` 显存有没有在涨。
- **用 `-u` 跑 Python**：`conda run` 会缓冲输出，`| grep | tail` 更会把缓冲吞掉，
  长时间任务完全看不到进度。
- 分类器本地路径：`models/siebert/sentiment-roberta-large-english`（约 1.5GB）。
  `run_sa_mr_pilot.py` 会自动优先用它，没有才回退 hub id。

---

## 9. 仓库分工（重要）

本项目是**两个仓库分离**的：

- **`/home/ubuntu/MTrain`** —— MR **增效数据生成端**。NLI/SNLI 的生成管线在这里
  （`augment_snli_all_label_mrs.py`、`MRAugment.py`、`AUGMENTATION_PIPELINE.md`）。
  **目前没有 SA/情感分析的任何代码**（已 grep 确认）。
- **`/home/ubuntu/LLMTrain/LLMTrain`** —— **实验端**。本文档描述的 SA 管线建在这里。

考虑到 MTrain 才是"生成"的归属地，**SA 生成器的后续改造（§6.2）可能需要决定放在哪一侧**，
并与 `AUGMENTATION_PIPELINE.md` 里已有的"生成 → 确定性过滤 → 独立判题 → 选择"四段式流程对齐。
接手前建议先读 `/home/ubuntu/MTrain/AUGMENTATION_PIPELINE.md`。

注意：本次改动**没有 commit**。当前分支 `feature/rq2-v4-control-expansion-20260916-110831`
（注意不是最初工作时的 `fix/mrinstr-pilot-v2`）。

---

## 10. 文件索引

```
RQ1/SA_MR_PIPELINE_HANDOFF.md    ← 本文档
scripts/sa_*.py                  ← 引擎与工具（见表 §2.1）
scripts/download_sa_datasets.py  scripts/generate_sa_mr.py  scripts/run_sa_mr_pilot.py
tests/test_sa_*.py  tests/test_generate_sa_mr.py
data/sa/…                        ← 数据与 manifest（gitignored）
data/sa/MR_testing/_pilot_v4/    ← pilot 结果、复核材料、两份已完成的复核表
/home/ubuntu/MTrain/AUGMENTATION_PIPELINE.md  ← 生成端既有流程（NLI）
.research/LLMTrain/RQ1_情感分析任务路线_v3(1).md  ← 上游设计文档
```
