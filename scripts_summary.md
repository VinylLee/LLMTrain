# scripts 目录代码逻辑总结

本文档按脚本逐个总结 `scripts/` 目录中的全部脚本文件，包括 `.py`、`.sh` 以及 `test_` 前缀脚本。每个脚本都从四个角度说明：代码逻辑、输入输出、调用关系、潜在问题。

## 目录总览

### 主实验与数据流
- `run_batch_experiments.py`
- `sample_mettrain_pairid.py`
- `convert_nli_to_ft.py`
- `run_experiment.py`
- `run_finetune.py`
- `summarize_seed_experiments.py`

### 推理、评估与报告
- `test_llm.py`
- `run_all.sh`
- `evaluate.py`
- `mrv.py`
- `mrv_excel.py`
- `test_deepseek.py`
- `run_lmstudio_qwen.sh`
- `run_qwen35_flash.sh`
- `prepare_gemma.sh`
- `test_original_data.py`

### 测试与烟雾脚本
- `test_ft_model.py`
- `test_lora_finetune.py`
- `test_mettrain_experiment.py`

## 1. `run_batch_experiments.py`

### 代码逻辑
- 读取 `experiments/configs/experiments_config.json` 或指定配置。
- 将实验名与 seed 组合成 `name_seedX`。
- 按阶段执行采样、转换、微调、Original 测试、MR 测试。
- 维护 `_progress.json` 作为断点续跑状态。
- 动态生成 LoRA 训练 YAML，再调用 LLaMA Factory。
- 训练结束后汇总生成 `SUMMARY.md` 或 `RESULTS.md`。

### 输入输出
- 输入：配置文件、seed 列表、数据集路径、模型名、模板名、CUDA 编号等。
- 输出：`output/experiments/<exp_seed>/` 下的 `model/`、`tests/original/`、`tests/mr/`、`experiment_meta.json`、`_progress.json`、汇总文档。

### 调用关系
- 调用 `sample_mettrain_pairid.py` 完成采样。
- 调用 `convert_nli_to_ft.py` 完成数据格式转换与注册。
- 调用外部训练命令完成 LoRA 微调。
- 调用 `test_mettrain_experiment.py` 做测试。

### 潜在问题
- 工作目录由脚本位置自动推导，可在不同机器复用。
- 使用 `subprocess.run(..., shell=True)`，对配置输入较为敏感。
- 依赖输出目录和配置命名约定，改目录结构会影响整条流水线。
- 批量实验一旦中途失败，进度文件与实际状态可能不完全一致。

## 2. `sample_mettrain_pairid.py`

### 代码逻辑
- 读取输入 JSONL。
- 按 `pair_id` 分组，保证原样本与 MR 变体不被拆散。
- 可选分层采样，按组内多数标签近似分层。
- 随机抽取组，直到达到目标样本量。

### 输入输出
- 输入：训练 JSONL、目标条数、seed、可选 `--stratify`。
- 输出：采样结果文件和 pair_id 记录文件。

### 调用关系
- 被 `run_batch_experiments.py` 直接调用。

### 潜在问题
- 如果没有 `pair_id`，会为每行生成唯一 ID，语义上的“成组采样”就会退化。
- 因为按组取整，最终样本数可能略大于目标值。
- 分层采样是近似实现，不保证严格按全局标签比例。

## 3. `convert_nli_to_ft.py`

### 代码逻辑
- 读取一个或多个 NLI JSONL 文件。
- 自动识别二分类 RTE，或通过 `--binary` 强制指定。
- 将 `premise`、`hypothesis`、`label` 转换成 Alpaca 格式的 `instruction`、`input`、`output`。
- 可选打乱、抽样、切分训练集与验证集。
- 把生成的数据集注册到 `data/dataset_info.json`。

### 输入输出
- 输入：NLI JSONL、数据集名、输出目录、是否切分、验证集比例、instruction 模板。
- 输出：`full.json` 或 `full_train.json`、`full_val.json`，以及 `dataset_info.json` 的新条目。

### 调用关系
- 被 `run_batch_experiments.py` 和 `run_experiment.py` 调用。

### 潜在问题
- 生成的 `.json` 文件内容实际是 JSONL，而不是 JSON 数组。
- `--shuffle` 当前默认总是启用，命令行很难关闭。
- 默认写入固定工作目录下的 `data/dataset_info.json`，不利于多仓库复用。

## 4. `run_experiment.py`

### 代码逻辑
- 作为较早的单实验编排入口，串联转换、训练和测试。
- 会写入单个实验对应的训练配置。
- 适合手工跑一次实验或做兼容性测试。

### 输入输出
- 输入：实验名、模型、数据集和训练参数。
- 输出：单次实验目录、微调配置和测试结果。

### 调用关系
- 通常会调用 `convert_nli_to_ft.py`、训练命令、`test_ft_model.py` 或 `test_mettrain_experiment.py`。

### 潜在问题
- 属于旧式入口，与当前批量三种子流程不完全一致。
- 如果和批量脚本同时使用，容易产生两套目录和两套结果口径。

## 5. `run_finetune.py`

### 代码逻辑
- 负责单次 LoRA 微调的手动包装。
- 根据模型、数据集和超参数生成训练配置。
- 适合做超参试跑。

### 输入输出
- 输入：模型名、数据集、学习率、epoch、rank、batch size 等。
- 输出：`ft_config.yaml`、实验元信息和训练过程。

### 调用关系
- 主要调用 LLaMA Factory 训练流程。

### 潜在问题
- 强依赖固定模型字典和固定工作目录。
- 更偏交互式调参，不适合作为正式批量实验主入口。

## 6. `summarize_seed_experiments.py`

### 代码逻辑
- 从每个实验、每个 seed 的测试结果中重算统计指标。
- 汇总各数据集、各测试类型的准确率。
- 可选生成 Gemma 与 Llama 的对比文档。

### 输入输出
- 输入：实验配置、输出根目录、seed 列表。
- 输出：`result.md` 或对比报告 `comparison_gemma3_4b_vs_llama32_3b.md`。

### 调用关系
- 直接读取 `tests/original/*.jsonl` 和 `tests/mr/*.jsonl`。
- 和 `run_batch_experiments.py` 产生的结果目录结构配套使用。

### 潜在问题
- 只认固定目录结构和 `correct` 字段。
- 如果测试结果缺文件或 JSON 格式异常，汇总会变得不完整。

## 7. `test_llm.py`

### 代码逻辑
- 遍历数据目录中的 JSON/JSONL 文件。
- 识别 NLI 三分类和 RTE 二分类任务。
- 构造 prompt，调用 OpenAI 兼容接口。
- 保存逐条预测、原始响应、元信息和任务类型。

### 输入输出
- 输入：数据目录、模型名、API URL/key 环境变量名、输出目录、运行 ID、请求延迟。
- 输出：`output/<run_id>/<llm>/<dataset>/<mr>.jsonl`。

### 调用关系
- 被 `run_all.sh`、`run_qwen35_flash.sh` 和 `run_lmstudio_qwen.sh` 等包装脚本调用。
- 与 `evaluate.py`、`mrv.py`、`mrv_excel.py` 配套使用。

### 潜在问题
- 数据集识别依赖路径约定，文件命名变化会影响输出归档。
- 任务类型判断主要靠路径里是否包含 `rte`，较脆弱。
- 对外部 API 的重试和超时处理虽有，但仍会受网络和服务稳定性影响。

## 8. `test_deepseek.py`

### 代码逻辑
- 作为较早版本的 OpenAI 兼容推理脚本。
- 功能与 `test_llm.py` 高度重叠。

### 输入输出
- 输入和输出形态与 `test_llm.py` 基本一致。

### 调用关系
- 现在更像历史兼容入口，而不是推荐主入口。

### 潜在问题
- 逻辑与 `test_llm.py` 重复，维护成本高。
- 如果两者行为不一致，容易造成结果口径分裂。

## 9. `run_all.sh`

### 代码逻辑
- 遍历指定任务目录下的所有 `*_MR` 数据集。
- 先调用 `test_llm.py` 逐文件推理。
- 再把单数据集结果合并后调用 `evaluate.py`。
- 最后再合并所有数据集并生成总报告。

### 输入输出
- 输入：模型名、延迟、输出目录、任务目录、API 环境变量名。
- 输出：`output/<llm>/<dataset>/`、`all_combined.jsonl`、`all_report.txt`、每个数据集的报告。

### 调用关系
- 调用 `test_llm.py` 和 `evaluate.py`。

### 潜在问题
- 依赖 `conda run -n LLMTrain3.9`，环境名固定。
- 用 `sort` 合并 JSONL 时依赖输出格式稳定。
- 对目录结构和文件命名依赖较强。

## 10. `evaluate.py`

### 代码逻辑
- 读取 `test_llm.py` 输出的预测文件。
- 归一化三分类或二分类预测标签。
- 计算 accuracy、macro precision/recall/F1、混淆矩阵。
- 支持按 `_source`、`mr_type` 等字段分组统计。
- 生成纯文本评估报告。

### 输入输出
- 输入：一个合并后的预测 JSONL 文件，以及可选的分组字段和报告路径。
- 输出：终端报告或指定的文本报告文件。

### 调用关系
- 被 `run_all.sh` 调用。
- 也可独立对单个合并结果使用。

### 潜在问题
- 标签归一化逻辑较复杂，若新增标签格式需要同步修改。
- 二分类与三分类的判别依赖 `meta.task`，若上游缺字段会影响结果。
- 与 `mrv.py` 存在部分重复的归一化逻辑。

## 11. `mrv.py`

### 代码逻辑
- 遍历模型输出目录。
- 按数据集和 MR 文件统计错误率，即 MRV。
- 生成终端表格，也可写入文本文件。

### 输入输出
- 输入：模型输出根目录、可选输出报告路径。
- 输出：MRV 统计表。

### 调用关系
- 依赖 `test_llm.py` 的输出目录约定。

### 潜在问题
- 与 `mrv_excel.py` 逻辑重复。
- 对输出文件结构和 `meta.task` 依赖较强。

## 12. `mrv_excel.py`

### 代码逻辑
- 基于同一套 MRV 统计结果生成 Excel 报告。
- 适合整理成表格后再做人工比较。

### 输入输出
- 输入：模型输出根目录。
- 输出：`.xlsx` 报告文件。

### 调用关系
- 与 `mrv.py` 共用同类输入结构。

### 潜在问题
- 依赖 `openpyxl` 等额外库。
- 新增 MR 时，静态列配置可能遗漏。

## 13. `run_lmstudio_qwen.sh`

### 代码逻辑
- 先检查 LM Studio 可达性。
- 再调用 `test_llm.py` 进行本地模型测试。

### 输入输出
- 输入：模型名、数据目录、延迟。
- 输出：`output/` 下的预测文件。

### 调用关系
- 调用 `test_llm.py`。

### 潜在问题
- 当前脚本里调用了 `python3 test.py`，但仓库根目录下并没有这个文件，按现状大概率会失败。
- 这个脚本与真实目录结构不完全对齐，需要修正后再用。

## 14. `run_qwen35_flash.sh`

### 代码逻辑
- 按数据集顺序批量运行 Qwen Flash 推理和评估。
- 目标是自动化地产出结果与报告。

### 输入输出
- 输入：模型名、API 环境变量、数据目录、输出目录。
- 输出：预测结果和评估报告。

### 调用关系
- 调用 `test_llm.py` 和 `evaluate.py`。

### 潜在问题
- 依赖固定的 conda 环境名和目录布局。
- 对本地环境和 API 兼容性有较强假设。

## 15. `prepare_gemma.sh`

### 代码逻辑
- 做 Gemma 模型准备和下载前置处理。
- 可能包含登录、缓存或配置改写。

### 输入输出
- 输入：Hugging Face token 或相关环境准备信息。
- 输出：Gemma 相关本地配置与模型缓存。

### 调用关系
- 作为微调前的准备辅助脚本。

### 潜在问题
- 这类脚本通常与具体环境绑定较强。
- 如果包含敏感信息打印，需要避免在日志中暴露 token。

## 16. `test_ft_model.py`

### 代码逻辑
- 早期 LoRA 测试脚本。
- 直接加载基座模型和 adapter，在原始数据上做有限测试。

### 输入输出
- 输入：base model、adapter 路径和原始数据集路径。
- 输出：终端测试结果或简化预测。

### 调用关系
- 被旧入口 `run_experiment.py` 使用。

### 潜在问题
- 路径和数据集定义偏旧。
- 不适合替代当前系统实验测试脚本。

## 17. `test_lora_finetune.py`

### 代码逻辑
- 做 LoRA 训练的烟雾测试。
- 临时构造小数据集、临时配置并跑短训练。
- 训练结束后清理临时文件。

### 输入输出
- 输入：模型、少量样本、训练参数。
- 输出：短训过程和环境可用性验证结果。

### 调用关系
- 用于验证训练环境和 LLaMA Factory 是否能跑通。

### 潜在问题
- 可能会覆盖 `data/dataset_info.json`。
- 数据集和模型硬编码较多，通用性弱。

## 18. `test_original_data.py`

### 代码逻辑
- 面向本地模型服务，对原始数据做全量测试。
- 支持 `_progress.json` 形式的断点续跑。

### 输入输出
- 输入：本地模型列表、原始数据目录、LM Studio 或类似服务。
- 输出：`output/original_data/<model>.jsonl`。

### 调用关系
- 和本地 API 测试场景配套。

### 潜在问题
- 很依赖本地服务、`lms` CLI 或类似环境。
- 运行时间较长，模型切换逻辑较脆弱。

## 19. `test_mettrain_experiment.py`

### 代码逻辑
- 加载微调后的 LoRA 模型。
- 分别测试 5 个 Original 数据集和 5 个 MR 数据集。
- 支持批量生成与标签归一化。
- 输出每条样本的 `gold`、`pred` 和 `correct`。

### 输入输出
- 输入：实验名或 LoRA 路径、base model、batch size、数据集过滤项。
- 输出：`output/experiments/<exp>/tests/original/*.jsonl` 和 `tests/mr/*.jsonl`。

### 调用关系
- 被 `run_batch_experiments.py` 和 `run_experiment.py` 调用。

### 潜在问题
- MR 测试把所有 MR 类型聚合在一起，缺少单 MR 追溯信息。
- 仍然依赖固定工作目录和 GPU 环境。
- 如果后续要做更细粒度的 MR 分析，输出结构需要先扩展。

## 20. 脚本间总体关系

- `run_batch_experiments.py` 是当前主实验总入口。
- `sample_mettrain_pairid.py` 和 `convert_nli_to_ft.py` 是训练数据准备链路。
- `test_mettrain_experiment.py` 是当前系统实验的正式测试入口。
- `test_llm.py` 是通用推理主入口，`run_all.sh` 是它的批处理包装。
- `evaluate.py`、`mrv.py`、`mrv_excel.py` 是结果分析链路。
- `run_experiment.py`、`run_finetune.py`、`test_ft_model.py`、`test_lora_finetune.py`、`test_original_data.py` 更偏历史兼容或烟雾测试。
- `test_deepseek.py` 与 `test_llm.py` 功能重叠，后者更适合作为统一入口。

## 21. 主要风险点汇总

- 脚本统一从自身位置推导项目根目录。
- 多个脚本默认依赖特定 conda 环境名、特定数据目录和特定输出目录结构。
- 旧入口与新入口并存，结果口径可能不一致。
- `test_llm.py`、`evaluate.py`、`mrv.py` 之间的标签归一化逻辑有重复，修改时容易漏同步。
- `run_lmstudio_qwen.sh` 当前存在明显路径错误，需要修正后再使用。
- `test_mettrain_experiment.py` 的结果结构缺少单 MR 粒度信息，限制后续分析。

## 22. 建议的使用优先级

1. 系统化 LoRA 实验优先使用 `run_batch_experiments.py`。
2. 通用 API 推理优先使用 `test_llm.py`，批量运行时用 `run_all.sh`。
3. 结果评估优先使用 `evaluate.py`，MRV 统计用 `mrv.py` 或 `mrv_excel.py`。
4. 历史脚本保留参考即可，不建议作为新实验主入口。
