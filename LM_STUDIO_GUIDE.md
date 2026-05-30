# LM Studio 本地Qwen3.6测试指南

## 前置条件

1. **LM Studio已启动**，运行在 `http://localhost:1234`
2. **Qwen3.6模型已下载并加载** 到LM Studio中

## 步骤1：在LM Studio中加载模型

1. 打开LM Studio应用
2. 左侧搜索 "qwen" 或 "qwen3.6"
3. 下载所需的Qwen模型（例如：`Qwen/Qwen2.5-3B` 或 `Qwen/Qwen2.5-7B`）
4. 点击"Load Model"加载到内存
5. 等待模型完全加载（界面会显示"Ready"）

## 步骤2：获取模型ID

在LM Studio左下角"Local Server"可以看到已加载模型的信息。模型ID通常显示在服务器列表中。

常见的Qwen模型ID：
- `qwen2.5-3b`
- `qwen2.5-7b`  
- `Qwen2.5-3B`
- `Qwen2.5-7B`

## 步骤3：使用test_deepseek.py测试单个文件

```bash
# 基本命令：指定API URL、模型名称、数据文件
python scripts/test_deepseek.py \
  --api-url-env LMSTUDIO_BASE_URL \
  --api-key-env LMSTUDIO_API_KEY \
  --llm qwen2.5-3b \
  --data-dir data/rte_test_MR \
  --output-dir output \
  --delay 0.1
```

## 步骤4：快速测试连接

```bash
# 运行test.py验证LM Studio连接
python3 test.py
```

## 完整使用示例

### 方案A：命令行直接指定API地址（推荐）

```bash
# 测试RTE数据集
python scripts/test_deepseek.py \
  --llm qwen2.5-7b \
  --data-dir data/rte_test_MR \
  --output-dir output \
  --delay 0.5
```

然后使用 `--api-url-env LMSTUDIO_BASE_URL` 配合 `.env` 中的配置：

```bash
# 使用.env中的LMSTUDIO_BASE_URL
DEEPSEEK_OPENAI_BASE_URL=http://localhost:1234 \
  python scripts/test_deepseek.py \
  --llm qwen2.5-7b \
  --data-dir data/rte_test_MR \
  --output-dir output
```

### 方案B：编辑.env文件

修改 `.env` 文件中的 `DEEPSEEK_OPENAI_BASE_URL`：

```env
# 将DeepSeek的URL临时改为本地LM Studio
DEEPSEEK_OPENAI_BASE_URL=http://localhost:1234
DEEPSEEK_KEY=dummy  # LM Studio不需要真实key
```

然后直接运行：

```bash
python scripts/test_deepseek.py --llm qwen2.5-7b --data-dir data/rte_test_MR
```

### 方案C：使用脚本包装（推荐长期使用）

创建 `run_lmstudio_test.sh`：

```bash
#!/bin/bash
set -e

# 配置
LM_STUDIO_URL="http://localhost:1234"
MODEL_NAME="${1:-qwen2.5-3b}"  # 默认模型，可从命令行指定
DATA_DIR="${2:-data}"
OUTPUT_DIR="output"
DELAY="0.5"

echo "Starting LM Studio test with model: $MODEL_NAME"
echo "Data directory: $DATA_DIR"
echo ""

# 运行测试脚本
python scripts/test_deepseek.py \
  --llm "$MODEL_NAME" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --delay "$DELAY"

echo ""
echo "Results saved to: $OUTPUT_DIR"
```

使用方式：

```bash
bash run_lmstudio_test.sh qwen2.5-7b data/snli_test_MR
```

## 故障排查

### 错误1：`model_not_found`

**原因**：模型在LM Studio中还没加载，或模型名称不对

**解决**：
1. 在LM Studio中确认模型已加载（左下角显示"Ready"）
2. 获取正确的模型ID（通常在LM Studio界面显示）
3. 更新 `--llm` 参数

### 错误2：`Connection refused`

**原因**：LM Studio没有启动或不在1234端口

**解决**：
1. 确认LM Studio应用已启动
2. 检查Local Server是否启用（左下角）
3. 确认没有改变默认端口

### 错误3：`Timeout`

**原因**：模型推理太慢或过载

**解决**：
1. 减小 `--delay` 参数（增加请求间隔）
2. 使用更小的模型（如 `qwen2.5-3b` 代替 `qwen2.5-7b`）
3. 减少batch大小或并发请求

## 查看结果

评估测试结果：

```bash
# 查看单个数据集的结果
python scripts/evaluate.py output/<run_id>/qwen2.5-3b/rte/all_combined.jsonl

# 分组统计（按数据集、MR类型）
python scripts/evaluate.py \
  output/<run_id>/qwen2.5-3b/all_combined.jsonl \
  --group-by _source mr_type
```

## 性能参考

| 模型 | 推理时间/样本 | 内存占用 | 适用场景 |
|------|-------------|--------|--------|
| qwen2.5-3b | ~0.5s | ~7GB | 快速测试 |
| qwen2.5-7b | ~1-2s | ~15GB | 准确度优先 |
| qwen2.5-14b | ~3-5s | ~30GB | 复杂推理 |

## 常见命令

```bash
# 1. 快速测试连接
python3 test.py

# 2. 测试单个MR类型
python scripts/test_deepseek.py \
  --llm qwen2.5-3b \
  --data-dir data/rte_test_MR \
  --delay 0.3

# 3. 测试全部数据集
bash scripts/run_all.sh --llm qwen2.5-3b --delay 0.5

# 4. 评估结果
python scripts/evaluate.py output/20260529_*/qwen2.5-3b/all_combined.jsonl --group-by _source mr_type
```
