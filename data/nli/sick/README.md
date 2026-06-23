# SICK数据集划分说明

这个脚本已经成功将SICK数据集按照`SemEval_set`字段划分为训练集、验证集和测试集。

## 数据集统计

- **总样本数**: 9,840条
- **训练集**: 4,439条样本 (45.1%)
- **验证集**: 495条样本 (5.0%) - 来自TRIAL集合
- **测试集**: 4,906条样本 (49.9%)

## 标签分布

### 训练集
- neutral: 2,524条 (56.9%)
- entailment: 1,274条 (28.7%)
- contradiction: 641条 (14.4%)

### 验证集  
- neutral: 281条 (56.8%)
- entailment: 143条 (28.9%)
- contradiction: 71条 (14.3%)

### 测试集
- neutral: 2,790条 (56.9%)
- entailment: 1,404条 (28.6%)
- contradiction: 712条 (14.5%)

## 生成的文件

### 完整格式文件
- `train.jsonl` - 训练集（包含所有原始信息）
- `validation.jsonl` - 验证集（包含所有原始信息）
- `test.jsonl` - 测试集（包含所有原始信息）

### 简化格式文件（推荐用于训练）
- `train_simple.jsonl` - 训练集（只包含premise, hypothesis, label）
- `validation_simple.jsonl` - 验证集（只包含premise, hypothesis, label）
- `test_simple.jsonl` - 测试集（只包含premise, hypothesis, label）

### 其他文件
- `dataset_info.json` - 数据集详细信息和统计
- `dataset.jsonl` - 完整数据集（所有样本）

## 数据格式

### 简化格式 (推荐)
```json
{
  "premise": "A group of kids is playing in a yard and an old man is standing in the background",
  "hypothesis": "A group of boys in a yard is playing and a man is standing in the background", 
  "label": 1
}
```

### 完整格式
```json
{
  "premise": "A group of kids is playing in a yard and an old man is standing in the background",
  "hypothesis": "A group of boys in a yard is playing and a man is standing in the background",
  "label": 1,
  "label_text": "neutral",
  "pair_id": 1,
  "pair_type": "S1S2_intra",
  "relatedness_score": 4.5,
  "semeval_set": "TRAIN",
  "sentence_a_original": "A group of children playing in a yard, a man in the background.",
  "sentence_b_original": "A group of children playing in a yard, a man in the background."
}
```

## 标签编码

- `0`: entailment (蕴含)
- `1`: neutral (中立)
- `2`: contradiction (矛盾)

## 使用示例

### Python中加载数据
```python
import json

# 加载简化格式的训练数据
train_data = []
with open('train_simple.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        train_data.append(json.loads(line))

print(f"训练集大小: {len(train_data)}")
print(f"示例: {train_data[0]}")
```

### 在NLI训练中使用
```python
from torch.utils.data import Dataset

class SICKDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'premise': item['premise'],
            'hypothesis': item['hypothesis'],
            'label': item['label']
        }

# 使用示例
train_dataset = SICKDataset('train_simple.jsonl')
val_dataset = SICKDataset('validation_simple.jsonl')
test_dataset = SICKDataset('test_simple.jsonl')
```

## 重新生成数据集

如果需要重新划分数据集，可以运行：

```bash
python split.py --show-examples
```

支持的命令行参数：
- `--data-file`: 指定SICK数据文件路径（默认: SICK_annotated.txt）
- `--format`: 输出格式，支持jsonl、json、csv（默认: jsonl）
- `--no-simple`: 不生成简化格式文件
- `--show-examples`: 显示每个数据集的示例

## 数据集特点

SICK数据集的特点：
1. **句子对类型多样**: 包含词汇替换、语法变换、语义操作等多种类型
2. **语义相关性评分**: 每个句子对都有1-5的语义相关性评分
3. **高质量标注**: 由专业标注人员标注，质量较高
4. **适合组合语义研究**: 专门设计用于研究组合语义理解

这个划分后的数据集可以直接用于训练您的NLI模型！
