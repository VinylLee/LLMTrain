#!/usr/bin/env python3
"""
SICK数据集划分脚本
根据SemEval_set字段将SICK数据集划分为训练集、验证集和测试集
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import argparse


class SICKDatasetSplitter:
    """SICK数据集划分器"""

    def __init__(self, data_file: str = "SICK_annotated.txt"):
        """
        初始化划分器
        
        Args:
            data_file: SICK数据集文件路径
        """
        self.data_file = Path(data_file)
        self.output_dir = self.data_file.parent

        # 标签映射：将SICK的标签转换为标准的NLI标签
        self.label_mapping = {
            "ENTAILMENT": 0,  # entailment
            "NEUTRAL": 1,  # neutral  
            "CONTRADICTION": 2  # contradiction
        }

        # 反向映射
        self.label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def load_data(self) -> pd.DataFrame:
        """加载SICK数据集"""
        print(f"正在加载数据集: {self.data_file}")

        # 读取TSV文件
        df = pd.read_csv(self.data_file, sep='\t')

        print(f"数据集总大小: {len(df)} 条样本")
        print(f"数据集列: {list(df.columns)}")

        # 显示SemEval_set的分布
        if 'SemEval_set' in df.columns:
            print("\nSemEval_set分布:")
            print(df['SemEval_set'].value_counts())

        # 显示标签分布
        if 'entailment_label' in df.columns:
            print("\n标签分布:")
            print(df['entailment_label'].value_counts())

        return df

    def convert_to_nli_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        将SICK数据集转换为标准NLI格式
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            List[Dict]: 标准NLI格式的数据列表
        """
        nli_data = []

        for _, row in df.iterrows():
            # 检查必要字段是否存在
            if pd.isna(row.get('sentence_A')) or pd.isna(
                    row.get('sentence_B')) or pd.isna(
                        row.get('entailment_label')):
                continue

            # 转换为标准格式
            sample = {
                "premise":
                str(row['sentence_A']).strip(),
                "hypothesis":
                str(row['sentence_B']).strip(),
                "label":
                self.label_mapping.get(row['entailment_label'],
                                       1),  # 默认为neutral
                "label_text":
                self.label_names[self.label_mapping.get(
                    row['entailment_label'], 1)],

                # 保留原始的额外信息
                "pair_id":
                int(row.get('pair_ID', 0)),
                "pair_type":
                str(row.get('pair_type', '')),
                "relatedness_score":
                float(row.get('relatedness_score', 0.0)),
                "semeval_set":
                str(row.get('SemEval_set', '')),

                # 原始句子（如果存在）
                "sentence_a_original":
                str(row.get('sentence_A_original', row['sentence_A'])),
                "sentence_b_original":
                str(row.get('sentence_B_original', row['sentence_B'])),
            }

            nli_data.append(sample)

        return nli_data

    def split_dataset(
            self,
            nli_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        根据SemEval_set字段划分数据集
        
        Args:
            nli_data: NLI格式的数据列表
            
        Returns:
            Dict: 包含train、validation、test三个子集的字典
        """
        splits = {
            "train": [],
            "validation": [],  # TRIAL作为验证集
            "test": []
        }

        # 映射SemEval_set到标准划分
        set_mapping = {
            "TRAIN": "train",
            "TRIAL": "validation",  # TRIAL通常用作验证集
            "TEST": "test"
        }

        for sample in nli_data:
            semeval_set = sample["semeval_set"]
            target_set = set_mapping.get(semeval_set, "train")  # 默认为训练集
            splits[target_set].append(sample)

        # 显示划分统计
        print("\n数据集划分统计:")
        for split_name, split_data in splits.items():
            print(f"{split_name}: {len(split_data)} 条样本")

            # 显示标签分布
            if split_data:
                label_counts = {}
                for sample in split_data:
                    label = sample["label_text"]
                    label_counts[label] = label_counts.get(label, 0) + 1
                print(f"  标签分布: {label_counts}")

        return splits

    def save_splits(self,
                    splits: Dict[str, List[Dict[str, Any]]],
                    format: str = "jsonl"):
        """
        保存划分后的数据集
        
        Args:
            splits: 划分后的数据集
            format: 保存格式，支持 "jsonl", "json", "csv"
        """
        print(f"\n正在保存数据集，格式: {format}")

        for split_name, split_data in splits.items():
            if not split_data:
                print(f"警告: {split_name} 集合为空，跳过保存")
                continue

            if format == "jsonl":
                output_file = self.output_dir / f"{split_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in split_data:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            elif format == "json":
                output_file = self.output_dir / f"{split_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=2)

            elif format == "csv":
                output_file = self.output_dir / f"{split_name}.csv"
                df = pd.DataFrame(split_data)
                df.to_csv(output_file, index=False, encoding='utf-8')

            print(
                f"已保存 {split_name} 集合到: {output_file} ({len(split_data)} 条样本)")

    def save_dataset_info(self, splits: Dict[str, List[Dict[str, Any]]]):
        """保存数据集信息"""
        info = {
            "dataset_name": "SICK",
            "description": "Sentences Involving Compositional Knowledge",
            "total_samples":
            sum(len(split_data) for split_data in splits.values()),
            "splits": {},
            "labels": {
                "0": "entailment",
                "1": "neutral",
                "2": "contradiction"
            },
            "label_mapping": self.label_mapping
        }

        for split_name, split_data in splits.items():
            if split_data:
                label_counts = {}
                for sample in split_data:
                    label = sample["label_text"]
                    label_counts[label] = label_counts.get(label, 0) + 1

                info["splits"][split_name] = {
                    "size": len(split_data),
                    "label_distribution": label_counts
                }

        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"已保存数据集信息到: {info_file}")

    def create_simplified_format(self, splits: Dict[str, List[Dict[str,
                                                                   Any]]]):
        """
        创建简化格式的数据集（只包含premise, hypothesis, label）
        """
        print("\n正在创建简化格式数据集...")

        for split_name, split_data in splits.items():
            if not split_data:
                continue

            simplified_data = []
            for sample in split_data:
                simplified_sample = {
                    "premise": sample["premise"],
                    "hypothesis": sample["hypothesis"],
                    "label": sample["label"]
                }
                simplified_data.append(simplified_sample)

            # 保存简化格式
            output_file = self.output_dir / f"{split_name}_simple.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in simplified_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"已保存简化格式 {split_name} 集合到: {output_file}")

    def run(self, output_format: str = "jsonl", create_simple: bool = True):
        """
        运行完整的数据集划分流程
        
        Args:
            output_format: 输出格式
            create_simple: 是否创建简化格式
        """
        # 1. 加载数据
        df = self.load_data()

        # 2. 转换为NLI格式
        nli_data = self.convert_to_nli_format(df)
        print(f"\n成功转换 {len(nli_data)} 条样本为NLI格式")

        # 3. 划分数据集
        splits = self.split_dataset(nli_data)

        # 4. 保存数据集
        self.save_splits(splits, output_format)

        # 5. 保存数据集信息
        self.save_dataset_info(splits)

        # 6. 创建简化格式（可选）
        if create_simple:
            self.create_simplified_format(splits)

        print("\n✅ 数据集划分完成！")
        return splits


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SICK数据集划分工具")
    parser.add_argument("--data-file",
                        "-f",
                        type=str,
                        default="SICK_annotated.txt",
                        help="SICK数据集文件路径")
    parser.add_argument("--format",
                        "-fmt",
                        type=str,
                        default="jsonl",
                        choices=["jsonl", "json", "csv"],
                        help="输出格式")
    parser.add_argument("--no-simple", action="store_true", help="不创建简化格式数据集")
    parser.add_argument("--show-examples",
                        "-e",
                        action="store_true",
                        help="显示每个划分的示例数据")

    args = parser.parse_args()

    # 检查数据文件是否存在
    if not Path(args.data_file).exists():
        print(f"错误: 数据文件 {args.data_file} 不存在")
        print("请确保SICK_annotated.txt文件在当前目录中")
        return

    # 创建划分器并运行
    splitter = SICKDatasetSplitter(args.data_file)
    splits = splitter.run(args.format, not args.no_simple)

    # 显示示例（可选）
    if args.show_examples:
        print("\n" + "=" * 80)
        print("数据示例:")
        for split_name, split_data in splits.items():
            if split_data:
                print(f"\n{split_name.upper()} 集合示例:")
                for i, sample in enumerate(split_data[:2]):  # 显示前2个样本
                    print(f"  示例 {i+1}:")
                    print(f"    Premise: {sample['premise']}")
                    print(f"    Hypothesis: {sample['hypothesis']}")
                    print(
                        f"    Label: {sample['label_text']} ({sample['label']})"
                    )
                    print(f"    Score: {sample['relatedness_score']}")


if __name__ == "__main__":
    main()
