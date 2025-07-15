import pandas as pd
import jieba
import re
from typing import List, Tuple, Dict
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    """自定义情感分析数据集"""
    def __init__(self, vectorized_data: List[Tuple[List[int], int]]):
        self.vectorized_data = vectorized_data

    def __len__(self):
        return len(self.vectorized_data)

    def __getitem__(self, idx):
        sequence, label = self.vectorized_data[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    自定义的collate_fn，用于处理和填充一个批次的数据。
    """
    # 1. 按序列长度降序排序
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    sequences, labels = zip(*batch)
    
    # 2. 获取每个序列的原始长度
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # 3. 对序列进行填充
    # pad_sequence期望(L, *)，所以先permute，填充后再permute回来
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # 4. 堆叠标签
    labels = torch.stack(labels, 0)
    
    return padded_sequences, labels, lengths


class Vocabulary:
    """词典类，用于管理词语到索引的映射"""
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def build_vocab(self, all_tokens: List[List[str]]):
        """
        根据所有文本的分词结果构建词典。

        Args:
            all_tokens (List[List[str]]): 包含所有样本分词结果的列表。
        """
        word_counts = Counter(token for tokens in all_tokens for token in tokens)
        # 过滤低频词
        filtered_words = [word for word, count in word_counts.items() if count >= self.min_freq]
        
        for word in filtered_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)

def vectorize_data(processed_data: List[Tuple[List[str], int]], vocab: Vocabulary) -> List[Tuple[List[int], int]]:
    """
    将处理后的文本数据转换为索引序列。

    Args:
        processed_data (List[Tuple[List[str], int]]): 预处理后的数据。
        vocab (Vocabulary): 已构建的词典对象。

    Returns:
        List[Tuple[List[int], int]]: 向量化后的数据。
    """
    vectorized = []
    for tokens, label in processed_data:
        # 使用<UNK>的索引来处理未登录词
        indexed_tokens = [vocab.word2idx.get(token, vocab.word2idx['<UNK>']) for token in tokens]
        vectorized.append((indexed_tokens, label))
    return vectorized


def load_and_preprocess_data(file_path: str) -> List[Tuple[List[str], int]]:
    """
    加载并预处理 ChnSentiCorp_htl_all 数据集。

    Args:
        file_path (str): 数据集CSV文件的路径。

    Returns:
        List[Tuple[List[str], int]]: 一个元组列表，每个元组包含
                                     一个分词后的单词列表和对应的整数标签。
    """
    # 使用pandas读取CSV文件，假设文件包含'label'和'review'两列
    df = pd.read_csv(file_path)

    processed_data = []
    for _, row in df.iterrows():
        label = int(row['label'])
        review_text = str(row['review'])

        # 1. 文本清洗：去除所有非中文字符、字母和数字
        cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', review_text)
        
        # 2. 分词：使用jieba进行分词
        tokens = jieba.lcut(cleaned_text.strip())

        # 过滤掉空字符串
        tokens = [token for token in tokens if token.strip()]

        if tokens: # 确保文本不为空
            processed_data.append((tokens, label))
            
    return processed_data

if __name__ == '__main__':
    # 提供一个示例用法，当直接运行此脚本时
    # 注意：请将 'data/ChnSentiCorp_htl_all.csv' 替换为您的实际文件路径
    
    # 创建一个虚拟的数据文件用于演示
    dummy_data = {
        'label': [1, 0],
        'review': [
            '这家酒店环境很好，服务也很周到！性价比高！',
            '非常糟糕的体验...房间有异味，设施陈旧。不会再来 #差评# T_T'
        ]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_path = 'data/dummy_data.csv'
    dummy_df.to_csv(dummy_path, index=False)
    
    print(f"创建虚拟数据集于: {dummy_path}")
    
    # 加载并预处理数据
    try:
        # 1. 加载和预处理
        processed_data = load_and_preprocess_data(dummy_path)
        print("\n--- 步骤 1: 预处理完成 ---")
        print(f"样本数量: {len(processed_data)}")
        
        # 2. 构建词典
        vocab = Vocabulary(min_freq=1)
        all_tokens_for_vocab = [tokens for tokens, _ in processed_data]
        vocab.build_vocab(all_tokens_for_vocab)
        print("\n--- 步骤 2: 词典构建完成 ---")
        print(f"词典大小: {len(vocab)}")
        print(f"词典内容 (部分): {list(vocab.word2idx.items())[:10]}")

        # 3. 向量化
        vectorized_data = vectorize_data(processed_data, vocab)
        print("\n--- 步骤 3: 向量化完成 ---")
        print(f"向量化结果示例: {vectorized_data[0]}")

        # 4. 创建Dataset和DataLoader
        dataset = SentimentDataset(vectorized_data)
        # collate_fn是关键，它使得我们可以动态地处理不同长度的序列
        data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        print("\n--- 步骤 4: DataLoader创建完成 ---")
        print(f"批处理大小: 2")

        # 从DataLoader中获取一个批次的数据来演示
        batch_sequences, batch_labels, batch_lengths = next(iter(data_loader))
        
        print("\nDataLoader输出示例 (一个批次):")
        print(f"  序列 (Tensor shape): {batch_sequences.shape}")
        print(f"  标签 (Tensor shape): {batch_labels.shape}")
        print(f"  原始长度 (Tensor shape): {batch_lengths.shape}")
        print("\n  序列内容 (已填充):")
        print(batch_sequences)
        print("\n  标签内容:")
        print(batch_labels)
        print("\n  原始长度内容:")
        print(batch_lengths)
        print("-" * 20)

    except FileNotFoundError:
        print(f"错误：请确保数据集文件位于正确路径。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}") 