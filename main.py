import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.utils import load_config
from src.data_loader import (
    load_and_preprocess_data, 
    Vocabulary, 
    vectorize_data,
    SentimentDataset,
    collate_fn
)
from src.trainer import Trainer

def run_experiment(config_path: str):
    """
    运行一个完整的实验，包括数据加载、训练和评估。
    """
    # 1. 加载配置
    config = load_config(config_path)
    print("--- 配置加载完成 ---")
    
    # 设置随机种子以保证可复现性
    random.seed(416)
    np.random.seed(416)
    torch.manual_seed(416)
    
    # 2. 加载和预处理数据
    full_data = load_and_preprocess_data(config['data']['path'])
    print("--- 原始数据加载和预处理完成 ---")
    
    # 3. 划分数据集 (76% 训练, 12% 验证, 12% 测试)
    labels = [item[1] for item in full_data]
    train_data, test_data = train_test_split(
        full_data, test_size=0.24, random_state=416, stratify=labels
    )
    # 再次划分测试集得到验证集和最终测试集
    test_labels = [item[1] for item in test_data]
    valid_data, test_data = train_test_split(
        test_data, test_size=0.50, random_state=416, stratify=test_labels
    )
    print(f"数据划分完成: 训练集({len(train_data)}), 验证集({len(valid_data)}), 测试集({len(test_data)})")

    # 4. 构建词典 (仅使用训练数据)
    train_tokens = [tokens for tokens, _ in train_data]
    vocab = Vocabulary(min_freq=config['data']['min_freq'])
    vocab.build_vocab(train_tokens)
    print(f"--- 词典构建完成 (仅使用训练集)，词典大小: {len(vocab)} ---")

    # 5. 向量化和创建DataLoaders
    train_vec = vectorize_data(train_data, vocab)
    valid_vec = vectorize_data(valid_data, vocab)
    test_vec = vectorize_data(test_data, vocab)
    
    train_dataset = SentimentDataset(train_vec)
    valid_dataset = SentimentDataset(valid_vec)
    test_dataset = SentimentDataset(test_vec)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print("--- DataLoaders 创建完成 ---")
    
    # 6. 初始化并运行训练器
    trainer = Trainer(
        config=config, vocab=vocab,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader
    )
    trainer.train()
    
    # 7. 在测试集上进行最终评估
    trainer.run_final_evaluation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行RNN模型对比实验")
    parser.add_argument('--config', type=str, required=True,
                        help='指定实验要使用的配置文件路径 (例如: configs/gru.yaml)')
    
    args = parser.parse_args()
    
    run_experiment(args.config) 