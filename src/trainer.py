import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import os
import numpy as np
from tqdm import tqdm
from src.models import RNNModel, LSTMModel, GRUModel, AttentionGRUModel
from src.data_loader import Vocabulary

class Trainer:
    """
    实验训练器类，负责管理模型的训练、验证和评估流程。
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 vocab: Vocabulary,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader):
        self.config = config
        self.vocab = vocab
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self._build_model()
        
        # --- 处理类别不平衡 ---
        class_weights = self._calculate_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
        
        # --- 早停和模型保存相关 ---
        self.epochs = config['training']['epochs']
        self.patience = config['training']['patience']
        self.best_valid_loss = float('inf')
        self.model_save_path = os.path.join('results/saved_models', f"{self.config['model']['name']}_best.pt")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def _calculate_class_weights(self) -> torch.Tensor:
        """根据训练集计算类别权重以处理数据不平衡问题"""
        # 获取所有训练样本的标签
        # 注意: .dataset 属性可以直接访问DataLoader内部的Dataset对象
        labels = [label for _, label in self.train_loader.dataset]
        label_counts = np.bincount(labels)  # 统计每个标签的样本数量
        
        # 使用sklearn中 'balanced' 模式的计算公式
        total_samples = sum(label_counts)
        num_classes = len(label_counts)
        weights = total_samples / (num_classes * label_counts)
        
        print(f"检测到类别数量: {num_classes}, 样本分布: {label_counts}")
        print(f"计算出的类别权重: {weights}")
        
        return torch.tensor(weights, dtype=torch.float).to(self.device)

    def _build_model(self):
        """根据配置动态构建模型"""
        model_config = self.config['model']
        
        # 使用一个映射来动态选择模型类
        # 这样以后添加新模型时，只需在这里加一行
        model_map = {
            "RNNModel": RNNModel,
            "LSTMModel": LSTMModel,
            "GRUModel": GRUModel,
            "AttentionGRUModel": AttentionGRUModel
        }
        
        model_name = model_config['name']
        if model_name not in model_map:
            raise ValueError(f"错误: 未知的模型名称 '{model_name}' 在配置文件中。")
        
        ModelClass = model_map[model_name]
        model_params = {
            'vocab_size': len(self.vocab), 'embed_dim': model_config['embed_dim'],
            'hidden_dim': model_config['hidden_dim'], 'output_dim': model_config['output_dim'],
            'n_layers': model_config['n_layers'], 'bidirectional': model_config['bidirectional'],
            'dropout': model_config['dropout'], 'padding_idx': self.vocab.word2idx['<PAD>']
        }
        self.model = ModelClass(**model_params).to(self.device)
        print(f"模型 '{model_name}' 已成功构建并移动到 {self.device}。")

    def _calculate_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """计算准确率"""
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y_true.view_as(top_pred)).sum()
        acc = correct.float() / y_true.shape[0]
        return acc.item()

    def _train_epoch(self) -> Tuple[float, float]:
        """执行一个训练轮次"""
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for batch in tqdm(self.train_loader, desc="训练中...", leave=False):
            text, labels, lengths = [t.to(self.device) for t in batch]
            
            self.optimizer.zero_grad()
            predictions = self.model(text, lengths)
            loss = self.criterion(predictions, labels)
            acc = self._calculate_accuracy(predictions, labels)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """在一个数据加载器上评估模型性能"""
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="评估中...", leave=False):
                text, labels, lengths = [t.to(self.device) for t in batch]
                predictions = self.model(text, lengths)
                loss = self.criterion(predictions, labels)
                acc = self._calculate_accuracy(predictions, labels)
                
                epoch_loss += loss.item()
                epoch_acc += acc
                
        return epoch_loss / len(loader), epoch_acc / len(loader)

    def train(self):
        """完整的训练、验证流程"""
        print("\n--- 开始训练 ---")
        patience_counter = 0
        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_epoch()
            valid_loss, valid_acc = self._evaluate(self.valid_loader)
            
            print(f'Epoch: {epoch+1:02} | 训练损失: {train_loss:.3f} | 训练准确率: {train_acc*100:.2f}%')
            print(f'\t          | 验证损失: {valid_loss:.3f} | 验证准确率: {valid_acc*100:.2f}%')
            
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                patience_counter = 0
                print(f'\t验证损失降低 ({self.best_valid_loss:.3f} --> {valid_loss:.3f})。保存模型至 {self.model_save_path}')
            else:
                patience_counter += 1
                print(f'\t验证损失未降低。早停计数: {patience_counter}/{self.patience}')
            
            if patience_counter >= self.patience:
                print(f'早停触发！连续 {self.patience} 个epoch验证损失未改善。')
                break
        
        # 训练结束后，加载性能最佳的模型
        self.model.load_state_dict(torch.load(self.model_save_path))
        print(f"--- 训练结束 ---")
        print(f"已加载性能最佳的模型 (验证损失: {self.best_valid_loss:.3f})")

        #TODO: 在测试集上评估模型性能

if __name__ == '__main__':
    from src.utils import load_config
    from src.data_loader import SentimentDataset, collate_fn
    
    print("--- 开始Trainer功能测试 ---")
    config_path = 'configs/gru.yaml'
    config = load_config(config_path)
    config['training']['epochs'] = 2 # 测试时减少轮次
    
    # --- 模拟依赖对象 (使用真实类) ---
    # 创建一个平衡的模拟数据集来测试流程
    mock_vocab = Vocabulary()
    mock_vocab.build_vocab([['hello', 'world', 'test', 'data']])
    
    mock_vectorized_data = [
        ([1, 2], 1), ([1, 2, 3], 0), ([3, 4], 1), ([1], 0)
    ]
    mock_dataset = SentimentDataset(mock_vectorized_data)
    
    mock_loader = DataLoader(mock_dataset, batch_size=2, collate_fn=collate_fn)
    
    print("成功创建模拟的词典和数据加载器。")
    try:
        trainer = Trainer(
            config=config, vocab=mock_vocab,
            train_loader=mock_loader, valid_loader=mock_loader, test_loader=mock_loader
        )
        trainer.train()
        print("\nTrainer 完整流程测试成功！")
    except Exception as e:
        print(f"\nTrainer 测试失败: {e}")
        raise
    print("\n--- Trainer功能测试结束 ---") 