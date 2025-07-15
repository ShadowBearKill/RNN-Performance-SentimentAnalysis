import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any

from src.models import GRUModel # 之后可以扩展到导入所有模型
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
        """
        初始化训练器。

        Args:
            config (Dict[str, Any]): 包含所有实验参数的配置字典。
            vocab (Vocabulary): 词典对象。
            train_loader (DataLoader): 训练数据加载器。
            valid_loader (DataLoader): 验证数据加载器。
            test_loader (DataLoader): 测试数据加载器。
        """
        self.config = config
        self.vocab = vocab
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 动态实例化模型
        self._build_model()
        
        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _build_model(self):
        """根据配置动态构建模型"""
        model_config = self.config['model']
        
        # 使用一个映射来动态选择模型类
        # 这样以后添加新模型时，只需在这里加一行
        model_map = {
            "GRUModel": GRUModel,
            # "LSTMModel": LSTMModel, # 以后可以添加
        }
        
        model_name = model_config['name']
        if model_name not in model_map:
            raise ValueError(f"错误: 未知的模型名称 '{model_name}' 在配置文件中。")
        
        ModelClass = model_map[model_name]

        # 准备模型初始化参数
        model_params = {
            'vocab_size': len(self.vocab),
            'embed_dim': model_config['embed_dim'],
            'hidden_dim': model_config['hidden_dim'],
            'output_dim': model_config['output_dim'],
            'n_layers': model_config['n_layers'],
            'bidirectional': model_config['bidirectional'],
            'dropout': model_config['dropout'],
            'padding_idx': self.vocab.word2idx['<PAD>']
        }

        self.model = ModelClass(**model_params).to(self.device)
        print(f"模型 '{model_name}' 已成功构建并移动到 {self.device}。")

    def train(self):
        """
        完整的训练、验证和测试流程。
        (将在任务4.3中具体实现)
        """
        print("\n开始训练...")
        # 具体的训练循环逻辑将在这里实现
        pass

if __name__ == '__main__':
    # 这个部分用于基本测试，验证Trainer是否能被正确初始化
    # 需要模拟一些输入对象
    from src.utils import load_config
    
    print("--- 开始Trainer框架搭建测试 ---")
    
    # 1. 加载配置
    config_path = 'configs/gru.yaml'
    config = load_config(config_path)
    print(f"成功加载配置文件: {config_path}")

    # 2. 模拟依赖对象
    class MockVocab:
        def __init__(self):
            self.word2idx = {'<PAD>': 0, 'hello': 1, 'world': 2}
        def __len__(self):
            return len(self.word2idx)

    mock_vocab = MockVocab()
    mock_dataloader = DataLoader([([1,2], 0)], batch_size=1) # 模拟一个简单的数据加载器
    print("成功创建模拟的词典和数据加载器。")

    # 3. 初始化Trainer
    try:
        trainer = Trainer(
            config=config,
            vocab=mock_vocab,
            train_loader=mock_dataloader,
            valid_loader=mock_dataloader,
            test_loader=mock_dataloader
        )
        print("\nTrainer 初始化成功！")
        print("模型已构建，优化器和损失函数已定义。")
        print("\n模型结构:")
        print(trainer.model)
    except Exception as e:
        print(f"\nTrainer 初始化失败: {e}")

    print("\n--- Trainer框架搭建测试结束 ---") 