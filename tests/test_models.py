import unittest
import torch
from src.models import GRUModel

class TestModels(unittest.TestCase):

    def setUp(self):
        """设置模型测试所需的通用参数"""
        self.vocab_size = 100
        self.embed_dim = 32
        self.hidden_dim = 64
        self.output_dim = 2
        self.n_layers = 1
        self.bidirectional = False
        self.dropout = 0.0
        self.padding_idx = 0
        self.batch_size = 4
        self.seq_len = 10

    def test_gru_model_forward_pass(self):
        """测试基础GRU模型的前向传播"""
        # 实例化模型
        model = GRUModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            padding_idx=self.padding_idx
        )
        
        # 创建模拟输入数据
        text = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        text_lengths = torch.tensor([self.seq_len] * self.batch_size)
        
        # 执行前向传播
        output = model(text, text_lengths)
        
        # 断言输出形状是否正确
        expected_shape = torch.Size([self.batch_size, self.output_dim])
        self.assertEqual(output.shape, expected_shape)

    def test_bidirectional_gru_model_forward_pass(self):
        """测试双向GRU模型的前向传播"""
        # 实例化双向模型
        model = GRUModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            bidirectional=True, # 设为双向
            dropout=self.dropout,
            padding_idx=self.padding_idx
        )
        
        # 创建模拟输入数据
        text = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        text_lengths = torch.tensor([self.seq_len] * self.batch_size)
        
        # 执行前向传播
        output = model(text, text_lengths)
        
        # 断言输出形状是否正确
        expected_shape = torch.Size([self.batch_size, self.output_dim])
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main() 