import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    所有模型变体的基类。
    它包含一个嵌入层和一个最终的分类器层，
    所有子类模型都将共享这些组件。
    """
    def __init__(self, 
                 vocab_size: int, 
                 embed_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 padding_idx: int):
        """
        Args:
            vocab_size (int): 词典的大小。
            embed_dim (int): 词嵌入的维度。
            hidden_dim (int): RNN隐藏层的维度。
            output_dim (int): 输出层的维度 (例如，2分类问题就是2)。
            padding_idx (int): 填充token的索引，用于在嵌入层忽略它。
        """
        super().__init__()
        
        # 嵌入层，将词索引转换为密集向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # 分类器层，将RNN的输出映射到最终的类别得分
        # 注意：子类可能需要根据其RNN的输出维度来调整这里的输入维度
        # 例如，双向RNN的输出维度是 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        """
        定义了前向传播的通用骨架。
        
        注意：这是一个抽象方法，子类必须实现自己的版本，
        因为核心的RNN/LSTM/GRU逻辑在这里并未定义。
        """
        # text = [batch_size, seq_len]
        # text_lengths = [batch_size]
        
        # embedded = [batch_size, seq_len, embed_dim]
        embedded = self.embedding(text)
        
        # ** 子类需要在这里插入自己的RNN层逻辑 **
        # 例如: rnn_output = self.rnn(embedded, ...)
        
        # 这只是一个占位符，强制子类去实现它
        raise NotImplementedError("子类必须实现 forward 方法！") 