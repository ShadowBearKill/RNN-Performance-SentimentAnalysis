import torch
import torch.nn as nn
from .base_model import BaseModel

class GRUModel(BaseModel):
    """
    一个基础的、使用GRU进行情感分类的模型。
    """
    def __init__(self, 
                 vocab_size: int, 
                 embed_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 n_layers: int,
                 bidirectional: bool,
                 dropout: float,
                 padding_idx: int):
        
        super().__init__(vocab_size, embed_dim, hidden_dim, output_dim, padding_idx)
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0, # Dropout只在多层时有效
            batch_first=True
        )
        
        # 如果是双向的，全连接层的输入维度需要加倍
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        """
        前向传播实现。
        
        Args:
            text (Tensor): 输入的文本索引序列, shape: [batch_size, seq_len]
            text_lengths (Tensor): 每个序列的原始长度, shape: [batch_size]
        
        Returns:
            Tensor: 模型的输出 logits, shape: [batch_size, output_dim]
        """
        # 1. 嵌入层
        # embedded shape: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(text)
        
        # 2. 打包填充序列 (pack padded sequence)
        # 这可以告诉RNN忽略填充部分，提高计算效率
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths.to('cpu'), # pack_padded_sequence需要CPU上的长度张量
            batch_first=True
        )
        
        # 3. GRU层
        # packed_output: 包含所有时间步的输出
        # hidden: 最后一个时间步的隐藏状态
        # hidden shape: [n_layers * num_directions, batch_size, hidden_dim]
        _, hidden = self.gru(packed_embedded)
        
        # 4. 处理隐藏状态
        # 如果是双向的，hidden会包含前向和后向的隐藏状态，我们需要连接它们
        if self.gru.bidirectional:
            # 连接最后一个前向层和最后一个后向层的隐藏状态
            # hidden[-2,:,:] 是最后一个前向层的
            # hidden[-1,:,:] 是最后一个后向层的
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # 如果是单向的，直接使用最后一层的隐藏状态
            hidden = hidden[-1,:,:]
            
        # hidden shape: [batch_size, hidden_dim * num_directions]
        
        # 5. 全连接层
        # output shape: [batch_size, output_dim]
        output = self.fc(hidden)
        
        return output 