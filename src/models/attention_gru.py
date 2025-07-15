import torch
import torch.nn as nn
from .base_model import BaseModel
from .attention import Attention

class AttentionGRUModel(BaseModel):
    """
    一个使用GRU + Self-Attention进行情感分类的模型。
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
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        num_directions = 2 if bidirectional else 1
        self.attention = Attention(hidden_dim * num_directions)
        
        # 确保全连接层的输入维度正确
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths.to('cpu'),
            batch_first=True,
            enforce_sorted=False # DataLoader中已排序，但为安全起见
        )
        
        packed_output, hidden = self.gru(packed_embedded)
        
        # 解包以获取所有时间步的输出
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output shape: [batch_size, seq_len, hidden_dim * num_directions]
        
        # 应用Attention
        context_vector = self.attention(output, text_lengths)
        # context_vector shape: [batch_size, hidden_dim * num_directions]
            
        # 通过全连接层进行分类
        final_output = self.fc(context_vector)
        
        return final_output 