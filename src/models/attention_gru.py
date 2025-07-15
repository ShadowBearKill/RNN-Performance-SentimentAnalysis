import torch
import torch.nn as nn
from .base_model import BaseModel

class AttentionGRUModel(BaseModel):
    """
    一个使用GRU + Multi-Head Self-Attention进行情感分类的模型。
    """
    def __init__(self, 
                 vocab_size: int, 
                 embed_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 n_layers: int,
                 bidirectional: bool,
                 dropout: float,
                 padding_idx: int,
                 n_heads: int):
        
        # 调用父类初始化，但要确保fc层的输入维度正确
        num_directions = 2 if bidirectional else 1
        gru_output_dim = hidden_dim * num_directions
        super().__init__(vocab_size, embed_dim, gru_output_dim, output_dim, padding_idx)
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 使用PyTorch内置的多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_output_dim,
            num_heads=n_heads,
            batch_first=True
        )


    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths.to('cpu'),
            batch_first=True,
        )
        
        packed_output, _ = self.gru(packed_embedded)
        
        # 解包以获取所有时间步的输出
        # rnn_outputs shape: [batch_size, seq_len, hidden_dim * num_directions]
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 创建key_padding_mask
        # True值表示该位置是填充，应该被忽略
        max_len = rnn_outputs.size(1)
        key_padding_mask = torch.arange(max_len, device=text.device)[None, :] >= text_lengths[:, None]
        
        # 应用多头注意力
        # Query, Key, Value 都是 GRU 的输出序列
        attn_output, _ = self.attention(
            rnn_outputs, rnn_outputs, rnn_outputs, key_padding_mask=key_padding_mask
        )
        # attn_output shape: [batch_size, seq_len, gru_output_dim]

        # 将注意力输出序列池化为单个向量
        # 我们只对非填充部分的输出进行平均
        mask = ~key_padding_mask.unsqueeze(-1)
        masked_attn_output = attn_output * mask
        summed_output = torch.sum(masked_attn_output, dim=1)
        # summed_output shape: [batch_size, gru_output_dim]
        # 防止除以零
        non_pad_count = text_lengths.unsqueeze(-1).clamp(min=1)
        # non_pad_count shape: [batch_size, 1]
        pooled_output = summed_output / non_pad_count
        # pooled_output shape: [batch_size, gru_output_dim]
        
        # 通过全连接层进行分类
        final_output = self.fc(pooled_output)
        
        return final_output 