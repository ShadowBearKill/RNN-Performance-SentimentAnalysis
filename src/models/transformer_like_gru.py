import torch
import torch.nn as nn
from .base_model import BaseModel

class TransformerLikeGRU(BaseModel):
    """
    一个受Transformer启发的模型。
    它先对Embedding应用Multi-Head Self-Attention，然后再送入GRU。
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
        
        # 注意：BaseModel的hidden_dim参数现在对应GRU的输出，而不是fc层的输入
        super().__init__(vocab_size, embed_dim, hidden_dim, output_dim, padding_idx)
        
        # 确保embed_dim可以被n_heads整除
        assert embed_dim % n_heads == 0, "embed_dim 必须能被 n_heads 整除"

        # 1. Self-Attention层
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True
        )
        # 2. Add & Norm
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        # 3. GRU层
        self.gru = nn.GRU(
            input_size=embed_dim, # GRU的输入是Attention的输出，维度不变
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 4. 分类器
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, text, text_lengths):
        # text shape: [batch_size, seq_len]
        
        # 1. Embedding
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embed_dim]
        
        # 2. Multi-Head Self-Attention
        key_padding_mask = torch.arange(text.size(1), device=text.device)[None, :] >= text_lengths[:, None]
        
        attn_output, _ = self.attention(
            embedded, embedded, embedded, key_padding_mask=key_padding_mask
        )
        
        # 3. Add & Norm (残差连接和层归一化)
        norm_output = self.layer_norm1(embedded + attn_output)
        
        # 4. GRU
        packed_input = nn.utils.rnn.pack_padded_sequence(
            norm_output, 
            text_lengths.to('cpu'),
            batch_first=True,
            enforce_sorted=False
        )
        
        _, hidden = self.gru(packed_input)
        
        # 5. 分类
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        output = self.fc(hidden)
        
        return output 