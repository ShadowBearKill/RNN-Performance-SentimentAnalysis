import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_output, text_lengths):
        # rnn_output shape: [batch_size, seq_len, hidden_dim]
        
        # 通过线性层计算能量(分数)
        # energy shape: [batch_size, seq_len, 1]
        energy = torch.tanh(self.attention(rnn_output))
        
        # 创建mask以忽略填充部分的影响
        mask = torch.arange(rnn_output.size(1), device=rnn_output.device)[None, :] < text_lengths[:, None]
        energy[~mask] = -float('inf')
        
        # 计算注意力权重
        # attention_weights shape: [batch_size, seq_len, 1]
        attention_weights = F.softmax(energy, dim=1)
        
        # 加权求和
        # context_vector shape: [batch_size, hidden_dim]
        context_vector = torch.sum(attention_weights * rnn_output, dim=1)
        
        return context_vector 