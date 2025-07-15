import torch
import torch.nn as nn
from .base_model import BaseModel

class RNNModel(BaseModel):
    """
    一个基础的、使用普通RNN进行情感分类的模型。
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
        
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths.to('cpu'),
            batch_first=True
        )
        
        _, hidden = self.rnn(packed_embedded)
        
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        output = self.fc(hidden)
        
        return output 