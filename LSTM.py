import torch
import torch.nn as nn

class SpectogramLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, batch_first=True):
        super(SpectogramLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        self.input_size=input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.linear_projection = nn.Conv1d(in_channels=self.hidden_size, 
                out_channels=23, kernel_size=1)
        
    def forward(self, input_t):
        batch_size = input_t.size(0)
        window_size = input_t.size(1)
        feature_size = input_t.size(2)
        h0 = torch.zeros((batch_size, self.num_layers, self.hidden_size), dtype=torch.float32)
        c0 = torch.zeros((batch_size, self.num_layers, self.hidden_size), dtype=torch.float32)

        output, (hn, cn) = self.lstm(input_t)
        

        return self.linear_projection(output.permute(0,2,1)).permute(0,2,1)