import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class AttnLSTM(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_layers=1, label_size=2):
        super(AttnLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim, num_layers=num_layers,
                            hidden_size=hidden_dim, bidirectional=True)
        self.attnLinear = nn.Linear(hidden_dim*2, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_att=False):
        # x : n, B, d
        lstm_out, _ = self.lstm(x)   # n, B, 2h
        attn = torch.tanh(self.attnLinear(lstm_out))  # n, B, h
        score = F.softmax(self.context(attn), dim=0)  # n, B, 1
        lstm_out = lstm_out.permute(1, 2, 0)  # B, 2h, n
        score = score.permute(1, 0, 2)  # B, n, 1
        a = torch.bmm(lstm_out, score).squeeze(dim=2)  # B, 2h

        return a

class PairAttnLSTM(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_layers=1, label_size=2):
        super(PairAttnLSTM, self).__init__()
        self.LSTM_c = AttnLSTM(embedding_dim, hidden_dim, num_layers, label_size)
        self.LSTM_r = AttnLSTM(embedding_dim, hidden_dim, num_layers, label_size)
        self.hidden2label = nn.Linear(hidden_dim*4, label_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x_c, x_r, y_true, return_att=False):
        v_c = self.LSTM_c(x_c)
        v_r = self.LSTM_r(x_r)
        v = torch.cat((v_c, v_r), dim=1) # B, 4h
        logits = self.hidden2label(v)
        loss = self.loss_function(logits, y_true)

        return loss, logits
