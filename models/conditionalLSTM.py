import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class AttnLSTM(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_layers=1, label_size=2):
        super(AttnLSTM, self).__init__()
        self.num_directions = 2
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim, num_layers=num_layers,
                            hidden_size=hidden_dim, bidirectional=True)
        self.attnLinear = nn.Linear(hidden_dim*2, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1)

    def forward(self, x, c_0=None, attn=True, return_att=False):
        # x : n, B, d
        if c_0 != None:
            batch_size = x.size(1)
            h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, 
                            self.hidden_dim, dtype=x.dtype, device=x.device)
            lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))   # (n, B, 2h), ((2, B, h), (2, B, h))
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        if attn:
            attn = torch.tanh(self.attnLinear(lstm_out))  # n, B, h
            score = F.softmax(self.context(attn), dim=0)  # n, B, 1
            lstm_out = lstm_out.permute(1, 2, 0)  # B, 2h, n
            score = score.permute(1, 0, 2)  # B, n, 1
            a = torch.bmm(lstm_out, score).squeeze(dim=2)  # B, 2h final sentence representation
            
            if return_att:
                return a, score, (h_n, c_n)
            
            return a, (h_n, c_n)
        else:
            return lstm_out, (h_n, c_n)

class ConditionalEncoding(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256, num_layers=1, label_size=2):
        super(ConditionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.LSTM_c = AttnLSTM(embedding_dim, hidden_dim, num_layers, label_size)
        self.LSTM_r = AttnLSTM(embedding_dim, hidden_dim, num_layers, label_size)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.loss_function = nn.CrossEntropyLoss()
    
    def forward(self, x_c, x_r, y_true, return_att=False):
        v_c, (hn_c, cn_c) = self.LSTM_c(x_c, attn=False) 

        v_r, _ = self.LSTM_r(x_r, cn_c) # B, 2h
        # v = torch.cat((v_c, v_r), dim=1) # B, 4h
        logits = self.hidden2label(v_r)
        loss = self.loss_function(logits, y_true)

        return loss, logits

class ConditionalAttn(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_layers=1, label_size=2):
        super(ConditionalAttn, self).__init__()
        self.num_directions = 2
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.LSTM_c = AttnLSTM(embedding_dim, hidden_dim, num_layers, label_size)
        self.LSTM_r = AttnLSTM(embedding_dim, hidden_dim, num_layers, label_size)
        self.attnLinear_c = nn.Linear(hidden_dim*2, hidden_dim)
        self.attnLinear_r = nn.Linear(hidden_dim*2, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1)
        self.proj_c = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.proj_r = nn.Linear(hidden_dim*2, hidden_dim*2)

        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.loss_function = nn.CrossEntropyLoss()
    
    def forward(self, x_c, x_r, y_true=None, return_att=False):
        v_c, (h_cn, c_cn) = self.LSTM_c(x_c, attn=False) # (n, B, 2h)

        if return_att:
            v_r, attn_score, _ = self.LSTM_r(x_r, c_cn, return_att=True) # (B, 2h) (B, n, 1)
        else:
            v_r, _ = self.LSTM_r(x_r, c_cn)

        ### Attention ###
        L = v_c.size(0)
        attn_vr = v_r.unsqueeze(0).repeat(L, 1, 1) # n, B, 2h
        attn = torch.tanh(self.attnLinear_c(v_c) + self.attnLinear_r(attn_vr)) # n, B, 2h
        score = F.softmax(self.context(attn), dim=0) #n, B, 1
        new_vc = torch.matmul(v_c.permute(1,2,0), score.permute(1,0,2)) # (B, 2h, n) x (B, n, 1) -> (B, 2h, 1)
        new_vc = new_vc.squeeze(-1) # (B, 2h)
        v = torch.tanh(self.proj_c(new_vc) + self.proj_r(v_r)) # B, 2h
        logits = self.hidden2label(v)

        if y_true != None:
            loss = self.loss_function(logits, y_true)
            if return_att:
                return loss, logits, attn_score, score
            return loss, logits
        
        if return_att:
            return logits, attn_score, score 

        return logits
