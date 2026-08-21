import torch
import torch.nn as nn

class TanhAttention(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.W_Q_h1 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_K_h1 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_V_h1 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_Q_h2 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_K_h2 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_V_h2 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_Q_h3 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_K_h3 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_V_h3 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_Q_h4 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_K_h4 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.W_V_h4 = nn.Parameter(torch.randn(d_model, d_model//4).uniform_(-1/d_model, 1/d_model))
        self.RMSNormMHA = nn.RMSNorm(d_model)
        self.Tanh = nn.Tanh()
    def forward(self, x):
        seq = x.size(1)
        #head1
        x_q = x@self.W_Q_h1
        x_k = x@self.W_K_h1
        x_v = x@self.W_V_h1
        mask = torch.tril(torch.ones(seq,seq).to(x.device), diagonal=0)
        matrix = x_q@x_k.transpose(1,2)
        matrix_new = []
        for i in range(seq):
            vec = matrix[:, i, :]
            std = vec*vec
            std = torch.mean(std, dim=1)
            std = torch.sqrt(std).unsqueeze(1)
            vec = vec/(std+0.001)
            matrix_new.append(vec)
        matrix = torch.stack(matrix_new, dim=1)
        matrix = matrix*mask.unsqueeze(0)
        matrix = self.Tanh(matrix)
        matrix_h1 = matrix@x_v

        #head2
        x_q = x@self.W_Q_h2
        x_k = x@self.W_K_h2
        x_v = x@self.W_V_h2
        mask = torch.tril(torch.ones(seq,seq).to(x.device), diagonal=0)
        matrix = x_q@x_k.transpose(1,2)
        matrix_new = []
        for i in range(seq):
            vec = matrix[:, i, :]
            std = vec*vec
            std = torch.mean(std, dim=1)
            std = torch.sqrt(std).unsqueeze(1)
            vec = vec/std
            matrix_new.append(vec)
        matrix = torch.stack(matrix_new, dim=1)
        matrix = matrix *mask.unsqueeze(0)
        matrix = self.Tanh(matrix)
        matrix_h2 = matrix@x_v

        #head3
        x_q = x@self.W_Q_h3
        x_k = x@self.W_K_h3
        x_v = x@self.W_V_h3
        mask = torch.tril(torch.ones(seq,seq).to(x.device), diagonal=0)
        matrix = x_q@x_k.transpose(1,2)
        matrix_new = []
        for i in range(seq):
            vec = matrix[:, i, :]
            std = vec*vec
            std = torch.mean(std, dim=1)
            std = torch.sqrt(std).unsqueeze(1)
            vec = vec/std
            matrix_new.append(vec)
        matrix = torch.stack(matrix_new, dim=1)
        matrix = matrix *mask.unsqueeze(0)
        matrix = self.Tanh(matrix)
        matrix_h3 = matrix@x_v

        #head4
        x_q = x@self.W_Q_h4
        x_k = x@self.W_K_h4
        x_v = x@self.W_V_h4
        mask = torch.tril(torch.ones(seq,seq).to(x.device), diagonal=0)
        matrix = x_q@x_k.transpose(1,2)
        matrix_new = []
        for i in range(seq):
            vec = matrix[:, i, :]
            std = vec*vec
            std = torch.mean(std, dim=1)
            std = torch.sqrt(std).unsqueeze(1)
            vec = vec/std
            matrix_new.append(vec)
        matrix = torch.stack(matrix_new, dim=1)
        matrix = matrix *mask.unsqueeze(0)
        matrix = self.Tanh(matrix)
        matrix_h4 = matrix@x_v


        global_matrix = [matrix_h1,matrix_h2,matrix_h3,matrix_h4]

        matrix = torch.cat(global_matrix,dim=2)
        matrix = self.RMSNormMHA(matrix)
        return matrix
