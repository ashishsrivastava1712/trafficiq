import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        z = x.mean(dim=1)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return x * z.unsqueeze(1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads
        dh = D // H
        q = self.q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, dh).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class MSTN(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 24, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        # CNN Branch
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)

        # BiLSTM Branch
        self.proj   = nn.Linear(input_dim, 128)
        self.bilstm = nn.LSTM(
            input_size=128, hidden_size=64,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1
        )

        # Fusion
        self.fusion_dim = 192
        self.gate = nn.Linear(192, 192)

        # SE + MHA
        self.se      = SEBlock(192, reduction=8)
        self.mha     = MultiHeadAttention(192, heads=4, dropout=0.1)
        self.norm    = nn.LayerNorm(192)
        self.dropout = nn.Dropout(dropout)

        # Regression Head
        self.head = nn.Linear(192, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # CNN Branch
        h = F.relu(self.bn1(self.conv1(x.transpose(1, 2))))
        h = F.relu(self.bn2(self.conv2(h)))
        z_cnn = h.mean(dim=2)

        # BiLSTM Branch
        xp = self.proj(x)
        lstm_out, _ = self.bilstm(xp)
        z_bilstm = lstm_out.mean(dim=1)

        # SGF Fusion
        z = torch.cat([z_cnn, z_bilstm], dim=1)
        z = z * torch.sigmoid(self.gate(z))

        # ETA — collapse to L=1
        z = z.unsqueeze(1)

        # SE Block
        z = self.se(z)

        # MHA
        z = self.mha(z)
        z = z.squeeze(1)

        # Norm + Dropout
        z_final = self.dropout(self.norm(z))

        return self.head(z_final).squeeze(-1)