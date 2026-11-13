"""
Baseline classifiers on 20-event windows.

  - DeepCASAS: port of `~/code/har/TDOST/DeepCASAS/model.py` (Liciotti et al. 2020 bi-LSTM as used by TDOST):
    token = sensor + value, Embedding(64) -> 1-layer bi-LSTM(64) -> Linear on the last time step; same init.
  - TDOST: port of the 'naive' ConvLSTM classifier in `~/code/har/TDOST/HAR/model.py` (TDOST's best HAR setting
    for v1-1 + all-distilroberta-v1 in all three houses): 3 x [Conv1d k3 reflect, ReLU, Dropout 0.2]
    (768 -> 32 -> 64 -> 128) -> 2-layer GRU(256, dropout 0.2) -> Linear on the last time step.
  - TreeCNN: implemented from Cao et al. (2020) (no public code). Each event is a one-hot vector over
    sensor+value tokens. Layer i: a shared Conv1d (k3) over every node, adjacent nodes concatenated along
    channels, then a residual block (3 x Conv1d k3 + ReLU, identity shortcut); each layer shortens the sequence by
    one, so k-1 layers reduce k events to one node, which a Linear layer classifies. Conv output channels
    16, 32, 64, 64, ... as in the paper. Activity labels only (no resident head).
"""
import numpy as np
import torch
import torch.nn as nn


class DeepCASAS(nn.Module):
    def __init__(self, vocab_size, num_classes, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.encoder = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.softmax = nn.Linear(dim * 2, num_classes)
        nn.init.uniform_(self.embedding.weight, -np.sqrt(1 / dim), np.sqrt(1 / dim))
        for name, p in self.encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4:n // 2].fill_(1.)
        nn.init.xavier_uniform_(self.softmax.weight)
        nn.init.zeros_(self.softmax.bias)

    def forward(self, x):
        out, _ = self.encoder(self.embedding(x))
        return self.softmax(out[:, -1])


class _ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv1d(cin, cout, 3, padding=1, padding_mode='reflect', bias=False), nn.ReLU(), nn.Dropout(0.2))


class TDOSTConvLSTM(nn.Module):
    def __init__(self, num_classes, input_size=768):
        super().__init__()
        self.conv = nn.Sequential(_ConvBlock(input_size, 32), _ConvBlock(32, 64), _ConvBlock(64, 128))
        self.rnn = nn.GRU(128, 256, num_layers=2, batch_first=True, dropout=0.2)
        self.softmax = nn.Linear(256, num_classes)
        nn.init.xavier_uniform_(self.softmax.weight)
        nn.init.zeros_(self.softmax.bias)

    def forward(self, x):
        z = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.rnn(z)
        return self.softmax(out[:, -1])


class _TreeLayer(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.conv = nn.Conv1d(cin, c, 3, padding=1)
        self.res = nn.Sequential(*[m for _ in range(3) for m in (nn.Conv1d(2 * c, 2 * c, 3, padding=1), nn.ReLU())])

    def forward(self, h):
        B, L, C, D = h.shape
        f = self.conv(h.reshape(B * L, C, D)).reshape(B, L, -1, D)
        pair = torch.cat([f[:, :-1], f[:, 1:]], dim=2)
        B, L1, C2, D = pair.shape
        flat = pair.reshape(B * L1, C2, D)
        return (flat + self.res(flat)).reshape(B, L1, C2, D)


class TreeCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, length=20, channels=(16, 32, 64)):
        super().__init__()
        self.vocab_size = vocab_size
        layers, cin = [], 1
        for i in range(length - 1):
            c = channels[min(i, len(channels) - 1)]
            layers.append(_TreeLayer(cin, c))
            cin = 2 * c
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(cin * vocab_size, num_classes)

    def forward(self, x):
        # x: (B, L) token ids (0 = padding -> all-zero vector).
        h = nn.functional.one_hot(x, self.vocab_size).float()
        h[..., 0] = 0
        h = h.unsqueeze(2)  # (B, L, 1, V)
        for layer in self.layers:
            h = layer(h)
        return self.fc(h[:, 0].flatten(1))
