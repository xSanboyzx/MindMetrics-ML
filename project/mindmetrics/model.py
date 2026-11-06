"""Architecture used by the original training checkpoint."""

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.3):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.lin(x)))) + x


class QoLRegressor(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, n_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)
