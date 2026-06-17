import torch
import torch.nn as nn


class CRNNAzimuthClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        num_azimuth: int = 36,
        freq_bins: int = 513,
        cnn_channels: tuple[int, int, int] = (16, 32, 64),
        gru_hidden: int = 64,
        gru_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.freq_after = freq_bins // 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.pool_freq = nn.AdaptiveAvgPool2d((1, None))
        self.gru = nn.GRU(
            input_size=cnn_channels[2],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        rnn_out_dim = gru_hidden * (2 if bidirectional else 1)
        self.head_az = nn.Linear(rnn_out_dim, num_azimuth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool_freq(x).squeeze(2)
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        if self.gru.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.head_az(h)
