import torch.nn as nn
import torch


class DurationPredictor(nn.Module):
    def __init__(
        self,
        text_num_embeds,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()

        text_dim = in_channels
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )

        self.norm_1 = nn.GroupNorm(1, filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = nn.GroupNorm(1, filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = x + 1
        x = self.text_embed(x).transpose(1, 2)
        x_mask = x_mask.unsqueeze(2).transpose(1, 2)

        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

    def phoneme_forward(self, phoneme_indices, phoneme_mask, g=None):
        """Forward pass with phoneme indices instead of text tokens"""

        x = phoneme_indices
        x = self.text_embed(x).transpose(1, 2)

        x_mask = phoneme_mask.unsqueeze(2).transpose(1, 2)

        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)

        return x * x_mask
