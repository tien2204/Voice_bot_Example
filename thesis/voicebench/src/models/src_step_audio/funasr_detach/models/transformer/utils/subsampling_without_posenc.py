


"""Subsampling layer definition."""

import math
import torch


class Conv2dSubsamplingWOPosEnc(torch.nn.Module):
    """Convolutional 2D subsampling.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        kernels (list): kernel sizes
        strides (list): stride sizes

    """

    def __init__(self, idim, odim, dropout_rate, kernels, strides):
        """Construct an Conv2dSubsamplingWOPosEnc object."""
        assert len(kernels) == len(strides)
        super().__init__()
        conv = []
        olen = idim
        for i, (k, s) in enumerate(zip(kernels, strides)):
            conv += [
                torch.nn.Conv2d(1 if i == 0 else odim, odim, k, s),
                torch.nn.ReLU(),
            ]
            olen = math.floor((olen - k) / s + 1)
        self.conv = torch.nn.Sequential(*conv)
        self.out = torch.nn.Linear(odim * olen, odim)
        self.strides = strides
        self.kernels = kernels

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: Subsampled tensor (
                where time' = time // 4.
            torch.Tensor: Subsampled mask (
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        for k, s in zip(self.kernels, self.strides):
            x_mask = x_mask[:, :, : -k + 1 : s]
        return x, x_mask
