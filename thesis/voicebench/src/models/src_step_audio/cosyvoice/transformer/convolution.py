














"""ConvolutionModule definition."""

from typing import Tuple

import torch
from torch import nn


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        
        
        
        
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (
            mask_pad (torch.Tensor): used for batch padding (
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (
        """
        
        x = x.transpose(1, 2)  

        
        if mask_pad.size(2) > 0:  
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  
                x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                assert cache.size(0) == x.size(0)  
                assert cache.size(1) == x.size(1)  
                x = torch.cat((cache, x), dim=2)
            assert x.size(2) > self.lorder
            new_cache = x[:, :, -self.lorder :]
        else:
            
            
            
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        
        x = self.pointwise_conv1(x)  
        x = nn.functional.glu(x, dim=1)  

        
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        
        if mask_pad.size(2) > 0:  
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache
