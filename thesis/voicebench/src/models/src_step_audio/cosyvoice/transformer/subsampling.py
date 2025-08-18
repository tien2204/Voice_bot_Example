














"""Subsampling layer definition."""

from typing import Tuple, Union

import torch


class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(
        self, offset: Union[int, torch.Tensor], size: int
    ) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class EmbedinigNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling"""

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: linear input tensor (
                where time' = time .
            torch.Tensor: linear input mask (
                where time' = time .

        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: linear input tensor (
                where time' = time .
            torch.Tensor: linear input mask (
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.pos_enc = pos_enc_class
        
        
        self.subsampling_rate = 2
        
        self.right_context = 4

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: Subsampled tensor (
                where time' = time // 2.
            torch.Tensor: Subsampled mask (
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        time = x.size(1)
        x = x.transpose(1, 2)  
        x = self.conv(x)
        x = x.transpose(1, 2)  
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2 :: 2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        )
        self.pos_enc = pos_enc_class
        
        
        self.subsampling_rate = 4
        
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: Subsampled tensor (
                where time' = time // 4.
            torch.Tensor: Subsampled mask (
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: Subsampled tensor (
                where time' = time // 6.
            torch.Tensor: Subsampled mask (
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: Subsampled tensor (
                where time' = time // 8.
            torch.Tensor: Subsampled mask (
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


class LegacyLinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (
            x_mask (torch.Tensor): Input mask (

        Returns:
            torch.Tensor: linear input tensor (
                where time' = time .
            torch.Tensor: linear input mask (
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask
