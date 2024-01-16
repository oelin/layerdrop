from typing import Any, Dict, Tuple

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


class LayerDrop(nn.Module):
    """LayerDrop.

    Implements LayerDrop (Fan et al., 2019), a structured dropout module for use
    in transformers that has a regularization effect during training. Models
    trained with LayerDrop learn to become sub-network-invariant, i.e. during
    inference, we can omit arbitrary layers without a substantial decrease in
    performance.

    There are several pruning strategies that can be employed. Following the
    authors, we use "Every Other". This strategy is simple and more complex
    options offer only marginal gains. 

    This module takes a collection of "stackable" layers such as transformer 
    layers. The "Every Other" scheme will be applied during training at a given
    drop rate. When layers are dropped, they're effectively replaced by the
    identity function.

    Example
    -------
    >>> module = LayerDrop(
    ...     rate=0.1,
    ...     layers=(
    ...         TransformerLayer(...),
    ...         TransformerLayer(...),
    ...         ...,
    ...     ),
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, rate: float, layers: Tuple[nn.Module]) -> None:
        """Initialize the module.

        Parameters
        ----------
        rate : float
            The drop rate.
        layers : Tuple[nn.Module]
            The collection of "stackable" layers.
        """

        super().__init__()

        self.modulus = int(1 / rate)
        self.layers = layers

        assert len(layers) >= self.modulus, (
            'Rate must be large enough so that '
            'at least one layer is dropped.'
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward pass."""

        for i, layer in enumerate(self.layers):
            if self.training and not (i % self.modulus):
                if random.random() < 0.5:
                    continue
            
            # Otherwise, don't drop the layer.

            x = layer(x, *args, **kwargs)
            
        return x
