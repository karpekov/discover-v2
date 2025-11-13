"""
Projection heads for CLIP alignment.

Provides both linear and MLP projection options.
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearProjection(nn.Module):
    """
    Simple linear projection head.

    Maps from d_model → projection_dim with a single linear layer.
    """

    def __init__(self, d_model: int, projection_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, projection_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, d_model]
        Returns:
            projected: [batch_size, projection_dim]
        """
        return self.proj(x)


class MLPProjection(nn.Module):
    """
    MLP projection head.

    Similar to SimCLR/CLIP projections:
    - 2-layer: d_model → hidden_dim → projection_dim
    - 3-layer: d_model → hidden_dim → hidden_dim → projection_dim

    Uses GELU activation and optional dropout.
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_bn: bool = False
    ):
        super().__init__()

        assert num_layers in [2, 3], "MLP projection supports 2 or 3 layers"

        layers = []

        # First layer: d_model → hidden_dim
        layers.append(nn.Linear(d_model, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Optional middle layer: hidden_dim → hidden_dim
        if num_layers == 3:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # Final layer: hidden_dim → projection_dim
        layers.append(nn.Linear(hidden_dim, projection_dim, bias=False))

        self.mlp = nn.Sequential(*layers)

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, d_model]
        Returns:
            projected: [batch_size, projection_dim]
        """
        return self.mlp(x)


def create_projection_head(
    projection_type: str,
    d_model: int,
    projection_dim: int,
    hidden_dim: int = 2048,
    num_layers: int = 2,
    dropout: float = 0.1,
    use_bn: bool = False
) -> nn.Module:
    """
    Factory function to create projection heads.

    Args:
        projection_type: 'linear' or 'mlp'
        d_model: Input dimension
        projection_dim: Output dimension
        hidden_dim: Hidden dimension for MLP (ignored for linear)
        num_layers: Number of layers for MLP (2 or 3, ignored for linear)
        dropout: Dropout rate for MLP (ignored for linear)
        use_bn: Whether to use batch normalization in MLP (ignored for linear)

    Returns:
        Projection head module
    """
    if projection_type == 'linear':
        return LinearProjection(d_model, projection_dim)
    elif projection_type == 'mlp':
        return MLPProjection(
            d_model=d_model,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn
        )
    else:
        raise ValueError(f"Unknown projection type: {projection_type}. Use 'linear' or 'mlp'.")


if __name__ == '__main__':
    # Test projection heads
    import torch

    batch_size = 8
    d_model = 768
    projection_dim = 512

    # Test linear projection
    print("Testing Linear Projection:")
    linear_proj = LinearProjection(d_model, projection_dim)
    x = torch.randn(batch_size, d_model)
    out = linear_proj(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in linear_proj.parameters()):,}")

    # Test MLP projection (2-layer)
    print("\nTesting MLP Projection (2-layer):")
    mlp_proj_2 = MLPProjection(d_model, projection_dim, hidden_dim=2048, num_layers=2)
    out = mlp_proj_2(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mlp_proj_2.parameters()):,}")

    # Test MLP projection (3-layer)
    print("\nTesting MLP Projection (3-layer):")
    mlp_proj_3 = MLPProjection(d_model, projection_dim, hidden_dim=2048, num_layers=3)
    out = mlp_proj_3(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mlp_proj_3.parameters()):,}")

    # Test factory function
    print("\nTesting Factory Function:")
    proj_linear = create_projection_head('linear', d_model, projection_dim)
    proj_mlp = create_projection_head('mlp', d_model, projection_dim, hidden_dim=2048, num_layers=2)

    print(f"  Linear: {sum(p.numel() for p in proj_linear.parameters()):,} params")
    print(f"  MLP: {sum(p.numel() for p in proj_mlp.parameters()):,} params")

