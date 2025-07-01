import torch
import torch.nn as nn

class _MLPBlock(nn.Module):
    """
    A single MLP block consisting of two linear layers:
    an up-projection followed by a down-projection.
    """
    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(in_features=model_dim, out_features=hidden_dim, bias=False)
        self.w_out = nn.Linear(in_features=hidden_dim, out_features=model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, model_dim]
        Returns: [batch, model_dim]
        """
        return self.w_out(torch.relu(self.w_in(x)))

class SimpleMLP(nn.Module):
    """
    An MLP model composed of a sequence of _MLPBlock instances.
    By default, it consists of 4 blocks, resulting in an 8-layer network.
    """
    def __init__(self, model_dim: int, hidden_dim: int, num_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[_MLPBlock(model_dim, hidden_dim) for _ in range(num_blocks)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, model_dim]
        Returns: [batch, model_dim]
        """
        return self.blocks(x)