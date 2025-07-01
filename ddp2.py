import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from mlp import SimpleMLP

warnings.filterwarnings(action="ignore", category=UserWarning)

def setup():
    dist.init_process_group("gloo")

    
class DDP(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.world_size = dist.get_world_size()
        self._broadcast_and_add_hooks()

    def _broadcast_and_add_hooks(self):
        """
        Broadcasts the model parameters from rank 0 to all other ranks.
        """
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
            param.register_hook(self._reduce_gradient_hook)
    
    def _reduce_gradient_hook(self, grad: torch.Tensor) -> torch.Tensor | None:
        """
        This hook is called when the gradient is computed.
        It reduces the gradient across all ranks.
        """
        if grad is None:
            return None
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad /= dist.get_world_size() # type: ignore
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == '__main__':
    setup()
    rank = dist.get_rank()

    model_dim = 64
    mlp = SimpleMLP(model_dim=model_dim, hidden_dim=4*model_dim)

    mlp = DDP(mlp)  # Wrap the model in DDP

    optimiser = torch.optim.SGD(mlp.parameters(), lr=0.01)

    batch_size = 32
    activation = torch.rand((batch_size, model_dim), dtype=torch.float32)
    output = mlp(activation)
    loss = torch.sum(output)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad(set_to_none=True)

    ones = torch.ones((batch_size, model_dim), dtype=torch.float32)
    output = mlp(ones)
    print(f"Rank {rank} output: {output[0,0].item()}\n")

    dist.destroy_process_group()
