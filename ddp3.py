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
        self._dist_handles = []
        self._broadcast_and_add_hooks()

    def _broadcast_and_add_hooks(self) -> None:
        """
        Broadcasts the model parameters from rank 0 to all other ranks.
        """
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
            param.register_post_accumulate_grad_hook(self._reduce_gradient_hook)
    
    def _reduce_gradient_hook(self, param: torch.Tensor) -> None:
        """
        This hook is called when the gradient is computed.
        It reduces the gradient across all ranks.
        """
        if param.grad is None:
            return None
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._dist_handles.append(handle)
        return None
    
    def wait_for_reductions(self) -> None:
        """
        Waits for all asynchronous reductions to complete.
        """
        for handle in self._dist_handles:
            handle.wait()
        self._dist_handles.clear()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(self.world_size)
    
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

    mlp.wait_for_reductions()

    optimiser.step()
    optimiser.zero_grad(set_to_none=True)

    ones = torch.ones((batch_size, model_dim), dtype=torch.float32)
    output = mlp(ones)
    print(f"Rank {rank} output: {output[0,0].item()}\n")

    dist.destroy_process_group()
