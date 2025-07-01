import warnings

import torch
import torch.distributed as dist
from mlp import SimpleMLP

warnings.filterwarnings(action="ignore", category=UserWarning)

def setup():
    dist.init_process_group("gloo")
    
if __name__ == '__main__':
    setup()
    rank = dist.get_rank()

    model_dim = 64
    mlp = SimpleMLP(model_dim=model_dim, hidden_dim=4*model_dim)

    # replicate the model parameters across all ranks
    for param in mlp.parameters():
        dist.broadcast(param.data, src=0)

    optimiser = torch.optim.SGD(mlp.parameters(), lr=0.01)

    batch_size = 32
    activation = torch.rand((batch_size, model_dim), dtype=torch.float32)
    output = mlp(activation)
    loss = torch.sum(output)
    loss.backward()
    for param in mlp.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.data /= dist.get_world_size() # type: ignore
    
    optimiser.step()
    optimiser.zero_grad(set_to_none=True)

    ones = torch.ones((batch_size, model_dim), dtype=torch.float32)
    output = mlp(ones)
    print(f"Rank {rank} output: {output[0,0].item()}\n")

    dist.destroy_process_group()
