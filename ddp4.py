import warnings
from mlp import SimpleMLP
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional, Dict

warnings.filterwarnings(action="ignore", category=UserWarning)

def setup():
    dist.init_process_group("gloo")

class GradientBucket:
    """
    GradientBucket manages a bucket of gradients for efficient
    all-reduce communication in distributed training.
    It packs multiple gradients into a contiguous buffer, performs all-reduce,
    and unpacks them back to their source tensors.
    """
    def __init__(self, params_in_bucket: List[nn.Parameter], device: torch.device):
        """
        Initializes a GradientBucket.

        Args:
            params_in_bucket: A list of nn.Parameter objects that belong to this bucket.
            device: The device where the buffer should be allocated.
        """
        if not params_in_bucket:
            raise ValueError("params_in_bucket must not be empty. At least one parameter is required.")

        self.params = params_in_bucket
        self.device = device
        
        if not all(p.device == self.device for p in self.params):
            raise ValueError("All parameters in a bucket must be on the same device.")
        
        if not all(p.dtype == self.params[0].dtype for p in self.params):
            raise ValueError("All parameters in a bucket must have the same dtype.")

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.dtype = self.params[0].dtype

        self.param_shapes = [p.shape for p in self.params]
        self.param_numels = [p.numel() for p in self.params]
        
        self.total_numel = sum(self.param_numels)
        if self.total_numel == 0:
            raise ValueError("Total number of elements in parameters must be greater than zero.")

        self.buffer = torch.zeros(self.total_numel, dtype=self.dtype, device=self.device)

        self.param_slices = {}
        current_offset = 0
        for i, p in enumerate(self.params):
            numel = self.param_numels[i]
            self.param_slices[p] = slice(current_offset, current_offset + numel)
            current_offset += numel

        self.ready_param_count = 0
        self._comm_handle: Optional[dist.Work] = None

    def add_gradient(self, param: nn.Parameter) -> None:
        """
        Copies the gradient of the given param into the bucket's buffer.
        """
        assert param.grad is not None, "add_gradient should only be called on params with a gradient."
        
        grad_data = param.grad.data
        param_slice = self.param_slices[param]
        self.buffer[param_slice].copy_(grad_data.view(-1))
        self.ready_param_count += 1
        
        if self.is_ready():
            self._start_reduction()

    def is_ready(self) -> bool:
        """Checks if all gradients for this bucket have been added."""
        return self.ready_param_count == len(self.params)

    def _start_reduction(self) -> None:
        """
        Initiates an asynchronous all-reduce operation on the bucket's buffer.
        This should only be called when is_ready() is True.
        """
        if self._comm_handle is not None and not self._comm_handle.is_completed():
            raise RuntimeError("Previous reduction for this bucket has not completed or was not reset.")

        self._comm_handle = dist.all_reduce(self.buffer, op=dist.ReduceOp.SUM, async_op=True)

    def finalize_reduction_and_copy_back(self) -> None:
        """
        Waits for the all-reduce operation to complete, averages the gradients
        in the buffer, copies them back, and resets the bucket for the next iteration.
        """
        if self._comm_handle is not None:
            self._comm_handle.wait()
        
        self.buffer.div_(self.world_size)
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            param_slice = self.param_slices[param]
            shape = self.param_shapes[i]
            
            param.grad.data.copy_(self.buffer[param_slice].view(shape))
        
        self.reset()
        
    def reset(self) -> None:
        """Resets the bucket's state for the next training iteration."""
        self.ready_param_count = 0
        self._comm_handle = None

class DDP(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.world_size = dist.get_world_size()
        self._buckets: List[GradientBucket] = []
        self._param_to_bucket: Dict[nn.Parameter, GradientBucket] = {}
        self._broadcast_and_add_hooks()

    def _broadcast_and_add_hooks(self) -> None:
        """
        Broadcasts the model parameters from rank 0 to all other ranks.
        """
        # We assume the model has a .blocks attribute that is an iterable of modules.
        for block in self.model.blocks: # type: ignore
            params_to_bucket = [p for p in block.parameters() if p.requires_grad]
            if not params_to_bucket:
                continue

            device = params_to_bucket[0].device
            bucket = GradientBucket(params_to_bucket, device)
            self._buckets.append(bucket)
            for param in params_to_bucket:
                self._param_to_bucket[param] = bucket
                dist.broadcast(param.data, src=0)
                param.register_post_accumulate_grad_hook(self._reduce_gradient_hook)
    
    def _reduce_gradient_hook(self, param: nn.Parameter) -> None:
        """
        This hook is called when the gradient is computed.
        It reduces the gradient across all ranks.
        """
        if param.grad is None:
            return

        bucket = self._param_to_bucket.get(param)
        if bucket is not None:
            bucket.add_gradient(param)
    
    def wait_for_reductions(self) -> None:
        """
        Waits for all asynchronous reductions to complete.
        """
        for bucket in reversed(self._buckets):
            bucket.finalize_reduction_and_copy_back()
    
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
