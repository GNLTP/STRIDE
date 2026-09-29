import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointWrapper(nn.Module):

    def __init__(self, module: nn.Module,
                 use_checkpoint: bool = True,
                 use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

    def _flatten_obj(self, obj, flat_tensors):
        if torch.is_tensor(obj):
            idx = len(flat_tensors)
            flat_tensors.append(obj)
            return ("tensor", idx)

        elif isinstance(obj, (list, tuple)):
            children = [self._flatten_obj(o, flat_tensors) for o in obj]
            
            return ("list", type(obj), children)

        else:
            
            return ("static", obj)

    def _flatten_args(self, args):
        flat_tensors = []
        spec = [self._flatten_obj(a, flat_tensors) for a in args]
        return flat_tensors, spec

    def _rebuild_obj(self, spec, flat_tensors):
        kind = spec[0]

        if kind == "tensor":
            _, idx = spec
            return flat_tensors[idx]

        elif kind == "list":
            _, container_type, children = spec
            built = [self._rebuild_obj(ch, flat_tensors) for ch in children]
            
            return container_type(built) if container_type is tuple else built

        elif kind == "static":
            _, obj = spec
            return obj

        else:
            raise ValueError(f"Unknown spec kind: {kind}")

    def _rebuild_args(self, flat_tensors, spec):
        return tuple(self._rebuild_obj(s, flat_tensors) for s in spec)

    def _run_with_spec(self, flat_tensors, spec):
        args = self._rebuild_args(list(flat_tensors), spec)
        return self.module(*args)

    def forward(self, *args):

        if (not torch.is_grad_enabled()) or (not self.use_checkpoint):
            return self.module(*args)

        flat_tensors, spec = self._flatten_args(args)

        if len(flat_tensors) == 0:
            return self.module(*args)

        def fw(*flat_inputs):
            return self._run_with_spec(flat_inputs, spec)

        if self.use_reentrant:
            out = checkpoint(fw, *flat_tensors)
        else:
            out = checkpoint(fw, *flat_tensors, use_reentrant=False)

        return out
