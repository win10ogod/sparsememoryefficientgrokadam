import math
import torch
from torch.optim import Optimizer
from torch.cuda.amp import autocast
from typing import Iterable, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseMemoryEfficientGrokAdam(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2, block_size: int = 1024,
                 sparsity_ratio: float = 0.1, warmup_steps: int = 1000, warmup_factor: float = 0.1,
                 lr_decay_steps: int = 10000, lr_decay_factor: float = 0.1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not block_size > 0:
            raise ValueError(f"Invalid block_size value: {block_size}")
        if not 0.0 <= sparsity_ratio <= 1.0:
            raise ValueError(f"Invalid sparsity_ratio value: {sparsity_ratio}")
        if not warmup_steps > 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")
        if not 0.0 <= warmup_factor <= 1.0:
            raise ValueError(f"Invalid warmup_factor value: {warmup_factor}")
        if not lr_decay_steps > 0:
            raise ValueError(f"Invalid lr_decay_steps value: {lr_decay_steps}")
        if not 0.0 <= lr_decay_factor <= 1.0:
            raise ValueError(f"Invalid lr_decay_factor value: {lr_decay_factor}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size,
                        sparsity_ratio=sparsity_ratio, warmup_steps=warmup_steps, warmup_factor=warmup_factor,
                        lr_decay_steps=lr_decay_steps, lr_decay_factor=lr_decay_factor)
        super(SparseMemoryEfficientGrokAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure)

    def _step_impl(self, closure: Optional[Callable[[], float]]) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue

            grads = [p.grad for p in params_with_grad]

            # Select a subset of parameters to update based on sparsity_ratio
            num_params_to_update = max(1, int(len(params_with_grad) * group['sparsity_ratio']))
            update_indices = torch.randperm(len(params_with_grad))[:num_params_to_update]
            params_to_update = [params_with_grad[i] for i in update_indices]
            grads_to_update = [grads[i] for i in update_indices]

            self._update_group(group, params_to_update, grads_to_update)

        return loss

    @staticmethod
    def _update_group(group: dict, params: list[torch.Tensor], grads: list[torch.Tensor]) -> None:
        beta1, beta2 = group['betas']
        block_size = group['block_size']
        warmup_steps = group['warmup_steps']
        warmup_factor = group['warmup_factor']
        lr_decay_steps = group['lr_decay_steps']
        lr_decay_factor = group['lr_decay_factor']

        for p, grad in zip(params, grads):
            state = group.get('state', {}).get(p, {})
            if not state:
                state = {'step': 0, 'exp_avg': torch.zeros_like(p, dtype=torch.bfloat16),
                         'exp_avg_sq': torch.zeros_like(p, dtype=torch.bfloat16)}
                if 'state' not in group:
                    group['state'] = {}
                group['state'][p] = state

            exp_avg, exp_avg_sq = state['exp_avg'].to(p.device), state['exp_avg_sq'].to(p.device)

            state['step'] += 1

            # Apply warmup and learning rate decay
            if state['step'] <= warmup_steps:
                lr = group['lr'] * (warmup_factor + (1 - warmup_factor) * state['step'] / warmup_steps)
            elif state['step'] > lr_decay_steps:
                lr = group['lr'] * lr_decay_factor
            else:
                lr = group['lr']

            with autocast():
                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                p.mul_(1 - lr * group['weight_decay'])
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-step_size)

            # Move states back to CPU
            state['exp_avg'] = exp_avg.to('cpu', dtype=torch.bfloat16)
            state['exp_avg_sq'] = exp_avg_sq.to('cpu', dtype=torch.bfloat16)

    def state_dict(self):
        state_dict = super().state_dict()
        for group in state_dict['param_groups']:
            group['grokking_signal_fns'] = None  # Cannot serialize functions
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group['grokking_signal_fns'] = self.defaults['grokking_signal_fns']

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('grokking_signal_fns', [])
            group.setdefault('grokking_signal_decay_rate', 0.1)
            group.setdefault('gradient_clipping', 1.0)
