from abc import ABC, abstractmethod
from collections import deque
import torch
import torch.nn as nn
import numpy as np

class BalancingStrategy(ABC):

    @abstractmethod
    def update(self, loss_components: dict[str, torch.Tensor], model: nn.Module, step: int) -> dict[str, float]:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict):
        pass

class FixedWeights(BalancingStrategy):

    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def update(self, loss_components, model, step):
        return self.weights.copy()

    def state_dict(self):
        return {'weights': self.weights}

    def load_state_dict(self, state):
        self.weights = state['weights']

class SoftAdapt(BalancingStrategy):

    def __init__(self, loss_names: list[str], tau: float=0.1, window_size: int=5, initial_weights: dict[str, float] | None=None):
        self.loss_names = loss_names
        self.tau = tau
        self.window_size = window_size
        self.initial_weights = (initial_weights or {name: 1.0 for name in loss_names}).copy()
        self.history: dict[str, deque] = {name: deque(maxlen=window_size + 1) for name in loss_names}
        self.current_weights = self.initial_weights.copy()
        self._warmup_steps = window_size + 1

    def update(self, loss_components, model, step):
        for name in self.loss_names:
            if name in loss_components:
                val = loss_components[name]
                self.history[name].append(val.item() if torch.is_tensor(val) else val)
        if step < self._warmup_steps:
            return self.current_weights.copy()
        deltas = {}
        for name in self.loss_names:
            h = list(self.history[name])
            if len(h) >= 2:
                deltas[name] = np.log(max(h[-1], 1e-12)) - np.log(max(h[0], 1e-12))
            else:
                deltas[name] = 0.0
        n = len(self.loss_names)
        delta_values = np.array([deltas[name] for name in self.loss_names])
        delta_values = delta_values / (self.tau + 1e-08)
        delta_values = delta_values - np.max(delta_values)
        exp_values = np.exp(delta_values)
        softmax = exp_values / (np.sum(exp_values) + 1e-08)
        adaptive = np.clip(n * softmax, 0.1, 10.0)
        raw_weights = np.array([self.initial_weights[name] * adaptive[i] for i, name in enumerate(self.loss_names)])
        target_sum = sum(self.initial_weights.values())
        raw_weights *= target_sum / (raw_weights.sum() + 1e-12)
        self.current_weights = {name: float(raw_weights[i]) for i, name in enumerate(self.loss_names)}
        return self.current_weights.copy()

    def state_dict(self):
        return {'history': {k: list(v) for k, v in self.history.items()}, 'current_weights': self.current_weights, 'initial_weights': self.initial_weights}

    def load_state_dict(self, state):
        for k, v in state['history'].items():
            self.history[k] = deque(v, maxlen=self.window_size + 1)
        self.current_weights = state['current_weights']
        self.initial_weights = state.get('initial_weights', self.initial_weights)

class GradNorm(BalancingStrategy):

    def __init__(self, loss_names: list[str], alpha: float=1.5, lr: float=0.01, initial_weights: dict[str, float] | None=None):
        self.loss_names = loss_names
        self.alpha = alpha
        self.lr = lr
        init = initial_weights or {name: 1.0 for name in loss_names}
        self.log_weights = {name: torch.tensor(np.log(init[name]), dtype=torch.float32, requires_grad=True) for name in loss_names}
        self.initial_losses: dict[str, float] | None = None

    def update(self, loss_components, model, step):
        if self.initial_losses is None:
            self.initial_losses = {name: loss_components[name].item() if torch.is_tensor(loss_components[name]) else loss_components[name] for name in self.loss_names if name in loss_components}
        shared_params = list(model.parameters())
        if not shared_params:
            return {name: torch.exp(self.log_weights[name]).item() for name in self.loss_names}
        shared_param = shared_params[0]
        grad_norms = {}
        weights = {name: torch.exp(self.log_weights[name]) for name in self.loss_names}
        for name in self.loss_names:
            if name not in loss_components:
                continue
            weighted_loss = weights[name] * loss_components[name]
            grad = torch.autograd.grad(weighted_loss, shared_param, retain_graph=True, create_graph=True)[0]
            grad_norms[name] = torch.norm(grad)
        if not grad_norms:
            return {name: weights[name].item() for name in self.loss_names}
        avg_norm = torch.mean(torch.stack(list(grad_norms.values())))
        loss_ratios = {}
        for name in self.loss_names:
            if name in loss_components and name in self.initial_losses:
                current = loss_components[name].item() if torch.is_tensor(loss_components[name]) else loss_components[name]
                initial = self.initial_losses[name]
                if initial > 0:
                    loss_ratios[name] = current / initial
                else:
                    loss_ratios[name] = 1.0
        avg_ratio = np.mean(list(loss_ratios.values())) if loss_ratios else 1.0
        grad_norm_loss = torch.tensor(0.0)
        for name in self.loss_names:
            if name in grad_norms and name in loss_ratios:
                relative_inverse = (loss_ratios[name] / (avg_ratio + 1e-08)) ** self.alpha
                target_norm = avg_norm * relative_inverse
                grad_norm_loss = grad_norm_loss + torch.abs(grad_norms[name] - target_norm)
        if grad_norm_loss.requires_grad:
            grad_norm_loss.backward(retain_graph=True)
            with torch.no_grad():
                for name in self.loss_names:
                    if self.log_weights[name].grad is not None:
                        self.log_weights[name] -= self.lr * self.log_weights[name].grad
                        self.log_weights[name].grad.zero_()
        n = len(self.loss_names)
        raw_weights = {name: torch.exp(self.log_weights[name]).item() for name in self.loss_names}
        total = sum(raw_weights.values())
        return {name: n * v / total for name, v in raw_weights.items()}

    def state_dict(self):
        return {'log_weights': {k: v.item() for k, v in self.log_weights.items()}, 'initial_losses': self.initial_losses}

    def load_state_dict(self, state):
        for k, v in state['log_weights'].items():
            self.log_weights[k] = torch.tensor(v, dtype=torch.float32, requires_grad=True)
        self.initial_losses = state['initial_losses']

def create_balancer(strategy: str, loss_names: list[str], initial_weights: dict[str, float], **kwargs) -> BalancingStrategy:
    if strategy == 'fixed':
        return FixedWeights(initial_weights)
    elif strategy == 'softadapt':
        return SoftAdapt(loss_names, tau=kwargs.get('tau', 0.1), window_size=kwargs.get('window_size', 5), initial_weights=initial_weights)
    elif strategy == 'gradnorm':
        return GradNorm(loss_names, alpha=kwargs.get('alpha', 1.5), lr=kwargs.get('lr', 0.01), initial_weights=initial_weights)
    else:
        raise ValueError(f"Unknown balancing strategy '{strategy}'")
