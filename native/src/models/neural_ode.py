import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from .networks import SmallFNN

class NeuralODEFunc(nn.Module):

    def __init__(self, state_dim: int, hidden_layers: int=3, hidden_units: int=32, activation: str='tanh', t_span: list[float] | tuple[float, float]=(0.0, 1.0), state_scales: list[float] | None=None, output_scales: list[float] | None=None, normalize_inputs: bool=False, init_strategy: str='all_small'):
        super().__init__()
        self.state_dim = state_dim
        self.normalize_inputs = normalize_inputs
        resolved_state_scales = state_scales or [1.0] * state_dim
        resolved_output_scales = output_scales or [1.0] * state_dim
        if len(resolved_state_scales) != state_dim or len(resolved_output_scales) != state_dim:
            raise ValueError('NeuralODEFunc scales must match state_dim')
        t_start, t_end = (float(t_span[0]), float(t_span[1]))
        self.register_buffer('time_center', torch.tensor(0.5 * (t_start + t_end)))
        self.register_buffer('time_half_range', torch.tensor(max(0.5 * (t_end - t_start), 1e-08)))
        self.register_buffer('state_scales', torch.tensor(resolved_state_scales, dtype=torch.float32).clamp_min(1e-08))
        self.register_buffer('output_scales', torch.tensor(resolved_output_scales, dtype=torch.float32))
        self.net = SmallFNN(input_dim=state_dim + 1, output_dim=state_dim, hidden_layers=hidden_layers, hidden_units=hidden_units, activation=activation, init_strategy=init_strategy)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            t_for_net = (t - self.time_center) / self.time_half_range
            y_for_net = y / self.state_scales
        else:
            t_for_net = t
            y_for_net = y
        if y.dim() == 1:
            t_input = t_for_net.reshape(1)
            inputs = torch.cat([t_input, y_for_net])
        else:
            if t_for_net.numel() == y.shape[0]:
                t_input = t_for_net.reshape(-1, 1)
            else:
                t_input = t_for_net.reshape(1).expand(y.shape[0]).unsqueeze(1)
            inputs = torch.cat([t_input, y_for_net], dim=1)
        output = self.net(inputs)
        return output * self.output_scales if self.normalize_inputs else output

class NeuralODESolver(nn.Module):

    def __init__(self, state_dim: int, hidden_layers: int=3, hidden_units: int=32, activation: str='tanh', solver: str='dopri5', rtol: float=1e-07, atol: float=1e-09, step_size: float | None=None, use_adjoint: bool=True, t_span: list[float] | tuple[float, float]=(0.0, 1.0), normalize_inputs: bool=False, state_scales: list[float] | None=None, output_scales: list[float] | None=None):
        super().__init__()
        self.func = NeuralODEFunc(state_dim=state_dim, hidden_layers=hidden_layers, hidden_units=hidden_units, activation=activation, t_span=t_span, state_scales=state_scales, output_scales=output_scales, normalize_inputs=normalize_inputs, init_strategy='small_output' if normalize_inputs else 'all_small')
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = {'step_size': step_size} if step_size is not None else {}
        self.integrator = odeint_adjoint if use_adjoint else odeint

    def forward(self, y0: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        solution = self.integrator(self.func, y0, t_eval, method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        return solution

    def get_dynamics_func(self) -> NeuralODEFunc:
        return self.func
