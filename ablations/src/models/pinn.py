import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import FNN, OutputTransform

def _inverse_softplus(x: float) -> float:
    if x > 20.0:
        return x
    return float(torch.log(torch.exp(torch.tensor(x)) - 1.0))

class PINNModel(nn.Module):

    def __init__(self, state_dim: int, hidden_layers: int=4, hidden_units: int=50, activation: str='tanh', init_method: str='glorot_normal', param_names: list[str] | None=None, param_init_values: dict[str, float] | None=None, positive_output: bool=True, output_transform: str | None=None, initial_condition: torch.Tensor | None=None, t_span: tuple[float, float] | list[float]=(0.0, 1.0), hard_initial_condition: bool=True, state_scales: list[float] | None=None):
        super().__init__()
        self.state_dim = state_dim
        self.net = FNN(input_dim=1, output_dim=state_dim, hidden_layers=hidden_layers, hidden_units=hidden_units, activation=activation, init_method=init_method)
        resolved_transform = output_transform or ('abs' if positive_output else 'none')
        if resolved_transform not in {'abs', 'smooth_abs', 'none'}:
            raise ValueError("PINN output_transform must be 'abs', 'smooth_abs', or 'none'")
        self.output_transform_name = resolved_transform
        self.output_transform = OutputTransform('abs' if resolved_transform == 'smooth_abs' else resolved_transform)
        self.hard_initial_condition = hard_initial_condition
        y0 = initial_condition.detach().clone().float() if initial_condition is not None else torch.zeros(state_dim)
        self.register_buffer('initial_condition', y0)
        resolved_state_scales = state_scales or [1.0] * state_dim
        if len(resolved_state_scales) != state_dim:
            raise ValueError('PINN state_scales must match state_dim')
        self.register_buffer('state_scales', torch.tensor(resolved_state_scales, dtype=torch.float32).clamp_min(1e-08))
        self.t_start = float(t_span[0])
        self.t_duration = max(float(t_span[1]) - self.t_start, 1e-08)
        self._phys_raw = nn.ParameterDict()
        self._phys_scales: dict[str, float] = {}
        if param_names and param_init_values:
            for name in param_names:
                init_val = param_init_values.get(name, 1.0)
                scale = max(abs(init_val), 1e-08)
                self._phys_scales[name] = scale
                raw_val = _inverse_softplus(init_val / scale)
                self._phys_raw[name] = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        tau = (t - self.t_start) / self.t_duration
        y_raw = self.net(tau)
        if self.hard_initial_condition:
            y_raw = self.initial_condition.unsqueeze(0) + tau * y_raw * self.state_scales.unsqueeze(0)
        if self.output_transform_name == 'smooth_abs':
            eps = y_raw.new_tensor(1e-06)
            smooth = torch.sqrt(y_raw.square() + eps)
            initial_smooth = torch.sqrt(self.initial_condition.square() + eps).unsqueeze(0)
            y = smooth + self.initial_condition.unsqueeze(0) - initial_smooth
        else:
            y = self.output_transform(y_raw)
        return y

    def compute_derivatives(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dy_dt_components = []
        for i in range(self.state_dim):
            dy_i = torch.autograd.grad(outputs=y[:, i], inputs=t, grad_outputs=torch.ones_like(y[:, i]), create_graph=True, retain_graph=True)[0]
            dy_dt_components.append(dy_i)
        return torch.cat(dy_dt_components, dim=1)

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        return {name: self._phys_scales[name] * F.softplus(raw) for name, raw in self._phys_raw.items()}

    def get_raw_phys_params(self):
        return self._phys_raw.parameters()
