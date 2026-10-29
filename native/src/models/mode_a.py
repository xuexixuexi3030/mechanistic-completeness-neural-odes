import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from .networks import SmallFNN
from ..systems.base import ODESystem

def _inverse_softplus(x: float) -> float:
    if x > 20.0:
        return x
    return float(torch.log(torch.exp(torch.tensor(x)) - 1.0))

class AugmentedDynamics(nn.Module):

    def __init__(self, system: ODESystem, phys_params: nn.ParameterDict, phys_scales: dict[str, float], correction_net: SmallFNN | None, t_span: list[float] | tuple[float, float], state_scales: list[float], normalize_inputs: bool, constrain_unknown: bool):
        super().__init__()
        self.system = system
        self.phys_params = phys_params
        self.phys_scales = phys_scales
        self.correction_net = correction_net
        self.normalize_inputs = normalize_inputs
        self.constrain_unknown = constrain_unknown
        t_start, t_end = (float(t_span[0]), float(t_span[1]))
        self.register_buffer('time_center', torch.tensor(0.5 * (t_start + t_end)))
        self.register_buffer('time_half_range', torch.tensor(max(0.5 * (t_end - t_start), 1e-08)))
        self.register_buffer('state_scales', torch.tensor(state_scales, dtype=torch.float32).clamp_min(1e-08))

    def correction(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.correction_net is None:
            return torch.zeros_like(y)
        if self.normalize_inputs:
            t_for_net = (t - self.time_center) / self.time_half_range
            y_for_net = y / self.state_scales
        else:
            t_for_net = t
            y_for_net = y
        if y.dim() == 1:
            t_input = t_for_net.reshape(1)
            net_input = torch.cat([t_input, y_for_net])
        else:
            if t_for_net.numel() == y.shape[0]:
                t_input = t_for_net.reshape(-1, 1)
            else:
                t_input = t_for_net.reshape(1).expand(y.shape[0]).unsqueeze(1)
            net_input = torch.cat([t_input, y_for_net], dim=1)
        latent = self.correction_net(net_input)
        return self.system.map_unknown_correction(latent, state=y, constrained=self.constrain_unknown)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = {name: self.phys_scales[name] * F.softplus(param) for name, param in self.phys_params.items()}
        known = self.system.rhs_known(t, y, params)
        correction = self.correction(t, y)
        return known + correction

class ModeAModel(nn.Module):

    def __init__(self, system: ODESystem, hidden_layers: int=4, hidden_units: int=50, node_hidden_layers: int=3, node_hidden_units: int=32, activation: str='tanh', param_names: list[str] | None=None, param_init_values: dict[str, float] | None=None, solver: str='dopri5', rtol: float=1e-07, atol: float=1e-09, step_size: float | None=None, use_adjoint: bool=True, t_span: list[float] | tuple[float, float]=(0.0, 1.0), use_correction: bool=True, normalize_inputs: bool=False, constrain_unknown: bool=False, state_scales: list[float] | None=None, **kwargs):
        super().__init__()
        self.state_dim = system.state_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = {'step_size': step_size} if step_size is not None else {}
        self.integrator = odeint_adjoint if use_adjoint else odeint
        self.phys_params = nn.ParameterDict()
        self.phys_scales: dict[str, float] = {}
        if param_names and param_init_values:
            for name in param_names:
                init_val = param_init_values.get(name, 1.0)
                scale = max(abs(init_val), 1e-08)
                self.phys_scales[name] = scale
                raw_val = _inverse_softplus(init_val / scale)
                self.phys_params[name] = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))
        correction_net = None
        if use_correction:
            correction_net = SmallFNN(input_dim=system.state_dim + 1, output_dim=system.unknown_latent_dim, hidden_layers=node_hidden_layers, hidden_units=node_hidden_units, activation=activation, init_strategy='small_output' if normalize_inputs else 'all_small')
        if state_scales:
            resolved_state_scales = [float(value) for value in state_scales]
        else:
            resolved_state_scales = [max(abs(float(value)), 1.0) for value in system.initial_condition().tolist()]
        if len(resolved_state_scales) != system.state_dim:
            raise ValueError(f'Mode A state_scales must contain exactly {system.state_dim} values, got {len(resolved_state_scales)}')
        self.dynamics = AugmentedDynamics(system, self.phys_params, self.phys_scales, correction_net, t_span=t_span, state_scales=resolved_state_scales, normalize_inputs=normalize_inputs, constrain_unknown=constrain_unknown)

    def forward(self, y0: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        solution = self.integrator(self.dynamics, y0, t_eval, method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        return solution

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        return {name: self.phys_scales[name] * F.softplus(param) for name, param in self.phys_params.items()}

    def get_raw_phys_params(self):
        return self.phys_params.parameters()

    def get_correction_net(self) -> SmallFNN | None:
        return self.dynamics.correction_net

    def get_correction(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.dynamics.correction(t, y)
