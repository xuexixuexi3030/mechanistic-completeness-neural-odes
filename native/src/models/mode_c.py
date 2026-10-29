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

class HybridDynamics(nn.Module):

    def __init__(self, system: ODESystem, phys_params: nn.ParameterDict, phys_scales: dict[str, float], correction_net: SmallFNN | None=None, correction_scale: float=0.1, use_physics: bool=True, bound_correction: bool=True):
        super().__init__()
        self.system = system
        self.phys_params = phys_params
        self.phys_scales = phys_scales
        self.correction_net = correction_net
        self.correction_scale = correction_scale
        self.use_physics = use_physics
        self.bound_correction = bound_correction

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = {name: self.phys_scales[name] * F.softplus(param) for name, param in self.phys_params.items()}
        physics = self.system.rhs_parametric(t, y, params) if self.use_physics else torch.zeros_like(y)
        if self.correction_net is not None:
            if y.dim() == 1:
                t_input = t.reshape(1)
                net_input = torch.cat([t_input, y])
            else:
                t_input = t.reshape(1).expand(y.shape[0]).unsqueeze(1)
                net_input = torch.cat([t_input, y], dim=1)
            correction = self.correction_net(net_input)
            if self.bound_correction:
                correction = torch.tanh(correction)
            physics = physics + self.correction_scale * correction
        return physics

class ModeCModel(nn.Module):

    def __init__(self, system: ODESystem, hidden_layers: int=4, hidden_units: int=50, node_hidden_layers: int=3, node_hidden_units: int=32, activation: str='tanh', param_names: list[str] | None=None, param_init_values: dict[str, float] | None=None, solver: str='dopri5', rtol: float=1e-07, atol: float=1e-09, step_size: float | None=None, use_adjoint: bool=True, use_correction: bool=True, correction_scale: float=0.1, use_physics: bool=True, bound_correction: bool=True, **kwargs):
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
            correction_net = SmallFNN(input_dim=system.state_dim + 1, output_dim=system.state_dim, hidden_layers=node_hidden_layers, hidden_units=node_hidden_units, activation=activation)
        self.dynamics = HybridDynamics(system, self.phys_params, self.phys_scales, correction_net, correction_scale=correction_scale, use_physics=use_physics, bound_correction=bound_correction)

    def forward(self, y0: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        solution = self.integrator(self.dynamics, y0, t_eval, method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        return solution

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        return {name: self.phys_scales[name] * F.softplus(param) for name, param in self.phys_params.items()}

    def get_raw_phys_params(self):
        return self.phys_params.parameters()
