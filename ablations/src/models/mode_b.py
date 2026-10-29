import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from .neural_ode import NeuralODEFunc

def _inverse_softplus(x: float) -> float:
    if x > 20.0:
        return x
    return float(torch.log(torch.exp(torch.tensor(x)) - 1.0))

class ModeBModel(nn.Module):

    def __init__(self, state_dim: int, hidden_layers: int=4, hidden_units: int=50, node_hidden_layers: int=3, node_hidden_units: int=32, activation: str='tanh', param_names: list[str] | None=None, param_init_values: dict[str, float] | None=None, solver: str='dopri5', rtol: float=1e-07, atol: float=1e-09, step_size: float | None=None, use_adjoint: bool=True, t_span: list[float] | tuple[float, float]=(0.0, 1.0), normalize_inputs: bool=False, state_scales: list[float] | None=None, output_scales: list[float] | None=None, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = {'step_size': step_size} if step_size is not None else {}
        self.integrator = odeint_adjoint if use_adjoint else odeint
        self.func = NeuralODEFunc(state_dim=state_dim, hidden_layers=node_hidden_layers, hidden_units=node_hidden_units, activation=activation, t_span=t_span, state_scales=state_scales, output_scales=output_scales, normalize_inputs=normalize_inputs, init_strategy='small_output' if normalize_inputs else 'all_small')
        self.phys_params = nn.ParameterDict()
        self.phys_scales: dict[str, float] = {}
        if param_names and param_init_values:
            for name in param_names:
                init_val = param_init_values.get(name, 1.0)
                scale = max(abs(init_val), 1e-08)
                self.phys_scales[name] = scale
                raw_val = _inverse_softplus(init_val / scale)
                self.phys_params[name] = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))

    def forward(self, y0: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        solution = self.integrator(self.func, y0, t_eval, method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        return solution

    def get_learned_derivatives(self, t_eval: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.func(t_eval, y_pred)

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        return {name: self.phys_scales[name] * F.softplus(param) for name, param in self.phys_params.items()}

    def get_raw_phys_params(self):
        return self.phys_params.parameters()

    def get_dynamics_func(self) -> NeuralODEFunc:
        return self.func
