import numpy as np
import torch
import torch.nn.functional as F
from .base import ODESystem

class GeneRegulatorySystem(ODESystem):
    DEFAULT_PARAMS = {'k_on': 1.0 / 60.0, 'k_off': 0.5 / 60.0, 'alpha1': 0.5, 'alpha2': 0.3, 'alpha3': 0.2, 'K_d': 5.0, 'n_hill': 2.0}
    DEFAULT_IC = [17.0, 25.0, 0.0]

    def __init__(self, params: dict[str, float] | None=None, initial_conditions: list[float] | None=None):
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._ic = initial_conditions or self.DEFAULT_IC

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def param_names(self) -> list[str]:
        return ['k_on', 'k_off']

    @property
    def true_params(self) -> dict[str, float]:
        return self._params.copy()

    def initial_condition(self) -> torch.Tensor:
        return torch.tensor(self._ic, dtype=torch.float32)

    @staticmethod
    def _hill(x: float | np.ndarray | torch.Tensor, alpha: float, K_d: float, n: float):
        return alpha * x ** n / (K_d ** n + x ** n)

    def rhs_numpy(self, t: float, y: np.ndarray) -> np.ndarray:
        prom, tf, promtf = y
        p = self._params
        binding = p['k_on'] * prom * tf
        unbinding = p['k_off'] * promtf
        fb1 = self._hill(promtf, p['alpha1'], p['K_d'], p['n_hill'])
        fb2 = self._hill(prom * tf, p['alpha2'], p['K_d'], p['n_hill'])
        fb3 = self._hill(promtf, p['alpha3'], p['K_d'], p['n_hill'])
        dprom_dt = -binding + unbinding + fb1
        dtf_dt = -binding + unbinding + fb2
        dpromtf_dt = binding - unbinding + fb3
        return np.array([dprom_dt, dtf_dt, dpromtf_dt])

    def rhs(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.rhs_known(t, y, params) + self.rhs_unknown(t, y)

    def rhs_known(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        if y.dim() == 1:
            prom, tf, promtf = (y[0], y[1], y[2])
        else:
            prom, tf, promtf = (y[..., 0], y[..., 1], y[..., 2])
        k_on = params['k_on']
        k_off = params['k_off']
        binding = k_on * prom * tf
        unbinding = k_off * promtf
        dprom = -binding + unbinding
        dtf = -binding + unbinding
        dpromtf = binding - unbinding
        if y.dim() == 1:
            return torch.stack([dprom, dtf, dpromtf])
        return torch.stack([dprom, dtf, dpromtf], dim=-1)

    def rhs_unknown(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self._params
        if y.dim() == 1:
            prom, tf, promtf = (y[0], y[1], y[2])
        else:
            prom, tf, promtf = (y[..., 0], y[..., 1], y[..., 2])
        fb1 = self._hill(promtf, p['alpha1'], p['K_d'], p['n_hill'])
        fb2 = self._hill(prom * tf, p['alpha2'], p['K_d'], p['n_hill'])
        fb3 = self._hill(promtf, p['alpha3'], p['K_d'], p['n_hill'])
        if y.dim() == 1:
            return torch.stack([fb1, fb2, fb3])
        return torch.stack([fb1, fb2, fb3], dim=-1)

    def map_unknown_correction(self, latent: torch.Tensor, state: torch.Tensor | None=None, constrained: bool=False) -> torch.Tensor:
        if not constrained or state is None:
            return latent
        positive = F.softplus(latent)
        prom = torch.clamp(state[..., 0], min=0.0)
        tf = torch.clamp(state[..., 1], min=0.0)
        promtf = torch.clamp(state[..., 2], min=0.0)
        promtf_gate = promtf / (1.0 + promtf)
        interaction = prom * tf
        interaction_gate = interaction / (1.0 + interaction)
        scales = latent.new_tensor([0.5, 0.3, 0.2])
        gates = torch.stack([promtf_gate, interaction_gate, promtf_gate], dim=-1)
        return positive * scales * gates
