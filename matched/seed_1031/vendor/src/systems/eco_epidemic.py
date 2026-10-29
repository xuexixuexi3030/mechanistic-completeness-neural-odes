import numpy as np
import torch
import torch.nn.functional as F
from .base import ODESystem

class EcoEpidemicSystem(ODESystem):
    DEFAULT_PARAMS = {'r': 0.5, 'K': 1000.0, 'beta': 0.3, 'gamma': 0.1, 'mu': 0.01}
    DEFAULT_IC = [950.0, 50.0, 0.0, 1000.0]

    def __init__(self, params: dict[str, float] | None=None, initial_conditions: list[float] | None=None):
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._ic = initial_conditions or self.DEFAULT_IC

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def param_names(self) -> list[str]:
        return ['r', 'K', 'beta', 'gamma']

    @property
    def known_param_names(self) -> list[str]:
        return ['r', 'K', 'gamma']

    @property
    def unknown_component_mask(self) -> list[float]:
        return [1.0, 1.0, 0.0, 1.0]

    @property
    def unknown_latent_dim(self) -> int:
        return 2

    def map_unknown_correction(self, latent: torch.Tensor, state: torch.Tensor | None=None, constrained: bool=False) -> torch.Tensor:
        if constrained and state is not None:
            S = torch.clamp(state[..., 0], min=0.0)
            I = torch.clamp(state[..., 1], min=0.0)
            N = torch.clamp(state[..., 3], min=1e-06)
            positive_rates = F.softplus(latent)
            infection_rate = 0.5 * positive_rates[..., 0]
            mortality_rate = 0.02 * positive_rates[..., 1]
            q = infection_rate * S * I / N
            mortality = mortality_rate * I
        else:
            q = latent[..., 0]
            mortality = latent[..., 1]
        zero = torch.zeros_like(q)
        return torch.stack([-q, q - mortality, zero, -mortality], dim=-1)

    @property
    def true_params(self) -> dict[str, float]:
        return self._params.copy()

    def initial_condition(self) -> torch.Tensor:
        return torch.tensor(self._ic, dtype=torch.float32)

    def rhs_numpy(self, t: float, y: np.ndarray) -> np.ndarray:
        S, I, R, N = y
        p = self._params
        N_safe = max(N, 1e-08)
        dS = p['r'] * S * (1 - N / p['K']) - p['beta'] * S * I / N_safe
        dI = p['beta'] * S * I / N_safe - p['gamma'] * I - p['mu'] * I
        dR = p['gamma'] * I
        dN = p['r'] * S * (1 - N / p['K']) - p['mu'] * I
        return np.array([dS, dI, dR, dN])

    def rhs(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.rhs_known(t, y, params) + self.rhs_unknown(t, y)

    def rhs_parametric(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        S, I, R, N = (y[i] for i in range(4)) if y.dim() == 1 else (y[..., i] for i in range(4))
        r = params['r']
        K = params['K']
        beta = params['beta']
        gamma = params['gamma']
        mu = self._params['mu']
        N_safe = torch.clamp(N, min=1e-08)
        growth = r * S * (1 - N / K)
        infection = beta * S * I / N_safe
        dS = growth - infection
        dI = infection - gamma * I - mu * I
        dR = gamma * I
        dN = growth - mu * I
        if y.dim() == 1:
            return torch.stack([dS, dI, dR, dN])
        return torch.stack([dS, dI, dR, dN], dim=-1)

    def rhs_known(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        if y.dim() == 1:
            S, I, R, N = (y[0], y[1], y[2], y[3])
        else:
            S, I, R, N = (y[..., 0], y[..., 1], y[..., 2], y[..., 3])
        r = params['r']
        K = params['K']
        gamma = params['gamma']
        dS = r * S * (1 - N / K)
        dI = -gamma * I
        dR = gamma * I
        dN = r * S * (1 - N / K)
        if y.dim() == 1:
            return torch.stack([dS, dI, dR, dN])
        return torch.stack([dS, dI, dR, dN], dim=-1)

    def rhs_unknown(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self._params
        if y.dim() == 1:
            S, I, R, N = (y[0], y[1], y[2], y[3])
        else:
            S, I, R, N = (y[..., 0], y[..., 1], y[..., 2], y[..., 3])
        N_safe = torch.clamp(N, min=1e-08)
        infection = p['beta'] * S * I / N_safe
        mortality = p['mu'] * I
        dS = -infection
        dI = infection - mortality
        dR_unknown = torch.zeros_like(I)
        dN = -mortality
        if y.dim() == 1:
            return torch.stack([dS, dI, dR_unknown, dN])
        return torch.stack([dS, dI, dR_unknown, dN], dim=-1)
