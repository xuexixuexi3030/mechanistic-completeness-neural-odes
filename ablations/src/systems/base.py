from abc import ABC, abstractmethod
import numpy as np
import torch
from scipy.integrate import solve_ivp

class ODESystem(ABC):

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        pass

    @property
    def known_param_names(self) -> list[str]:
        return self.param_names

    @property
    def unknown_component_mask(self) -> list[float]:
        return [1.0] * self.state_dim

    @property
    def unknown_latent_dim(self) -> int:
        return self.state_dim

    def map_unknown_correction(self, latent: torch.Tensor, state: torch.Tensor | None=None, constrained: bool=False) -> torch.Tensor:
        mask = latent.new_tensor(self.unknown_component_mask)
        if constrained:
            latent = torch.nn.functional.softplus(latent)
        return latent * mask

    @property
    @abstractmethod
    def true_params(self) -> dict[str, float]:
        pass

    @abstractmethod
    def initial_condition(self) -> torch.Tensor:
        pass

    @abstractmethod
    def rhs_numpy(self, t: float, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rhs(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def rhs_known(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def rhs_unknown(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def rhs_parametric(self, t: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.rhs(t, y, params)

    def generate_data(self, t_span: tuple[float, float], num_points: int, noise_std: float=0.0, seed: int=42) -> tuple[torch.Tensor, torch.Tensor]:
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        y0 = self.initial_condition().numpy()
        sol = solve_ivp(self.rhs_numpy, t_span, y0, method='Radau', t_eval=t_eval, rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError(f'ODE integration failed: {sol.message}')
        t_tensor = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(sol.y.T, dtype=torch.float32)
        if noise_std > 0:
            rng = np.random.RandomState(seed)
            noise = rng.randn(*y_tensor.shape) * noise_std
            y_tensor = y_tensor + torch.tensor(noise, dtype=torch.float32)
        return (t_tensor, y_tensor)

    def physics_residual(self, t: torch.Tensor, y: torch.Tensor, dy_dt: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return dy_dt - self.rhs(t, y, params)
