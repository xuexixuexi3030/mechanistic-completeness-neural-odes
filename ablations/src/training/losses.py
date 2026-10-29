import torch
import torch.nn as nn

def data_loss(y_pred: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_obs) ** 2)

def data_loss_normalized(y_pred: torch.Tensor, y_obs: torch.Tensor, normalization_floor: float=1.0) -> torch.Tensor:
    scale = torch.clamp(y_obs.abs().max(dim=0).values, min=float(normalization_floor))
    residual = (y_pred - y_obs) / scale.unsqueeze(0)
    return torch.mean(residual ** 2)

def data_loss_componentwise(y_pred: torch.Tensor, y_obs: torch.Tensor, normalize: bool=False, normalization_floor: float=1.0, component_weights: list[float] | tuple[float, ...] | None=None, component_max_weight: float=0.0) -> dict[str, torch.Tensor]:
    state_dim = y_pred.shape[-1]
    if normalize:
        scale = torch.clamp(y_obs.abs().max(dim=0).values, min=float(normalization_floor))
    else:
        scale = torch.ones(state_dim, device=y_pred.device)
    components = {}
    component_values = []
    for i in range(state_dim):
        comp_loss = torch.mean(((y_pred[:, i] - y_obs[:, i]) / scale[i]) ** 2)
        components[f'data_comp{i}'] = comp_loss
        component_values.append(comp_loss)
    stacked = torch.stack(component_values)
    components['data_unweighted'] = torch.mean(stacked)
    if component_weights:
        if len(component_weights) != state_dim:
            raise ValueError('component_weights must match the state dimension')
        weights = stacked.new_tensor(component_weights)
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError('component_weights must be finite and positive')
        weights = weights / torch.mean(weights)
        mean_loss = torch.mean(weights * stacked)
    else:
        mean_loss = components['data_unweighted']
    if not 0.0 <= component_max_weight < 1.0:
        raise ValueError('component_max_weight must be in [0, 1)')
    bottleneck_loss = torch.max(stacked)
    components['data_bottleneck'] = bottleneck_loss
    components['data'] = (1.0 - component_max_weight) * mean_loss + component_max_weight * bottleneck_loss
    return components

def physics_residual_loss(dy_dt_learned: torch.Tensor, dy_dt_physics: torch.Tensor, normalize: bool=False) -> torch.Tensor:
    if normalize:
        scale = torch.clamp(dy_dt_physics.detach().abs().max(dim=0).values, min=1e-06)
        residual = (dy_dt_learned - dy_dt_physics) / scale.unsqueeze(0)
        return torch.mean(residual ** 2)
    return torch.mean((dy_dt_learned - dy_dt_physics) ** 2)

def initial_condition_loss(y_pred_0: torch.Tensor, y0_true: torch.Tensor, normalize: bool=False, normalization_floor: float=1.0) -> torch.Tensor:
    if normalize:
        scale = torch.clamp(y0_true.abs(), min=float(normalization_floor))
        return torch.mean(((y_pred_0 - y0_true) / scale) ** 2)
    return torch.mean((y_pred_0 - y0_true) ** 2)

def regularization_loss(model: nn.Module, target_modules: tuple=()) -> torch.Tensor:
    first_param = next(model.parameters(), None)
    if first_param is None:
        return torch.tensor(0.0, dtype=torch.float32)
    reg = first_param.new_zeros(())
    excluded_param_ids = set()
    if hasattr(model, 'get_raw_phys_params'):
        excluded_param_ids = {id(param) for param in model.get_raw_phys_params()}
    if not target_modules:
        for param in model.parameters():
            if id(param) not in excluded_param_ids:
                reg = reg + torch.sum(param ** 2)
    else:
        for module in model.modules():
            if isinstance(module, target_modules):
                for param in module.parameters():
                    if id(param) not in excluded_param_ids:
                        reg = reg + torch.sum(param ** 2)
    return reg

def total_loss(components: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, dtype=torch.float32, device=next(iter(components.values())).device)
    logged = {}
    for name, value in components.items():
        w = weights.get(name, 1.0)
        total = total + w * value
        logged[name] = value.item()
    logged['total'] = total.item()
    return (total, logged)
