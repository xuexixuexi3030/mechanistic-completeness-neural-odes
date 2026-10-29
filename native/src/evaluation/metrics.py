import torch
import numpy as np
from ..systems.base import ODESystem

def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.mean((y_pred - y_true) ** 2).item()

def relative_l2_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    with torch.no_grad():
        num = torch.norm(y_pred - y_true, p=2)
        den = torch.norm(y_true, p=2)
        if den < 1e-12:
            return float('inf')
        return (num / den).item()

def physics_residual_norm(t: torch.Tensor, y_pred: torch.Tensor, system: ODESystem, params: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        n = t.shape[0]
        if n < 2:
            return 0.0
        t_flat = t.squeeze(1) if t.dim() > 1 else t
        dt = t_flat[1:] - t_flat[:-1]
        dy = y_pred[1:] - y_pred[:-1]
        dy_dt = dy / dt.unsqueeze(1)
        t_mid = (t_flat[1:] + t_flat[:-1]) / 2.0
        y_mid = (y_pred[1:] + y_pred[:-1]) / 2.0
        residuals = []
        use_full_parametric_rhs = all((name in params for name in system.param_names))
        for i in range(n - 1):
            if use_full_parametric_rhs:
                f_val = system.rhs_parametric(t_mid[i], y_mid[i], params)
            else:
                f_val = system.rhs(t_mid[i], y_mid[i], params)
            res = dy_dt[i] - f_val
            residuals.append(res)
        residuals = torch.stack(residuals)
        return torch.sqrt(torch.mean(residuals ** 2)).item()

def parameter_estimation_error(estimated: dict[str, torch.Tensor], true_values: dict[str, float]) -> dict[str, dict[str, float]]:
    results = {}
    for name, est in estimated.items():
        est_val = est.item() if torch.is_tensor(est) else est
        true_val = true_values.get(name)
        if true_val is None:
            continue
        abs_err = abs(est_val - true_val)
        rel_err = abs_err / (abs(true_val) + 1e-12)
        results[name] = {'estimated': est_val, 'true': true_val, 'absolute_error': abs_err, 'relative_error': rel_err}
    return results

def evaluate_model(model, system: ODESystem, data: dict, device: torch.device) -> dict:
    model.eval()
    t_test = data['t_test'].to(device)
    y_test = data['y_test'].to(device)
    y_test_clean = data.get('y_test_clean')
    if y_test_clean is not None:
        y_test_clean = y_test_clean.to(device)
    t_full = data['t_full'].to(device)
    y_full = data['y_full'].to(device)
    y0 = data['y0'].to(device)
    metrics = {}
    with torch.no_grad():
        if hasattr(model, 'compute_derivatives'):
            y_pred_test = model(t_test)
            y_pred_full = model(t_full)
        else:
            t_test_flat = t_test.squeeze(1)
            t_full_flat = t_full.squeeze(1)
            sort_idx = torch.argsort(t_test_flat)
            t_sorted = t_test_flat[sort_idx]
            y_test_sorted = y_test[sort_idx]
            y_test = y_test_sorted
            if y_test_clean is not None:
                y_test_clean = y_test_clean[sort_idx]
            sort_idx_full = torch.argsort(t_full_flat)
            t_full_sorted = t_full_flat[sort_idx_full]
            y_pred_full = model(y0, t_full_sorted)
            y_full = y_full[sort_idx_full]
            test_positions = torch.searchsorted(t_full_sorted, t_sorted)
            if torch.any(test_positions >= t_full_sorted.numel()) or not torch.allclose(t_full_sorted[test_positions], t_sorted, rtol=1e-05, atol=1e-07):
                raise ValueError('Test times are not contained in the full evaluation grid')
            y_pred_test = y_pred_full[test_positions]
    metrics['test_mse'] = mse(y_pred_test, y_test)
    metrics['test_relative_l2'] = relative_l2_error(y_pred_test, y_test)
    metrics['full_mse'] = mse(y_pred_full, y_full)
    metrics['full_relative_l2'] = relative_l2_error(y_pred_full, y_full)
    if y_test_clean is not None:
        metrics['test_mse_clean'] = mse(y_pred_test, y_test_clean)
        metrics['test_relative_l2_clean'] = relative_l2_error(y_pred_test, y_test_clean)
    for i in range(system.state_dim):
        comp_name = f'component_{i}'
        metrics[f'{comp_name}_mse'] = mse(y_pred_test[:, i], y_test[:, i])
        metrics[f'{comp_name}_relative_l2'] = relative_l2_error(y_pred_test[:, i], y_test[:, i])
        if y_test_clean is not None:
            metrics[f'{comp_name}_mse_clean'] = mse(y_pred_test[:, i], y_test_clean[:, i])
            metrics[f'{comp_name}_relative_l2_clean'] = relative_l2_error(y_pred_test[:, i], y_test_clean[:, i])
    if 'y_train_clean' in data:
        y_train = data['y_train'].to(device)
        y_train_clean = data['y_train_clean'].to(device)
        metrics['observation_noise_train_relative_l2'] = relative_l2_error(y_train, y_train_clean)
        for i in range(system.state_dim):
            metrics[f'observation_noise_component_{i}_relative_l2'] = relative_l2_error(y_train[:, i], y_train_clean[:, i])
        metrics['noise_mode'] = data.get('noise_mode', 'absolute')
        metrics['noise_std_configured'] = data.get('noise_std_configured', 0.0)
        metrics['preserve_initial_observation'] = data.get('preserve_initial_observation', False)
    if system.true_params and all((name in system.true_params for name in system.param_names)):
        true_params = {name: torch.as_tensor(system.true_params[name], dtype=y_pred_full.dtype, device=y_pred_full.device) for name in system.param_names}
        t_eval = t_full if t_full.dim() > 1 else t_full.unsqueeze(1)
        metrics['physics_residual_true_param_norm'] = physics_residual_norm(t_eval, y_pred_full, system, true_params)
    if hasattr(model, 'get_params_dict'):
        params = model.get_params_dict()
        if params:
            t_eval = t_full if t_full.dim() > 1 else t_full.unsqueeze(1)
            metrics['physics_residual_norm'] = physics_residual_norm(t_eval, y_pred_full, system, params)
            param_errors = parameter_estimation_error(params, system.true_params)
            for pname, errs in param_errors.items():
                metrics[f'param_{pname}_estimated'] = errs['estimated']
                metrics[f'param_{pname}_true'] = errs['true']
                metrics[f'param_{pname}_abs_error'] = errs['absolute_error']
                metrics[f'param_{pname}_rel_error'] = errs['relative_error']
    return metrics
