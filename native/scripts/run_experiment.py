import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import torch
from src.config import load_config, apply_overrides
from src.utils.seed import set_all_seeds
from src.utils.device import resolve_device, restart_in_project_venv_for_cuda
from src.utils.logging import ExperimentLogger
from src.utils.visualization import plot_solution_comparison, plot_parameter_convergence, plot_loss_history
from src.systems import SYSTEM_REGISTRY
from src.models import MODEL_REGISTRY
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.evaluation.profiler import TrainingProfiler

def build_system(config):
    system_cls = SYSTEM_REGISTRY[config.system.name]
    kwargs = {}
    if config.system.true_params:
        kwargs['params'] = config.system.true_params
    if config.system.initial_conditions:
        kwargs['initial_conditions'] = config.system.initial_conditions
    return system_cls(**kwargs)

def build_model(config, system):
    mc = config.model
    mode = mc.mode
    common = {'hidden_layers': mc.hidden_layers, 'hidden_units': mc.hidden_units, 'activation': mc.activation}
    param_names = system.known_param_names if mode == 'mode_a' else system.param_names
    if config.system.param_init:
        missing = set(param_names).difference(config.system.param_init)
        if missing:
            raise ValueError(f'Missing initial guesses for parameters: {sorted(missing)}')
        param_init = {name: float(config.system.param_init[name]) for name in param_names}
    else:
        true_p = system.true_params
        param_init = {name: true_p[name] * 1.5 for name in param_names}
    if mode == 'pinn':
        from src.models.pinn import PINNModel
        return PINNModel(state_dim=system.state_dim, init_method=mc.init_method, param_names=param_names, param_init_values=param_init, positive_output=mc.positive_output, output_transform=mc.pinn_output_transform or ('abs' if mc.positive_output else 'none'), initial_condition=system.initial_condition(), t_span=config.system.t_span, state_scales=mc.pinn_state_scales, **common)
    elif mode == 'node':
        from src.models.neural_ode import NeuralODESolver
        return NeuralODESolver(state_dim=system.state_dim, hidden_layers=mc.node_hidden_layers, hidden_units=mc.node_hidden_units, activation=mc.activation, solver=mc.ode_solver, rtol=mc.ode_rtol, atol=mc.ode_atol, step_size=mc.ode_step_size, use_adjoint=mc.use_adjoint, t_span=config.system.t_span, normalize_inputs=mc.node_normalize_inputs, state_scales=mc.node_state_scales, output_scales=mc.node_output_scales)
    elif mode == 'mode_a':
        from src.models.mode_a import ModeAModel
        return ModeAModel(system=system, node_hidden_layers=mc.node_hidden_layers, node_hidden_units=mc.node_hidden_units, param_names=param_names, param_init_values=param_init, solver=mc.ode_solver, rtol=mc.ode_rtol, atol=mc.ode_atol, step_size=mc.ode_step_size, use_adjoint=mc.use_adjoint, t_span=config.system.t_span, use_correction=mc.mode_a_use_correction, normalize_inputs=mc.mode_a_normalize_inputs, constrain_unknown=mc.mode_a_constrain_unknown, state_scales=mc.mode_a_state_scales, **common)
    elif mode == 'mode_b':
        from src.models.mode_b import ModeBModel
        return ModeBModel(state_dim=system.state_dim, node_hidden_layers=mc.node_hidden_layers, node_hidden_units=mc.node_hidden_units, param_names=param_names, param_init_values=param_init, solver=mc.ode_solver, rtol=mc.ode_rtol, atol=mc.ode_atol, step_size=mc.ode_step_size, use_adjoint=mc.use_adjoint, t_span=config.system.t_span, normalize_inputs=mc.mode_b_normalize_inputs, state_scales=mc.mode_b_state_scales, output_scales=mc.mode_b_output_scales, **common)
    elif mode == 'mode_c':
        from src.models.mode_c import ModeCModel
        return ModeCModel(system=system, node_hidden_layers=mc.node_hidden_layers, node_hidden_units=mc.node_hidden_units, param_names=param_names, param_init_values=param_init, solver=mc.ode_solver, rtol=mc.ode_rtol, atol=mc.ode_atol, step_size=mc.ode_step_size, use_adjoint=mc.use_adjoint, correction_scale=mc.correction_scale, use_correction=mc.mode_c_use_correction, use_physics=mc.mode_c_use_physics, bound_correction=mc.mode_c_bound_correction, **common)
    else:
        raise ValueError(f'Unknown model mode: {mode}')

def load_initial_checkpoint(model, checkpoint_path: str) -> Path | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Initial checkpoint was not found: {path}')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    state_dict = payload['model_state_dict'] if isinstance(payload, dict) and 'model_state_dict' in payload else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f'Checkpoint does not contain a state_dict: {path}')
    model.load_state_dict(state_dict, strict=True)
    return path

def run_experiment(config_path: str, overrides: list[str] | None=None):
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)
    device = resolve_device(config.device)
    config.device = str(device)
    if not config.experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config.experiment_name = f'{config.model.mode}_{config.system.name}_{timestamp}'
    print(f'=' * 60)
    print(f'Experiment: {config.experiment_name}')
    print(f'Mode: {config.model.mode} | System: {config.system.name}')
    if device.type == 'cuda':
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f'Device: {device} ({torch.cuda.get_device_name(gpu_index)})')
    else:
        print(f'Device: {device}')
    print(f'=' * 60)
    set_all_seeds(config.training.seed)
    output_dir = os.path.join(config.output_dir, config.experiment_name)
    logger = ExperimentLogger(output_dir)
    system = build_system(config)
    model = build_model(config, system)
    initialized_from = load_initial_checkpoint(model, config.training.init_checkpoint)
    if initialized_from is not None:
        print(f'Warm start: {initialized_from}')
    param_count = sum((p.numel() for p in model.parameters()))
    print(f'Model parameters: {param_count:,}')
    profiler = TrainingProfiler()
    profiler.start()
    trainer = Trainer(model, system, config, logger)
    result = trainer.train()
    profiler.stop()
    profile_data = profiler.get_results()
    print(f'\nTraining completed in {result.train_time:.1f}s')
    print('\nEvaluating on test set...')
    metrics = evaluate_model(model, system, result.data, device)
    metrics.update(profile_data)
    metrics['num_parameters'] = param_count
    metrics['best_epoch'] = result.best_epoch
    metrics['best_training_loss'] = result.best_loss
    metrics['stopped_early'] = result.stopped_early
    metrics['used_ema'] = result.used_ema
    metrics['update_clip_events'] = result.update_clip_events
    metrics['loss_backtrack_events'] = result.loss_backtrack_events
    metrics['loss_backtrack_reductions'] = result.loss_backtrack_reductions
    metrics['loss_backtrack_rejections'] = result.loss_backtrack_rejections
    metrics['data_source'] = result.data.get('data_source', 'synthetic')
    if 'data_path' in result.data:
        metrics['data_path'] = result.data['data_path']
    if initialized_from is not None:
        metrics['initialized_from'] = str(initialized_from)
    if result.ema_selection_loss is not None:
        metrics['ema_selection_loss'] = result.ema_selection_loss
    if result.stop_reason:
        metrics['stop_reason'] = result.stop_reason
    print(f'\n{'Metric':<35} {'Value':>15}')
    print('-' * 52)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f'{key:<35} {value:>15.6e}')
        else:
            print(f'{key:<35} {str(value):>15}')
    logger.save_metrics(metrics)
    observation_state_names = None
    if hasattr(system, 'state_names'):
        observation_state_names = list(system.state_names)
    elif config.system.name == 'gene_regulatory':
        observation_state_names = ['Prom', 'TF', 'PromTF']
    elif config.system.name == 'eco_epidemic':
        observation_state_names = ['S', 'I', 'R', 'N']
    logger.save_observations(result.data, observation_state_names)
    print('\nGenerating plots...')
    _generate_plots(model, system, result, config, logger, device)
    logger.close()
    print(f'\nResults saved to: {output_dir}')
    return metrics

def _generate_plots(model, system, result, config, logger, device):
    import numpy as np
    data = result.data
    y0 = data['y0'].to(device)
    t_full = data['t_full'].to(device)
    y_full = data['y_full'].to(device)
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'compute_derivatives'):
            y_pred = model(t_full).cpu().numpy()
        else:
            t_flat = t_full.squeeze(1)
            sort_idx = torch.argsort(t_flat)
            t_sorted = t_flat[sort_idx]
            y_pred = model(y0, t_sorted).cpu().numpy()
            y_full = y_full[sort_idx]
    t_np = t_full.squeeze(1).cpu().numpy()
    if hasattr(torch.Tensor, 'numpy'):
        y_true_np = y_full.cpu().numpy() if torch.is_tensor(y_full) else y_full
    if hasattr(system, 'state_names'):
        comp_names = list(system.state_names)
    elif config.system.name == 'gene_regulatory':
        comp_names = ['Promoter', 'TF', 'PromTF']
    elif config.system.name == 'eco_epidemic':
        comp_names = ['S (Susceptible)', 'I (Infected)', 'R (Recovered)', 'N (Total)']
    else:
        comp_names = [f'Component {i}' for i in range(system.state_dim)]
    plot_solution_comparison(t_np, y_true_np, t_np, y_pred, component_names=comp_names, title=f'{config.model.mode.upper()} - {config.system.name}', save_path=logger.get_figure_path('solution_comparison.pdf'))
    if result.param_history and system.true_params:
        plot_parameter_convergence(result.param_history, system.true_params, title='Parameter Convergence', save_path=logger.get_figure_path('parameter_convergence.pdf'))
    if result.history:
        plot_loss_history(result.history, title='Training Loss History', save_path=logger.get_figure_path('loss_history.pdf'))

def main():
    parser = argparse.ArgumentParser(description='Run Neural ODE-PINN fusion experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default=None, help='Execution device (default: value from config, normally cuda)')
    parser.add_argument('--override', nargs='*', default=[], help='Config overrides (key=value)')
    args = parser.parse_args()
    restart_in_project_venv_for_cuda(args.device, 'scripts.run_experiment')
    overrides = list(args.override)
    if args.device is not None:
        overrides.append(f'device={args.device}')
    run_experiment(args.config, overrides)
if __name__ == '__main__':
    main()
