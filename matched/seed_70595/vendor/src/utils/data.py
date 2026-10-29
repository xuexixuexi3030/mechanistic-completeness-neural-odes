import csv
from pathlib import Path
import torch
import numpy as np
from ..systems.base import ODESystem
from ..config import ExperimentConfig

def generate_synthetic_data(system: ODESystem, config: ExperimentConfig) -> dict:
    sc = config.system
    t_span = (sc.t_span[0], sc.t_span[1])
    total_points = sc.num_train_points + sc.num_test_points
    t_full, y_full = system.generate_data(t_span, total_points, noise_std=0.0, seed=config.training.seed)
    noise_mode = sc.noise_mode.lower().strip()
    if noise_mode == 'absolute':
        t_obs, y_obs = system.generate_data(t_span, total_points, noise_std=sc.noise_std, seed=config.training.seed)
        noise_scales = torch.full((system.state_dim,), float(sc.noise_std), dtype=y_full.dtype)
    elif noise_mode == 'relative_state_std':
        if not 0.0 <= sc.noise_std <= 1.0:
            raise ValueError('relative_state_std noise_std must be a fraction in [0, 1]')
        t_obs = t_full.clone()
        state_std = torch.std(y_full, dim=0, correction=0)
        noise_scales = state_std * float(sc.noise_std)
        rng = np.random.RandomState(config.training.seed)
        standard_noise = torch.tensor(rng.randn(*y_full.shape), dtype=y_full.dtype)
        y_obs = y_full + standard_noise * noise_scales.unsqueeze(0)
    else:
        raise ValueError("system.noise_mode must be 'absolute' or 'relative_state_std'")
    if sc.preserve_initial_observation:
        y_obs[0] = y_full[0]
    n_train = sc.num_train_points
    if sc.split_strategy == 'uniform':
        train_idx = torch.linspace(0, total_points - 1, n_train).round().long().unique()
        if train_idx.numel() != n_train:
            raise ValueError('Uniform split produced duplicate training indices')
        test_mask = torch.ones(total_points, dtype=torch.bool)
        test_mask[train_idx] = False
        test_idx = torch.arange(total_points)[test_mask]
    elif sc.split_strategy == 'temporal':
        train_idx = torch.arange(n_train)
        test_idx = torch.arange(n_train, total_points)
    else:
        raise ValueError(f"Unknown split_strategy '{sc.split_strategy}'; use 'uniform' or 'temporal'")
    return {'t_train': t_obs[train_idx], 'y_train': y_obs[train_idx], 'y_train_clean': y_full[train_idx], 't_test': t_obs[test_idx], 'y_test': y_obs[test_idx], 'y_test_clean': y_full[test_idx], 't_full': t_full, 'y_full': y_full, 'y_observed_full': y_obs, 'train_indices': train_idx, 'test_indices': test_idx, 'y0': system.initial_condition(), 'data_source': 'synthetic', 'noise_mode': noise_mode, 'noise_std_configured': float(sc.noise_std), 'noise_state_scales': noise_scales, 'preserve_initial_observation': bool(sc.preserve_initial_observation)}

def load_csv_observations(system: ODESystem, config: ExperimentConfig) -> dict:
    sc = config.system
    if sc.noise_std != 0.0:
        raise ValueError('noise_std must be 0 for measured CSV observations')
    if not sc.data_path:
        raise ValueError("system.data_path is required when data_source='csv'")
    if len(sc.state_columns) != system.state_dim:
        raise ValueError('system.state_columns must match system.state_dim')
    path = Path(sc.data_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Measured data CSV was not found: {path}')
    times: list[float] = []
    states: list[list[float]] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {sc.time_column, *sc.state_columns}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing CSV columns in {path}: {sorted(missing)}')
        for row_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row[sc.time_column]))
                states.append([float(row[name]) for name in sc.state_columns])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Non-numeric measured value in {path} at row {row_number}') from exc
    if len(times) < 3:
        raise ValueError('Measured CSV must contain at least three time points')
    t_full = torch.tensor(times, dtype=torch.float32).unsqueeze(1)
    y_full = torch.tensor(states, dtype=torch.float32)
    if not torch.isfinite(t_full).all() or not torch.isfinite(y_full).all():
        raise ValueError('Measured CSV contains NaN or infinite values')
    if torch.any(t_full[1:, 0] <= t_full[:-1, 0]):
        raise ValueError('Measured times must be strictly increasing')
    if torch.any(y_full < 0):
        raise ValueError('Measured state values must be non-negative')
    expected_span = torch.tensor(sc.t_span, dtype=torch.float32)
    actual_span = torch.stack((t_full[0, 0], t_full[-1, 0]))
    if not torch.allclose(expected_span, actual_span, rtol=1e-06, atol=1e-06):
        raise ValueError(f'Configured t_span {sc.t_span} does not match CSV span {actual_span.tolist()}')
    configured_y0 = system.initial_condition()
    if not torch.allclose(configured_y0, y_full[0], rtol=1e-05, atol=1e-07):
        raise ValueError('Configured initial_conditions must equal the first measured CSV row')
    total_points = t_full.shape[0]
    n_train = sc.num_train_points
    if not 1 < n_train < total_points:
        raise ValueError(f'num_train_points must be between 2 and {total_points - 1} for {path}')
    if sc.split_strategy == 'uniform':
        train_idx = torch.linspace(0, total_points - 1, n_train).round().long().unique()
        if train_idx.numel() != n_train:
            raise ValueError('Uniform split produced duplicate measured-data indices')
        test_mask = torch.ones(total_points, dtype=torch.bool)
        test_mask[train_idx] = False
        test_idx = torch.arange(total_points)[test_mask]
    elif sc.split_strategy == 'temporal':
        train_idx = torch.arange(n_train)
        test_idx = torch.arange(n_train, total_points)
    else:
        raise ValueError(f"Unknown split_strategy '{sc.split_strategy}'; use 'uniform' or 'temporal'")
    return {'t_train': t_full[train_idx], 'y_train': y_full[train_idx], 't_test': t_full[test_idx], 'y_test': y_full[test_idx], 't_full': t_full, 'y_full': y_full, 'y0': y_full[0].clone(), 'data_source': 'csv', 'data_path': str(path), 'train_indices': train_idx, 'test_indices': test_idx}

def load_experiment_data(system: ODESystem, config: ExperimentConfig) -> dict:
    source = config.system.data_source.lower().strip()
    if source == 'synthetic':
        return generate_synthetic_data(system, config)
    if source == 'csv':
        return load_csv_observations(system, config)
    raise ValueError("system.data_source must be 'synthetic' or 'csv'")

def add_noise(y: torch.Tensor, noise_std: float, seed: int=42) -> torch.Tensor:
    if noise_std <= 0:
        return y
    rng = np.random.RandomState(seed)
    noise = rng.randn(*y.shape) * noise_std
    return y + torch.tensor(noise, dtype=y.dtype)
