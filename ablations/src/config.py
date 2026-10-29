from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class SystemConfig:
    name: str = 'gene_regulatory'
    true_params: dict = field(default_factory=dict)
    param_init: dict = field(default_factory=dict)
    initial_conditions: list = field(default_factory=list)
    t_span: list = field(default_factory=lambda: [0.0, 100.0])
    num_train_points: int = 30
    num_test_points: int = 100
    noise_std: float = 0.0
    split_strategy: str = 'uniform'
    data_source: str = 'synthetic'
    data_path: str = ''
    time_column: str = 'time'
    state_columns: list = field(default_factory=list)

@dataclass
class ModelConfig:
    mode: str = 'mode_a'
    hidden_layers: int = 4
    hidden_units: int = 50
    activation: str = 'tanh'
    init_method: str = 'glorot_normal'
    node_hidden_layers: int = 3
    node_hidden_units: int = 32
    ode_solver: str = 'dopri5'
    ode_rtol: float = 1e-07
    ode_atol: float = 1e-09
    ode_step_size: float = 1.0
    use_adjoint: bool = True
    positive_output: bool = True
    pinn_output_transform: str = ''
    pinn_state_scales: list = field(default_factory=list)
    node_normalize_inputs: bool = False
    node_state_scales: list = field(default_factory=list)
    node_output_scales: list = field(default_factory=list)
    correction_scale: float = 0.1
    mode_c_use_correction: bool = True
    mode_c_use_physics: bool = True
    mode_c_bound_correction: bool = True
    mode_a_use_correction: bool = True
    mode_a_normalize_inputs: bool = False
    mode_a_constrain_unknown: bool = False
    mode_a_state_scales: list = field(default_factory=list)
    mode_b_normalize_inputs: bool = False
    mode_b_state_scales: list = field(default_factory=list)
    mode_b_output_scales: list = field(default_factory=list)
    mode_b_neighborhood_scale: float = 0.0
    mode_b_neighborhood_samples: int = 0

@dataclass
class TrainingConfig:
    adam_lr: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-08
    adam_amsgrad: bool = False
    adam_epochs: int = 80000
    adam_decay_start: int = 0
    adam_final_lr: float = 0.0
    lbfgs_epochs: int = 20000
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lambda_data: float = 1.0
    lambda_physics: float = 1.0
    lambda_physics_data: float = 0.0
    lambda_gamma: float = 0.0
    lambda_correction: float = 0.0
    lambda_reg: float = 0.0001
    lambda_ic: float = 10.0
    balancing_strategy: str = 'softadapt'
    softadapt_tau: float = 0.1
    softadapt_window: int = 5
    gradnorm_alpha: float = 1.5
    log_every: int = 200
    checkpoint_every: int = 5000
    seed: int = 42
    num_collocation: int = 500
    phys_lr: float = 0.0
    phys_lr_overrides: dict = field(default_factory=dict)
    loss_type: str = 'mse'
    data_normalization_floor: float = 1.0
    component_loss: bool = False
    data_component_weights: list = field(default_factory=list)
    data_component_max_weight: float = 0.0
    normalize_physics_loss: bool = False
    init_checkpoint: str = ''
    pinn_adaptive_collocation: bool = False
    pinn_adaptive_fraction: float = 0.0
    pinn_candidate_multiplier: int = 4
    pinn_resample_every: int = 500
    ode_window_points: int = 0
    ode_window_epochs: int = 0
    ode_windows_per_epoch: int = 1
    ode_window_strategy: str = 'rotating'
    gradient_clip_norm: float = 5.0
    physics_warmup_epochs: int = 0
    physics_ramp_epochs: int = 0
    physics_ramp_power: float = 1.0
    physics_ramp_schedule: str = 'linear'
    physics_data_derivative_method: str = 'midpoint'
    parameter_refine_start: int = 0
    parameter_refine_network_lr_factor: float = 1.0
    parameter_refine_physics_lr_factor: float = 1.0
    lr_stages: list = field(default_factory=list)
    lr_stage_transition_epochs: int = 0
    max_network_update_norm: float = 0.0
    max_physics_update_norm: float = 0.0
    loss_backtrack_start_epoch: int = 0
    loss_backtrack_end_epoch: int = 0
    loss_backtrack_trigger_ratio: float = 0.0
    loss_backtrack_factor: float = 0.5
    loss_backtrack_max_steps: int = 0
    loss_backtrack_rejection_lr_factor: float = 0.25
    alternating_update_start: int = 0
    alternating_network_steps: int = 0
    alternating_physics_steps: int = 0
    ema_decay: float = 0.0
    ema_start_epoch: int = 0
    ema_use_for_final: bool = False
    ema_max_loss_ratio: float = 1.05
    target_loss: float = 0.0
    target_loss_patience: int = 1
    early_stopping_start_epoch: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_relative_improvement: float = 0.0

@dataclass
class ExperimentConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = 'results'
    device: str = 'cuda'
    experiment_name: str = ''

def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def _dict_to_config(d: dict) -> ExperimentConfig:
    system_dict = d.get('system', {})
    model_dict = d.get('model', {})
    training_dict = d.get('training', {})
    return ExperimentConfig(system=SystemConfig(**system_dict), model=ModelConfig(**model_dict), training=TrainingConfig(**training_dict), output_dir=d.get('output_dir', 'results'), device=d.get('device', 'cuda'), experiment_name=d.get('experiment_name', ''))

def _load_raw_config(yaml_path: Path, stack: tuple[Path, ...]=()) -> dict:
    yaml_path = yaml_path.resolve()
    if yaml_path in stack:
        chain = ' -> '.join((str(path) for path in (*stack, yaml_path)))
        raise ValueError(f'Circular config inheritance: {chain}')
    with yaml_path.open('r') as handle:
        raw = yaml.safe_load(handle) or {}
    base_name = raw.pop('base', None)
    if not base_name:
        return raw
    base_path = yaml_path.parent / base_name
    base_raw = _load_raw_config(base_path, (*stack, yaml_path))
    return _deep_merge(base_raw, raw)

def load_config(yaml_path: str) -> ExperimentConfig:
    return _dict_to_config(_load_raw_config(Path(yaml_path)))

def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    import copy
    cfg = copy.deepcopy(config)
    for override in overrides:
        key, value = override.split('=', 1)
        parts = key.split('.')
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        attr_name = parts[-1]
        current_value = getattr(obj, attr_name)
        if isinstance(current_value, bool):
            parsed = value.lower() in ('true', '1', 'yes')
        elif isinstance(current_value, int):
            parsed = int(value)
        elif isinstance(current_value, float):
            parsed = float(value)
        elif isinstance(current_value, (list, dict)):
            parsed = yaml.safe_load(value)
            if not isinstance(parsed, type(current_value)):
                raise TypeError(f'Override {key!r} must decode to {type(current_value).__name__}, got {type(parsed).__name__}')
        else:
            parsed = value
        setattr(obj, attr_name, parsed)
    return cfg
