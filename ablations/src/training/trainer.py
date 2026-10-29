import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from ..config import ExperimentConfig
from ..systems.base import ODESystem
from ..models.pinn import PINNModel
from ..models.mode_a import ModeAModel
from ..models.mode_b import ModeBModel
from ..models.mode_c import ModeCModel
from ..models.neural_ode import NeuralODESolver
from .losses import data_loss, data_loss_normalized, data_loss_componentwise, physics_residual_loss, initial_condition_loss, regularization_loss, total_loss
from .loss_balancing import create_balancer
from .optimizer import HybridOptimizer
from ..utils.logging import ExperimentLogger
from ..utils.data import load_experiment_data

@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict]
    param_history: list[dict]
    final_weights: dict[str, float]
    train_time: float
    data: dict
    best_epoch: int
    best_loss: float
    stopped_early: bool
    stop_reason: str | None
    used_ema: bool
    ema_selection_loss: float | None
    update_clip_events: int
    loss_backtrack_events: int
    loss_backtrack_reductions: int
    loss_backtrack_rejections: int

class Trainer:

    def __init__(self, model: nn.Module, system: ODESystem, config: ExperimentConfig, logger: ExperimentLogger):
        self.model = model
        self.system = system
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.is_pinn = isinstance(model, PINNModel)
        self.is_mode_a = isinstance(model, ModeAModel)
        self.is_mode_b = isinstance(model, ModeBModel)
        self.is_pure_node = isinstance(model, NeuralODESolver)
        self.loss_names = ['data']
        if self.is_pinn:
            self.loss_names.extend(['physics', 'ic'])
        elif self.is_mode_b:
            self.loss_names.append('physics')
            if config.training.lambda_physics_data > 0:
                self.loss_names.append('physics_data')
            if config.training.lambda_gamma > 0 and 'gamma' in system.param_names:
                self.loss_names.append('gamma')
        if self.is_mode_a and config.training.lambda_correction > 0:
            self.loss_names.append('correction')
        if config.training.lambda_reg > 0:
            self.loss_names.append('reg')
        tc = config.training
        initial_weights = {'data': tc.lambda_data}
        if 'physics' in self.loss_names:
            initial_weights['physics'] = tc.lambda_physics
        if 'physics_data' in self.loss_names:
            initial_weights['physics_data'] = tc.lambda_physics_data
        if 'gamma' in self.loss_names:
            initial_weights['gamma'] = tc.lambda_gamma
        if 'correction' in self.loss_names:
            initial_weights['correction'] = tc.lambda_correction
        if 'ic' in self.loss_names:
            initial_weights['ic'] = tc.lambda_ic
        if 'reg' in self.loss_names:
            initial_weights['reg'] = tc.lambda_reg
        self.balancer = create_balancer(strategy=tc.balancing_strategy, loss_names=self.loss_names, initial_weights=initial_weights, tau=tc.softadapt_tau, window_size=tc.softadapt_window, alpha=tc.gradnorm_alpha)
        phys_params = None
        phys_param_names = {}
        if hasattr(model, 'get_raw_phys_params'):
            phys_params = list(model.get_raw_phys_params())
            named_container = getattr(model, 'phys_params', None)
            if named_container is None:
                named_container = getattr(model, '_phys_raw', None)
            if named_container is not None:
                phys_param_names = {id(param): name for name, param in named_container.items()}
        self.optimizer = HybridOptimizer(params=model.parameters(), adam_lr=tc.adam_lr, adam_beta1=tc.adam_beta1, adam_beta2=tc.adam_beta2, adam_eps=tc.adam_eps, adam_amsgrad=tc.adam_amsgrad, adam_epochs=tc.adam_epochs, adam_decay_start=tc.adam_decay_start, adam_final_lr=tc.adam_final_lr, lbfgs_lr=tc.lbfgs_lr, lbfgs_max_iter=tc.lbfgs_max_iter, lbfgs_epochs=tc.lbfgs_epochs, phys_params=phys_params if tc.phys_lr > 0 or tc.phys_lr_overrides else None, phys_lr=tc.phys_lr, phys_param_names=phys_param_names, phys_lr_overrides=tc.phys_lr_overrides, parameter_refine_start=tc.parameter_refine_start, parameter_refine_network_lr_factor=tc.parameter_refine_network_lr_factor, parameter_refine_physics_lr_factor=tc.parameter_refine_physics_lr_factor, lr_stages=tc.lr_stages, lr_stage_transition_epochs=tc.lr_stage_transition_epochs, max_network_update_norm=tc.max_network_update_norm, max_physics_update_norm=tc.max_physics_update_norm, alternating_update_start=tc.alternating_update_start, alternating_network_steps=tc.alternating_network_steps, alternating_physics_steps=tc.alternating_physics_steps)
        self._use_normalized = tc.loss_type == 'normalized_mse'
        self._data_normalization_floor = float(tc.data_normalization_floor)
        if not math.isfinite(self._data_normalization_floor) or self._data_normalization_floor <= 0:
            raise ValueError('data_normalization_floor must be finite and positive')
        self._use_component = tc.component_loss
        self._data_component_weights = tuple((float(value) for value in tc.data_component_weights))
        self._data_component_max_weight = float(tc.data_component_max_weight)
        if self._data_component_weights:
            if not self._use_component:
                raise ValueError('data_component_weights requires component_loss=true')
            if len(self._data_component_weights) != system.state_dim:
                raise ValueError('data_component_weights must match system.state_dim')
            if any((not np.isfinite(value) or value <= 0.0 for value in self._data_component_weights)):
                raise ValueError('data_component_weights must be finite and positive')
        if not 0.0 <= self._data_component_max_weight < 1.0:
            raise ValueError('data_component_max_weight must be in [0, 1)')
        if self._data_component_max_weight > 0.0 and (not self._use_component):
            raise ValueError('data_component_max_weight requires component_loss=true')
        self._last_unweighted_data_loss = None
        self._last_uniform_physics_loss = None
        self._last_comp_losses = {}
        self._adaptive_collocation_values = None
        self._adaptive_collocation_epoch = -1
        self._current_epoch = 0
        self._normalize_physics = tc.normalize_physics_loss
        if tc.pinn_adaptive_collocation:
            if not self.is_pinn:
                raise ValueError('pinn_adaptive_collocation requires a PINN model')
            if not 0.0 < tc.pinn_adaptive_fraction < 1.0:
                raise ValueError('pinn_adaptive_fraction must be in (0, 1)')
            if tc.pinn_candidate_multiplier < 2:
                raise ValueError('pinn_candidate_multiplier must be >= 2')
            if tc.pinn_resample_every < 1:
                raise ValueError('pinn_resample_every must be >= 1')
        if tc.ode_window_epochs > 0:
            if not self.is_pure_node:
                raise ValueError('ode_window_epochs is supported only for pure Neural ODE')
            if tc.ode_window_strategy not in {'rotating', 'growing_prefix'}:
                raise ValueError("ode_window_strategy must be 'rotating' or 'growing_prefix'")
            if tc.ode_window_points < 2:
                raise ValueError('ode_window_points must be >= 2')
            if tc.ode_windows_per_epoch < 1:
                raise ValueError('ode_windows_per_epoch must be >= 1')
            if tc.ode_window_epochs >= tc.adam_epochs:
                raise ValueError('ode_window_epochs must be smaller than adam_epochs')
        self._loss_backtrack_enabled = (self.is_mode_b or self.is_pure_node) and tc.loss_backtrack_max_steps > 0 and (tc.loss_backtrack_trigger_ratio > 1.0) and (tc.loss_backtrack_end_epoch > tc.loss_backtrack_start_epoch)
        if tc.loss_backtrack_start_epoch < 0 or tc.loss_backtrack_end_epoch < 0:
            raise ValueError('loss backtrack epochs must be >= 0')
        if tc.loss_backtrack_max_steps < 0:
            raise ValueError('loss_backtrack_max_steps must be >= 0')
        if not 0.0 < tc.loss_backtrack_factor < 1.0:
            raise ValueError('loss_backtrack_factor must be in (0, 1)')
        if not 0.0 < tc.loss_backtrack_rejection_lr_factor <= 1.0:
            raise ValueError('loss_backtrack_rejection_lr_factor must be in (0, 1]')
        if not math.isfinite(tc.physics_ramp_power) or tc.physics_ramp_power <= 0.0:
            raise ValueError('physics_ramp_power must be finite and > 0')
        if tc.physics_ramp_schedule not in {'linear', 'cosine'}:
            raise ValueError("physics_ramp_schedule must be 'linear' or 'cosine'")

    def _physics_curriculum_factor(self, epoch: int) -> float:
        tc = self.config.training
        if tc.physics_warmup_epochs <= 0:
            return 1.0
        if epoch < tc.physics_warmup_epochs:
            return 0.0
        if tc.physics_ramp_epochs <= 0:
            return 1.0
        progress = min(1.0, (epoch - tc.physics_warmup_epochs + 1) / tc.physics_ramp_epochs)
        if tc.physics_ramp_schedule == 'cosine':
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(progress ** tc.physics_ramp_power)

    def train(self) -> TrainingResult:
        tc = self.config.training
        total_epochs = tc.adam_epochs + tc.lbfgs_epochs
        data = load_experiment_data(self.system, self.config)
        t_train = data['t_train'].to(self.device)
        y_train = data['y_train'].to(self.device)
        y0 = data['y0'].to(self.device)
        self.logger.save_config(self.config)
        start_time = time.time()
        best_loss = float('inf')
        best_epoch = -1
        best_state = None
        ema_state = None
        ema_selection_loss = None
        used_ema = False
        stopped_early = False
        stop_reason = None
        backtrack_reference_loss = None
        backtrack_reference_params = None
        loss_backtrack_events = 0
        loss_backtrack_reductions = 0
        loss_backtrack_rejections = 0
        target_loss_streak = 0
        plateau_reference_loss = float('inf')
        epochs_without_meaningful_improvement = 0
        if tc.target_loss < 0.0:
            raise ValueError('target_loss must be >= 0')
        if tc.target_loss_patience < 1:
            raise ValueError('target_loss_patience must be >= 1')
        if tc.early_stopping_start_epoch < 0:
            raise ValueError('early_stopping_start_epoch must be >= 0')
        if tc.early_stopping_patience < 0:
            raise ValueError('early_stopping_patience must be >= 0')
        if not 0.0 <= tc.early_stopping_min_relative_improvement < 1.0:
            raise ValueError('early_stopping_min_relative_improvement must be in [0, 1)')
        tc_init = self.config.training
        current_weights = {'data': tc_init.lambda_data}
        if 'physics' in self.loss_names:
            current_weights['physics'] = tc_init.lambda_physics
        if 'physics_data' in self.loss_names:
            current_weights['physics_data'] = tc_init.lambda_physics_data
        if 'gamma' in self.loss_names:
            current_weights['gamma'] = tc_init.lambda_gamma
        if 'correction' in self.loss_names:
            current_weights['correction'] = tc_init.lambda_correction
        if 'ic' in self.loss_names:
            current_weights['ic'] = tc_init.lambda_ic
        if 'reg' in self.loss_names:
            current_weights['reg'] = tc_init.lambda_reg
        selection_weights = current_weights.copy()
        if tc_init.physics_warmup_epochs > 0:
            for name in ('physics', 'physics_data', 'gamma'):
                if name in current_weights:
                    current_weights[name] = 0.0
        for epoch in range(total_epochs):
            self._current_epoch = epoch
            self.optimizer.update_learning_rate(epoch)
            if any((name in current_weights for name in ('physics', 'physics_data', 'gamma'))) and tc.physics_warmup_epochs > 0:
                physics_factor = self._physics_curriculum_factor(epoch)
                if 'physics' in current_weights:
                    current_weights['physics'] = tc.lambda_physics * physics_factor
                if 'physics_data' in current_weights:
                    current_weights['physics_data'] = tc.lambda_physics_data * physics_factor
                if 'gamma' in current_weights:
                    current_weights['gamma'] = tc.lambda_gamma * physics_factor
            if self.optimizer.should_switch(epoch):
                self.optimizer.switch_to_lbfgs()
                print(f'[Epoch {epoch}] Switching to L-BFGS optimizer')
            if self.optimizer.phase == 'lbfgs':
                loss_info = {}

                def closure():
                    self.optimizer.zero_grad()
                    components = self._compute_losses(t_train, y_train, y0)
                    loss, info = total_loss(components, current_weights)
                    loss.backward()
                    loss_info.update(info)
                    return loss
                self.optimizer.step(closure)
                components = self._compute_losses(t_train, y_train, y0)
                loss, loss_info = total_loss(components, current_weights)
                selection_loss, _ = self._selection_objective(components, selection_weights)
                loss_info['selection_total'] = float(selection_loss.item())
                if not torch.isfinite(loss) or not torch.isfinite(selection_loss):
                    stopped_early = True
                    stop_reason = f'non-finite L-BFGS loss at epoch {epoch}'
                    print(f'[STOP] {stop_reason}; restoring best epoch {best_epoch}')
                    break
                loss_value = float(selection_loss.item())
                if self._selection_eligible() and loss_value < best_loss:
                    best_loss = loss_value
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                self.optimizer.zero_grad()
                components = self._compute_losses(t_train, y_train, y0)
                loss, loss_info = total_loss(components, current_weights)
                selection_loss, _ = self._selection_objective(components, selection_weights)
                loss_info['selection_total'] = float(selection_loss.item())
                if not torch.isfinite(loss) or not torch.isfinite(selection_loss):
                    stopped_early = True
                    stop_reason = f'non-finite loss at epoch {epoch}'
                    print(f'[STOP] {stop_reason}; restoring best epoch {best_epoch}')
                    break
                backtrack_steps = 0
                backtrack_scale = 1.0
                backtrack_rejected = 0
                backtrack_trial_ratio = 1.0
                check_backtrack = self._loss_backtrack_enabled and tc.loss_backtrack_start_epoch < epoch <= tc.loss_backtrack_end_epoch and (backtrack_reference_loss is not None) and (backtrack_reference_params is not None)
                if check_backtrack:
                    initial_candidate = float(selection_loss.item())
                    backtrack_trial_ratio = initial_candidate / max(backtrack_reference_loss, 1e-30)
                    if not np.isfinite(backtrack_trial_ratio) or backtrack_trial_ratio > tc.loss_backtrack_trigger_ratio:
                        loss_backtrack_events += 1
                        accepted = False
                        for _ in range(tc.loss_backtrack_max_steps):
                            backtrack_steps += 1
                            loss_backtrack_reductions += 1
                            backtrack_scale *= tc.loss_backtrack_factor
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    before = backtrack_reference_params[name]
                                    param.copy_(before + (param - before) * tc.loss_backtrack_factor)
                                trial_components = self._compute_losses(t_train, y_train, y0)
                                trial_selection, _ = total_loss(trial_components, selection_weights)
                                trial_value = float(trial_selection.item())
                            if np.isfinite(trial_value) and trial_value <= backtrack_reference_loss * tc.loss_backtrack_trigger_ratio:
                                accepted = True
                                break
                        if not accepted:
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    param.copy_(backtrack_reference_params[name])
                            backtrack_scale = 0.0
                            backtrack_rejected = 1
                            loss_backtrack_rejections += 1
                            self.optimizer.scale_current_adam_learning_rates(tc.loss_backtrack_rejection_lr_factor)
                        self.optimizer.zero_grad()
                        components = self._compute_losses(t_train, y_train, y0)
                        loss, loss_info = total_loss(components, current_weights)
                        selection_loss, _ = self._selection_objective(components, selection_weights)
                        loss_info['selection_total'] = float(selection_loss.item())
                loss_info.update({'backtrack_steps': backtrack_steps, 'backtrack_scale': backtrack_scale, 'backtrack_rejected': backtrack_rejected, 'backtrack_trial_ratio': backtrack_trial_ratio})
                loss_value = selection_loss.item()
                if self._selection_eligible() and loss_value < best_loss:
                    best_loss = loss_value
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                if self._loss_backtrack_enabled and tc.loss_backtrack_start_epoch <= epoch < tc.loss_backtrack_end_epoch:
                    backtrack_reference_loss = loss_value
                    backtrack_reference_params = {name: param.detach().clone() for name, param in self.model.named_parameters()}
                loss.backward()
                self.optimizer.mask_inactive_adam_gradients(epoch)
                if tc.gradient_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip_norm)
                    if not torch.isfinite(grad_norm):
                        stopped_early = True
                        stop_reason = f'non-finite gradient at epoch {epoch}'
                        print(f'[STOP] {stop_reason}; restoring best epoch {best_epoch}')
                        break
                self.optimizer.step()
                update_stats = self.optimizer.last_update_stats
                loss_info.update({'update_network_norm': update_stats['network_norm'], 'update_physics_norm': update_stats['physics_norm'], 'update_network_scale': update_stats['network_scale'], 'update_physics_scale': update_stats['physics_scale'], 'update_clipped': update_stats['clipped']})
            if tc.target_loss > 0.0 and self._selection_eligible():
                if loss_value <= tc.target_loss:
                    target_loss_streak += 1
                else:
                    target_loss_streak = 0
                if target_loss_streak >= tc.target_loss_patience:
                    stopped_early = True
                    stop_reason = f'target loss {tc.target_loss:.3e} sustained for {tc.target_loss_patience} epochs'
                    print(f'[STOP] {stop_reason}; best epoch {best_epoch}')
                    break
            if self._selection_eligible() and tc.early_stopping_patience > 0:
                relative_delta = tc.early_stopping_min_relative_improvement
                meaningful_threshold = plateau_reference_loss * (1.0 - relative_delta)
                if loss_value < meaningful_threshold:
                    plateau_reference_loss = float(loss_value)
                    if epoch >= tc.early_stopping_start_epoch:
                        epochs_without_meaningful_improvement = 0
                elif epoch >= tc.early_stopping_start_epoch:
                    epochs_without_meaningful_improvement += 1
                if epoch >= tc.early_stopping_start_epoch and epochs_without_meaningful_improvement >= tc.early_stopping_patience:
                    stopped_early = True
                    stop_reason = f'selection objective did not improve by at least {100 * relative_delta:.3f}% for {tc.early_stopping_patience} epochs'
                    print(f'[STOP] {stop_reason}; restoring best epoch {best_epoch}')
                    break
            if tc.ema_decay > 0.0 and epoch >= tc.ema_start_epoch and (self.optimizer.phase == 'adam'):
                decay = min(max(float(tc.ema_decay), 0.0), 0.999999)
                current_state = self.model.state_dict()
                if ema_state is None:
                    ema_state = {key: value.detach().clone() for key, value in current_state.items()}
                else:
                    for key, value in current_state.items():
                        current = value.detach()
                        if torch.is_floating_point(current):
                            ema_state[key].mul_(decay).add_(current, alpha=1.0 - decay)
                        else:
                            ema_state[key] = current.clone()
            if epoch > 0:
                components_detached = {k: torch.tensor(loss_info.get(k, 0.0)) for k in self.loss_names if k in loss_info}
                current_weights = self.balancer.update(components_detached, self.model, epoch)
                if tc.physics_warmup_epochs > 0:
                    physics_factor = self._physics_curriculum_factor(epoch)
                    if 'physics' in current_weights:
                        current_weights['physics'] = tc.lambda_physics * physics_factor
                    if 'physics_data' in current_weights:
                        current_weights['physics_data'] = tc.lambda_physics_data * physics_factor
                    if 'gamma' in current_weights:
                        current_weights['gamma'] = tc.lambda_gamma * physics_factor
            if epoch % tc.log_every == 0:
                weight_info = {f'w_{k}': v for k, v in current_weights.items()}
                loss_info.update(weight_info)
                loss_info['selection_eligible'] = float(self._selection_eligible())
                if self._use_component and hasattr(self, '_last_comp_losses'):
                    loss_info.update(self._last_comp_losses)
                self.logger.log_losses(epoch, loss_info)
                if hasattr(self.model, 'get_params_dict'):
                    params = self.model.get_params_dict()
                    param_vals = {k: v.item() for k, v in params.items()}
                    if param_vals:
                        self.logger.log_parameters(epoch, param_vals)
                if epoch % (tc.log_every * 5) == 0:
                    param_str = ''
                    if hasattr(self.model, 'get_params_dict'):
                        params = self.model.get_params_dict()
                        param_str = ' | '.join((f'{k}={v.item():.6f}' for k, v in params.items()))
                    phase = self.optimizer.phase.upper()
                    print(f'[{phase}][Epoch {epoch:>6d}] total={loss_info.get('total', 0):.6e} data={loss_info.get('data', 0):.6e} w_data={current_weights.get('data', 0):.3f} w_phys={current_weights.get('physics', 0):.3f} {param_str}')
            if epoch > 0 and epoch % tc.checkpoint_every == 0:
                ckpt_path = self.logger.get_checkpoint_path(epoch)
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'balancer_state_dict': self.balancer.state_dict()}, ckpt_path)
        train_time = time.time() - start_time
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if tc.ema_use_for_final and ema_state is not None and (not self.is_pinn):
            self.model.load_state_dict(ema_state)
            with torch.no_grad():
                ema_components = self._compute_losses(t_train, y_train, y0)
                ema_objective, _ = self._selection_objective(ema_components, selection_weights)
                ema_selection_loss = float(ema_objective.item())
            ratio_limit = max(float(tc.ema_max_loss_ratio), 1.0)
            if np.isfinite(ema_selection_loss) and ema_selection_loss <= best_loss * ratio_limit:
                used_ema = True
            elif best_state is not None:
                self.model.load_state_dict(best_state)
        torch.save(self.model.state_dict(), str(self.logger.output_dir / 'model_best.pt'))
        torch.save(self.model.state_dict(), str(self.logger.output_dir / 'model_final.pt'))
        return TrainingResult(model=self.model, history=self.logger.history, param_history=self.logger.param_history, final_weights=current_weights, train_time=train_time, data=data, best_epoch=best_epoch, best_loss=best_loss, stopped_early=stopped_early, stop_reason=stop_reason, used_ema=used_ema, ema_selection_loss=ema_selection_loss, update_clip_events=self.optimizer.update_clip_events, loss_backtrack_events=loss_backtrack_events, loss_backtrack_reductions=loss_backtrack_reductions, loss_backtrack_rejections=loss_backtrack_rejections)

    def _compute_losses(self, t_train: torch.Tensor, y_train: torch.Tensor, y0: torch.Tensor) -> dict[str, torch.Tensor]:
        components = {}
        if self.is_pinn:
            components = self._pinn_losses(t_train, y_train, y0)
        elif self.is_mode_b:
            components = self._mode_b_losses(t_train, y_train, y0)
        else:
            components = self._ode_losses(t_train, y_train, y0)
        if 'reg' in self.loss_names:
            components['reg'] = regularization_loss(self.model)
        return components

    def _selection_objective(self, components: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
        selection_components = dict(components)
        if self._data_component_weights and self._last_unweighted_data_loss is not None:
            selection_components['data'] = self._last_unweighted_data_loss
        if self.config.training.pinn_adaptive_collocation and self._last_uniform_physics_loss is not None and ('physics' in selection_components):
            selection_components['physics'] = self._last_uniform_physics_loss
        return total_loss(selection_components, weights)

    def _selection_eligible(self) -> bool:
        return not (self.is_pure_node and self.config.training.ode_window_epochs > 0 and (self._current_epoch < self.config.training.ode_window_epochs))

    def _pinn_losses(self, t_train: torch.Tensor, y_train: torch.Tensor, y0: torch.Tensor) -> dict[str, torch.Tensor]:
        t_obs = t_train.requires_grad_(True)
        y_pred = self.model(t_obs)
        if self._use_component:
            comp_dict = data_loss_componentwise(y_pred, y_train, normalize=self._use_normalized, normalization_floor=self._data_normalization_floor, component_weights=self._data_component_weights, component_max_weight=self._data_component_max_weight)
            components = {'data': comp_dict['data']}
            self._last_unweighted_data_loss = comp_dict['data_unweighted']
            self._last_comp_losses = {k: v.item() for k, v in comp_dict.items() if k != 'data'}
        elif self._use_normalized:
            components = {'data': data_loss_normalized(y_pred, y_train, normalization_floor=self._data_normalization_floor)}
            self._last_unweighted_data_loss = components['data']
        else:
            components = {'data': data_loss(y_pred, y_train)}
            self._last_unweighted_data_loss = components['data']
        tc = self.config.training
        if tc.pinn_adaptive_collocation:
            t_values = self._get_adaptive_collocation_values()
        else:
            t_values = torch.linspace(self.config.system.t_span[0], self.config.system.t_span[1], tc.num_collocation, device=self.device).unsqueeze(1)
        t_coll = t_values.detach().clone().requires_grad_(True)
        y_coll = self.model(t_coll)
        dy_dt = self.model.compute_derivatives(t_coll, y_coll)
        params = self.model.get_params_dict()
        f_physics = self.system.rhs_parametric(t_coll, y_coll, params)
        components['physics'] = physics_residual_loss(dy_dt, f_physics, normalize=self._normalize_physics)
        if tc.pinn_adaptive_collocation:
            t_uniform = torch.linspace(self.config.system.t_span[0], self.config.system.t_span[1], tc.num_collocation, device=self.device).unsqueeze(1).requires_grad_(True)
            y_uniform = self.model(t_uniform)
            dy_uniform = self.model.compute_derivatives(t_uniform, y_uniform)
            f_uniform = self.system.rhs_parametric(t_uniform, y_uniform, self.model.get_params_dict())
            uniform_loss = physics_residual_loss(dy_uniform, f_uniform, normalize=self._normalize_physics)
            self._last_uniform_physics_loss = uniform_loss.detach()
            self._last_comp_losses['physics_uniform'] = float(uniform_loss.item())
        t0 = torch.tensor([[self.config.system.t_span[0]]], device=self.device)
        y_pred_0 = self.model(t0).squeeze(0)
        components['ic'] = initial_condition_loss(y_pred_0, y0, normalize=self._use_normalized, normalization_floor=self._data_normalization_floor)
        return components

    def _get_adaptive_collocation_values(self) -> torch.Tensor:
        tc = self.config.training
        needs_refresh = self._adaptive_collocation_values is None or (self._current_epoch % tc.pinn_resample_every == 0 and self._adaptive_collocation_epoch != self._current_epoch)
        if not needs_refresh:
            return self._adaptive_collocation_values
        total = int(tc.num_collocation)
        adaptive_count = max(1, min(total - 1, round(total * tc.pinn_adaptive_fraction)))
        uniform_count = total - adaptive_count
        candidate_count = max(total, total * int(tc.pinn_candidate_multiplier))
        candidates = torch.linspace(self.config.system.t_span[0], self.config.system.t_span[1], candidate_count, device=self.device).unsqueeze(1).requires_grad_(True)
        y_candidate = self.model(candidates)
        dy_candidate = self.model.compute_derivatives(candidates, y_candidate)
        f_candidate = self.system.rhs_parametric(candidates, y_candidate, self.model.get_params_dict())
        residual = dy_candidate - f_candidate
        if self._normalize_physics:
            scale = torch.clamp(f_candidate.detach().abs().max(dim=0).values, min=1e-06)
            residual = residual / scale.unsqueeze(0)
        scores = torch.mean(residual.detach() ** 2, dim=1)
        top_indices = torch.topk(scores, k=adaptive_count, largest=True).indices
        adaptive = candidates.detach()[top_indices]
        uniform = torch.linspace(self.config.system.t_span[0], self.config.system.t_span[1], uniform_count, device=self.device).unsqueeze(1)
        self._adaptive_collocation_values = torch.cat([uniform, adaptive], dim=0)
        self._adaptive_collocation_epoch = self._current_epoch
        return self._adaptive_collocation_values

    def _ode_losses(self, t_train: torch.Tensor, y_train: torch.Tensor, y0: torch.Tensor) -> dict[str, torch.Tensor]:
        t_flat = t_train.squeeze(1)
        sort_idx = torch.argsort(t_flat)
        t_sorted = t_flat[sort_idx]
        y_sorted = y_train[sort_idx]
        if self.is_pure_node and self.config.training.ode_window_epochs > 0 and (self._current_epoch < self.config.training.ode_window_epochs):
            return self._pure_node_window_losses(t_sorted, y_sorted)
        y_pred = self.model(y0, t_sorted)
        if self._use_component:
            comp_dict = data_loss_componentwise(y_pred, y_sorted, normalize=self._use_normalized, normalization_floor=self._data_normalization_floor, component_weights=self._data_component_weights, component_max_weight=self._data_component_max_weight)
            components = {'data': comp_dict['data']}
            self._last_unweighted_data_loss = comp_dict['data_unweighted']
            self._last_comp_losses = {k: v.item() for k, v in comp_dict.items() if k != 'data'}
        elif self._use_normalized:
            components = {'data': data_loss_normalized(y_pred, y_sorted, normalization_floor=self._data_normalization_floor)}
            self._last_unweighted_data_loss = components['data']
        else:
            components = {'data': data_loss(y_pred, y_sorted)}
            self._last_unweighted_data_loss = components['data']
        if self.is_mode_a and 'correction' in self.loss_names:
            correction = self.model.get_correction(t_sorted, y_pred)
            components['correction'] = torch.mean(correction ** 2)
        return components

    def _pure_node_window_losses(self, t_sorted: torch.Tensor, y_sorted: torch.Tensor) -> dict[str, torch.Tensor]:
        tc = self.config.training
        total_points = int(t_sorted.numel())
        minimum_points = min(int(tc.ode_window_points), total_points)
        if tc.ode_window_strategy == 'growing_prefix':
            progress = min(1.0, (self._current_epoch + 1) / max(int(tc.ode_window_epochs), 1))
            points = minimum_points + int(math.floor(progress * (total_points - minimum_points)))
            points = max(minimum_points, min(points, total_points))
            t_window = t_sorted[:points]
            y_obs = y_sorted[:points]
            y_pred = self.model(y_obs[0], t_window)
        else:
            points = minimum_points
            max_start = total_points - points
            possible_starts = list(range(max_start + 1))
            count = min(int(tc.ode_windows_per_epoch), len(possible_starts))
            first = self._current_epoch * count % len(possible_starts)
            starts = [possible_starts[(first + offset) % len(possible_starts)] for offset in range(count)]
            predictions = []
            observations = []
            for start in starts:
                stop = start + points
                t_window = t_sorted[start:stop]
                y_window = y_sorted[start:stop]
                predictions.append(self.model(y_window[0], t_window))
                observations.append(y_window)
            y_pred = torch.cat(predictions, dim=0)
            y_obs = torch.cat(observations, dim=0)
        if self._use_component:
            comp_dict = data_loss_componentwise(y_pred, y_obs, normalize=self._use_normalized, normalization_floor=self._data_normalization_floor, component_weights=self._data_component_weights, component_max_weight=self._data_component_max_weight)
            components = {'data': comp_dict['data']}
            self._last_unweighted_data_loss = comp_dict['data_unweighted']
            self._last_comp_losses = {key: value.item() for key, value in comp_dict.items() if key != 'data'}
        elif self._use_normalized:
            components = {'data': data_loss_normalized(y_pred, y_obs, normalization_floor=self._data_normalization_floor)}
            self._last_unweighted_data_loss = components['data']
        else:
            components = {'data': data_loss(y_pred, y_obs)}
            self._last_unweighted_data_loss = components['data']
        return components

    def _mode_b_losses(self, t_train: torch.Tensor, y_train: torch.Tensor, y0: torch.Tensor) -> dict[str, torch.Tensor]:
        t_flat = t_train.squeeze(1)
        sort_idx = torch.argsort(t_flat)
        t_sorted = t_flat[sort_idx]
        y_sorted = y_train[sort_idx]
        y_pred = self.model(y0, t_sorted)
        if self._use_component:
            comp_dict = data_loss_componentwise(y_pred, y_sorted, normalize=self._use_normalized, normalization_floor=self._data_normalization_floor, component_weights=self._data_component_weights, component_max_weight=self._data_component_max_weight)
            components = {'data': comp_dict['data']}
            self._last_unweighted_data_loss = comp_dict['data_unweighted']
            self._last_comp_losses = {k: v.item() for k, v in comp_dict.items() if k != 'data'}
        elif self._use_normalized:
            components = {'data': data_loss_normalized(y_pred, y_sorted, normalization_floor=self._data_normalization_floor)}
            self._last_unweighted_data_loss = components['data']
        else:
            components = {'data': data_loss(y_pred, y_sorted)}
            self._last_unweighted_data_loss = components['data']
        dy_dt_learned = self.model.get_learned_derivatives(t_sorted, y_pred)
        params = self.model.get_params_dict()
        dy_dt_known = self.system.rhs_parametric(t_sorted, y_pred, params)
        neighbourhood_samples = self.config.model.mode_b_neighborhood_samples
        neighbourhood_scale = self.config.model.mode_b_neighborhood_scale
        if neighbourhood_samples > 0 and neighbourhood_scale > 0:
            configured_scales = self.config.model.mode_b_state_scales
            if len(configured_scales) == y_pred.shape[-1]:
                state_scales = y_pred.new_tensor(configured_scales)
            else:
                state_scales = torch.clamp(y_sorted.detach().abs().max(dim=0).values, min=1.0)
            learned_parts = [dy_dt_learned]
            known_parts = [dy_dt_known]
            centres = y_pred.detach()
            time_index = torch.arange(1, centres.shape[0] + 1, device=centres.device, dtype=centres.dtype).unsqueeze(1)
            state_index = torch.arange(1, centres.shape[1] + 1, device=centres.device, dtype=centres.dtype).unsqueeze(0)
            for sample_index in range(neighbourhood_samples):
                direction = torch.sin(1.61803398875 * (sample_index + 1) * time_index * state_index)
                perturbation = direction * state_scales.unsqueeze(0) * neighbourhood_scale
                y_near = torch.clamp(centres + perturbation, min=0.0)
                learned_parts.append(self.model.get_learned_derivatives(t_sorted, y_near))
                known_parts.append(self.system.rhs_parametric(t_sorted, y_near, params))
            dy_dt_learned = torch.cat(learned_parts, dim=0)
            dy_dt_known = torch.cat(known_parts, dim=0)
        components['physics'] = physics_residual_loss(dy_dt_learned, dy_dt_known, normalize=self._normalize_physics)
        derivative_method = self.config.training.physics_data_derivative_method
        if derivative_method not in ('midpoint', 'central'):
            raise ValueError(f"physics_data_derivative_method must be 'midpoint' or 'central', got {derivative_method!r}")
        if 'physics_data' in self.loss_names and t_sorted.numel() > 1:
            if derivative_method == 'central' and t_sorted.numel() >= 3:
                dy_dt_observed = torch.gradient(y_sorted, spacing=(t_sorted,), dim=(0,), edge_order=2)[0]
                t_physics_data = t_sorted
                y_physics_data = y_sorted
            else:
                dt = t_sorted[1:] - t_sorted[:-1]
                dy_dt_observed = (y_sorted[1:] - y_sorted[:-1]) / dt.unsqueeze(1)
                t_physics_data = 0.5 * (t_sorted[1:] + t_sorted[:-1])
                y_physics_data = 0.5 * (y_sorted[1:] + y_sorted[:-1])
            dy_dt_parametric = self.system.rhs_parametric(t_physics_data, y_physics_data, params)
            components['physics_data'] = physics_residual_loss(dy_dt_observed, dy_dt_parametric, normalize=self._normalize_physics)
        if 'gamma' in self.loss_names and t_sorted.numel() > 1:
            if derivative_method == 'central' and t_sorted.numel() >= 3:
                dR_dt_observed = torch.gradient(y_sorted[:, 2], spacing=(t_sorted,), dim=(0,), edge_order=2)[0]
                infected_for_gamma = y_sorted[:, 1]
            else:
                dt = t_sorted[1:] - t_sorted[:-1]
                dR_dt_observed = (y_sorted[1:, 2] - y_sorted[:-1, 2]) / dt
                infected_for_gamma = 0.5 * (y_sorted[1:, 1] + y_sorted[:-1, 1])
            gamma_rhs = params['gamma'] * infected_for_gamma
            gamma_scale = torch.clamp(dR_dt_observed.abs().max(), min=0.001)
            components['gamma'] = torch.mean(((dR_dt_observed - gamma_rhs) / gamma_scale) ** 2)
        return components
