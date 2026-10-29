import math
import torch
import torch.optim as optim

class HybridOptimizer:

    def __init__(self, params, adam_lr: float=0.001, adam_beta1: float=0.9, adam_beta2: float=0.999, adam_eps: float=1e-08, adam_amsgrad: bool=False, adam_epochs: int=80000, adam_decay_start: int=0, adam_final_lr: float=0.0, lbfgs_lr: float=1.0, lbfgs_max_iter: int=20, lbfgs_epochs: int=20000, phys_params=None, phys_lr: float=0.0, phys_param_names: dict[int, str] | None=None, phys_lr_overrides: dict[str, float] | None=None, parameter_refine_start: int=0, parameter_refine_network_lr_factor: float=1.0, parameter_refine_physics_lr_factor: float=1.0, lr_stages: list[dict] | None=None, lr_stage_transition_epochs: int=0, max_network_update_norm: float=0.0, max_physics_update_norm: float=0.0, alternating_update_start: int=0, alternating_network_steps: int=0, alternating_physics_steps: int=0):
        self.adam_epochs = adam_epochs
        self.adam_lr = adam_lr
        self.adam_decay_start = adam_decay_start
        self.adam_final_lr = adam_final_lr
        self.lbfgs_epochs = lbfgs_epochs
        self.total_epochs = adam_epochs + lbfgs_epochs
        self.parameter_refine_start = parameter_refine_start
        self.parameter_refine_network_lr_factor = parameter_refine_network_lr_factor
        self.parameter_refine_physics_lr_factor = parameter_refine_physics_lr_factor
        self.lr_stages = self._normalize_lr_stages(lr_stages or [])
        self.lr_stage_transition_epochs = int(lr_stage_transition_epochs)
        if self.lr_stage_transition_epochs < 0:
            raise ValueError('lr_stage_transition_epochs must be >= 0')
        self.max_network_update_norm = float(max_network_update_norm)
        self.max_physics_update_norm = float(max_physics_update_norm)
        for name, value in (('max_network_update_norm', self.max_network_update_norm), ('max_physics_update_norm', self.max_physics_update_norm)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and >= 0')
        self._last_update_stats = {'network_norm': 0.0, 'physics_norm': 0.0, 'network_scale': 1.0, 'physics_scale': 1.0, 'clipped': 0}
        self._update_clip_events = 0
        if not 0.0 <= adam_beta1 < 1.0 or not 0.0 <= adam_beta2 < 1.0:
            raise ValueError('Adam beta values must be in [0, 1)')
        if not math.isfinite(adam_eps) or adam_eps <= 0.0:
            raise ValueError('adam_eps must be finite and > 0')
        self.alternating_update_start = int(alternating_update_start)
        self.alternating_network_steps = int(alternating_network_steps)
        self.alternating_physics_steps = int(alternating_physics_steps)
        if self.alternating_update_start < 0:
            raise ValueError('alternating_update_start must be >= 0')
        if self.alternating_network_steps < 0 or self.alternating_physics_steps < 0:
            raise ValueError('alternating step counts must be >= 0')
        self._alternating_enabled = self.alternating_network_steps > 0 and self.alternating_physics_steps > 0
        all_input_params = list(params)
        phys_params = list(phys_params) if phys_params is not None else []
        phys_lr_overrides = phys_lr_overrides or {}
        phys_param_names = phys_param_names or {}
        if phys_params and (phys_lr > 0 or phys_lr_overrides):
            phys_param_set = {id(p) for p in phys_params}
            net_params = [p for p in all_input_params if id(p) not in phys_param_set]
            self.param_groups = [{'params': net_params, 'lr': adam_lr, 'group_type': 'network'}]
            default_phys_lr = phys_lr if phys_lr > 0 else adam_lr
            has_named_stage_factors = any((stage['physics_factor_overrides'] for stage in self.lr_stages))
            if has_named_stage_factors:
                for param in phys_params:
                    name = phys_param_names.get(id(param), '')
                    lr = float(phys_lr_overrides.get(name, default_phys_lr))
                    self.param_groups.append({'params': [param], 'lr': lr, 'group_type': 'physics', 'param_name': name})
            else:
                grouped_phys: dict[float, list] = {}
                for param in phys_params:
                    name = phys_param_names.get(id(param), '')
                    lr = float(phys_lr_overrides.get(name, default_phys_lr))
                    grouped_phys.setdefault(lr, []).append(param)
                self.param_groups.extend(({'params': group, 'lr': lr, 'group_type': 'physics'} for lr, group in grouped_phys.items()))
            self.all_params = net_params + phys_params
        else:
            self.all_params = all_input_params
            self.param_groups = [{'params': self.all_params, 'lr': adam_lr, 'group_type': 'network'}]
        if self._alternating_enabled and (not any((group.get('group_type') == 'physics' for group in self.param_groups))):
            raise ValueError('Alternating updates require a physics parameter group')
        self.adam = optim.Adam(self.param_groups, betas=(float(adam_beta1), float(adam_beta2)), eps=float(adam_eps), amsgrad=bool(adam_amsgrad))
        self._adam_base_lrs = [float(group['lr']) for group in self.adam.param_groups]
        self._lbfgs = None
        self._lbfgs_lr = lbfgs_lr
        self._lbfgs_max_iter = lbfgs_max_iter
        self._current_epoch = 0
        self._phase = 'adam'

    @staticmethod
    def _normalize_lr_stages(stages: list[dict]) -> list[dict]:
        normalized = []
        seen_starts = set()
        for raw in stages:
            if not isinstance(raw, dict):
                raise TypeError('Each lr_stages entry must be a mapping')
            start = int(raw.get('start', -1))
            if start < 0:
                raise ValueError('lr_stages.start must be >= 0')
            if start in seen_starts:
                raise ValueError(f'Duplicate lr_stages.start: {start}')
            seen_starts.add(start)
            network_factor = float(raw.get('network_factor', 1.0))
            physics_factor = float(raw.get('physics_factor', 1.0))
            overrides = {str(name): float(value) for name, value in (raw.get('physics_factor_overrides') or {}).items()}
            factors = [network_factor, physics_factor, *overrides.values()]
            if any((not math.isfinite(value) or value < 0.0 for value in factors)):
                raise ValueError('lr_stages factors must be finite and >= 0')
            normalized.append({'start': start, 'network_factor': network_factor, 'physics_factor': physics_factor, 'physics_factor_overrides': overrides})
        return sorted(normalized, key=lambda stage: stage['start'])

    @property
    def phase(self) -> str:
        return self._phase

    def _ensure_lbfgs(self):
        if self._lbfgs is None:
            self._lbfgs = optim.LBFGS(self.all_params, lr=self._lbfgs_lr, max_iter=self._lbfgs_max_iter, history_size=50, line_search_fn='strong_wolfe')

    def should_switch(self, epoch: int) -> bool:
        return epoch >= self.adam_epochs and self._phase == 'adam'

    def update_learning_rate(self, epoch: int) -> None:
        if self._phase != 'adam':
            return
        factor = 1.0
        if self.adam_decay_start > 0 and epoch >= self.adam_decay_start and (self.adam_epochs > self.adam_decay_start):
            progress = min(1.0, (epoch - self.adam_decay_start) / max(self.adam_epochs - self.adam_decay_start - 1, 1))
            final_factor = max(self.adam_final_lr, 0.0) / max(self.adam_lr, 1e-16)
            factor = final_factor + (1.0 - final_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        active_stage = None
        active_stage_index = -1
        for index, stage in enumerate(self.lr_stages):
            if epoch >= stage['start']:
                active_stage = stage
                active_stage_index = index
            else:
                break
        refine_active = not self.lr_stages and self.parameter_refine_start > 0 and (epoch >= self.parameter_refine_start)
        for group, base_lr in zip(self.adam.param_groups, self._adam_base_lrs):
            refine_factor = 1.0
            if active_stage is not None:

                def stage_factor(stage: dict | None) -> float:
                    if stage is None:
                        return 1.0
                    if group.get('group_type') == 'physics':
                        return stage['physics_factor_overrides'].get(group.get('param_name', ''), stage['physics_factor'])
                    return stage['network_factor']
                target_factor = stage_factor(active_stage)
                transition = self.lr_stage_transition_epochs
                elapsed = epoch - active_stage['start']
                if transition > 0 and elapsed < transition:
                    previous_stage = self.lr_stages[active_stage_index - 1] if active_stage_index > 0 else None
                    previous_factor = stage_factor(previous_stage)
                    progress = max(0.0, min(1.0, elapsed / transition))
                    blend = 0.5 * (1.0 - math.cos(math.pi * progress))
                    refine_factor = previous_factor + (target_factor - previous_factor) * blend
                else:
                    refine_factor = target_factor
            elif refine_active:
                if group.get('group_type') == 'physics':
                    refine_factor = self.parameter_refine_physics_lr_factor
                else:
                    refine_factor = self.parameter_refine_network_lr_factor
            group['lr'] = base_lr * factor * refine_factor

    def switch_to_lbfgs(self):
        self._phase = 'lbfgs'
        self._ensure_lbfgs()

    def mask_inactive_adam_gradients(self, epoch: int) -> str:
        if self._phase != 'adam' or not self._alternating_enabled or epoch < self.alternating_update_start:
            return 'all'
        cycle_length = self.alternating_network_steps + self.alternating_physics_steps
        position = (epoch - self.alternating_update_start) % cycle_length
        active_type = 'network' if position < self.alternating_network_steps else 'physics'
        for group in self.adam.param_groups:
            if group.get('group_type') == active_type:
                continue
            for param in group['params']:
                param.grad = None
        return active_type

    def zero_grad(self):
        if self._phase == 'adam':
            self.adam.zero_grad()
        else:
            self._ensure_lbfgs()
            self._lbfgs.zero_grad()

    def step(self, closure=None):
        self._current_epoch += 1
        if self._phase == 'adam':
            limits = {'network': self.max_network_update_norm, 'physics': self.max_physics_update_norm}
            snapshots: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {'network': [], 'physics': []}
            if any((limit > 0.0 for limit in limits.values())):
                for group in self.adam.param_groups:
                    group_type = group.get('group_type', 'network')
                    if limits.get(group_type, 0.0) <= 0.0:
                        continue
                    for param in group['params']:
                        if param.grad is not None:
                            snapshots[group_type].append((param, param.detach().clone()))
            self.adam.step()
            stats = {'network_norm': 0.0, 'physics_norm': 0.0, 'network_scale': 1.0, 'physics_scale': 1.0, 'clipped': 0}
            with torch.no_grad():
                for group_type, items in snapshots.items():
                    if not items:
                        continue
                    squared_norm = torch.zeros((), device=items[0][0].device)
                    for param, before in items:
                        squared_norm.add_(torch.sum((param - before) ** 2))
                    update_norm = float(torch.sqrt(squared_norm).item())
                    limit = limits[group_type]
                    scale = min(1.0, limit / max(update_norm, 1e-30))
                    stats[f'{group_type}_norm'] = update_norm
                    stats[f'{group_type}_scale'] = scale
                    if scale < 1.0:
                        for param, before in items:
                            param.copy_(before + (param - before) * scale)
                        stats['clipped'] = 1
                if stats['clipped']:
                    self._update_clip_events += 1
            self._last_update_stats = stats
        else:
            self._ensure_lbfgs()
            if closure is None:
                raise ValueError('L-BFGS requires a closure function')
            self._lbfgs.step(closure)

    def scale_current_adam_learning_rates(self, factor: float) -> None:
        if self._phase != 'adam':
            return
        factor = float(factor)
        if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
            raise ValueError('Adam LR scale factor must be finite and in (0, 1]')
        for group in self.adam.param_groups:
            group['lr'] *= factor

    @property
    def last_update_stats(self) -> dict[str, float]:
        return dict(self._last_update_stats)

    @property
    def update_clip_events(self) -> int:
        return self._update_clip_events

    def state_dict(self) -> dict:
        state = {'phase': self._phase, 'current_epoch': self._current_epoch, 'adam': self.adam.state_dict(), 'update_clip_events': self._update_clip_events}
        if self._lbfgs is not None:
            state['lbfgs'] = self._lbfgs.state_dict()
        return state

    def load_state_dict(self, state: dict):
        self._phase = state['phase']
        self._current_epoch = state['current_epoch']
        self.adam.load_state_dict(state['adam'])
        self._update_clip_events = int(state.get('update_clip_events', 0))
        if 'lbfgs' in state:
            self._ensure_lbfgs()
            self._lbfgs.load_state_dict(state['lbfgs'])
