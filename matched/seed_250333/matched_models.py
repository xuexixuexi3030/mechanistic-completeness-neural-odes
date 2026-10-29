import protocol
import torch
from torch import nn
from src.models.mode_a import ModeAModel
from src.models.mode_b import ModeBModel
from src.models.mode_c import ModeCModel
from src.models.networks import SmallFNN

def rhs(system, completeness, t, y, p):
    if system == 'gene':
        prom, tf, complex_ = y.unbind(-1)
        binding = p['k_on'] * prom * tf - p['k_off'] * complex_
        result = torch.stack([-binding, -binding, binding], -1)
        if completeness == 'full':

            def hill(x, alpha):
                positive_x = x.clamp_min(1e-12)
                h = torch.sigmoid(p['n_hill'] * (positive_x.log() - p['K_d'].log()))
                return alpha * h * (x > 0).to(y.dtype)
            result = result + torch.stack([hill(complex_, p['alpha1']), hill(prom * tf, p['alpha2']), hill(complex_, p['alpha3'])], -1)
        return result
    S, I, R, N = y.unbind(-1)
    growth = p['r'] * S * (1 - N / p['K'])
    recovery = p['gamma'] * I
    result = torch.stack([growth, -recovery, recovery, growth], -1)
    if completeness == 'full':
        infection = p['beta'] * S * I / N.clamp_min(1e-08)
        mortality = p['mu'] * I
        result = result + torch.stack([-infection, infection - mortality, torch.zeros_like(S), -mortality], -1)
    return result

class PriorView:

    def __init__(self, cfg):
        self.system = cfg['system']
        self.completeness = cfg['completeness']
        self.state_dim = 3 if self.system == 'gene' else 4
        self.unknown_latent_dim = self.state_dim

    def rhs_known(self, t, y, p):
        return rhs(self.system, self.completeness, t, y, p)
    rhs_parametric = rhs_known

    def map_unknown_correction(self, latent, state=None, constrained=False):
        if constrained:
            raise ValueError('Native structural constraints are excluded from the matched-prior suite')
        return latent

class SharedNetwork(nn.Module):

    def __init__(self, cfg, state_scale):
        super().__init__()
        d = len(state_scale)
        self.register_buffer('state_scale', state_scale.detach().clone())
        self.register_buffer('derivative_scale', state_scale / cfg['training']['derivative_scale_time'])
        self.register_buffer('time_half', torch.tensor(cfg['system_spec']['t_end'] / 2))
        self.net = SmallFNN(d + 1, d, hidden_layers=cfg['training']['hidden_layers'], hidden_units=cfg['system_spec']['hidden_units'], activation='tanh', init_strategy='small_output')

    def forward(self, inputs):
        normalized = torch.cat([(inputs[..., :1] - self.time_half) / self.time_half, inputs[..., 1:] / self.state_scale], -1)
        return self.net(normalized) * self.derivative_scale

class TimeStateField(nn.Module):

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, t, y):
        t_col = t.reshape(1) if y.ndim == 1 else t.reshape(-1, 1).expand(y.shape[0], 1)
        return self.network(torch.cat([t_col, y], -1))

def build(cfg, state_scale, device='cpu'):
    protocol.validate(cfg)
    view = PriorView(cfg)
    spec = cfg['system_spec']
    common = dict(param_names=list(spec['param_init']), param_init_values=spec['param_init'], solver='rk4', step_size=cfg['training']['step_size'], rtol=0.0001, atol=1e-06, use_adjoint=False, node_hidden_layers=cfg['training']['hidden_layers'], node_hidden_units=spec['hidden_units'])
    torch.manual_seed(cfg['seed'])
    if cfg['model'] == 'MC':
        model = ModeAModel(view, state_scales=state_scale.tolist(), normalize_inputs=False, constrain_unknown=False, **common)
    elif cfg['model'] == 'MA':
        model = ModeCModel(view, correction_scale=1.0, bound_correction=False, **common)
    else:
        model = ModeBModel(view.state_dim, **common)
    torch.manual_seed(cfg['seed'])
    network = SharedNetwork(cfg, state_scale)
    if cfg['model'] == 'PG':
        model.func = TimeStateField(network)
    else:
        model.dynamics.correction_net = network
    return (model.to(device), view, network.to(device))

def vector_field(model, cfg, t, y):
    if cfg['model'] != 'MA' or y.ndim == 1:
        return (model.func if cfg['model'] == 'PG' else model.dynamics)(t, y)
    return torch.stack([model.dynamics(ti, yi) for ti, yi in zip(t.reshape(-1), y)])

def active_names(cfg):
    if cfg['model'] == 'PG' and cfg['guidance'] == 0:
        return []
    return list(cfg['system_spec']['param_init']) if cfg['completeness'] == 'full' else cfg['system_spec']['common_params']
