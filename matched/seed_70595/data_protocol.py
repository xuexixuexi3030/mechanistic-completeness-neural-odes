import hashlib
import protocol
import torch
from src.config import ExperimentConfig
from src.systems import SYSTEM_REGISTRY
from src.utils.data import generate_synthetic_data

def make_data(cfg):
    spec = cfg['system_spec']
    source = SYSTEM_REGISTRY[spec['name']]()
    observation_config = ExperimentConfig()
    observation_config.system.name = spec['name']
    observation_config.system.t_span = [0.0, spec['t_end']]
    observation_config.system.num_train_points = 10
    observation_config.system.num_test_points = spec['test_points']
    observation_config.system.split_strategy = 'uniform'
    observation_config.system.noise_mode = 'relative_state_std'
    observation_config.system.noise_std = cfg['noise']
    observation_config.system.preserve_initial_observation = True
    observation_config.training.seed = cfg['seed']
    data = generate_synthetic_data(source, observation_config)
    for key in ['t_train', 't_test', 't_full']:
        data[key] = data[key].reshape(-1)
    data['state_scale'] = data['y_train'].square().mean(0).sqrt().clamp_min(1.0)
    hasher = hashlib.sha256()
    for key in ['t_train', 'y_train', 'y0']:
        hasher.update(data[key].numpy().tobytes())
    data['observations_sha256'] = hasher.hexdigest()
    return (data, source)

def loss(model, view, network, cfg, t, y_obs, y0):
    from matched_models import vector_field
    prediction = model(y0, t)
    data_loss = ((prediction - y_obs) / network.state_scale).square().mean()
    guide = torch.zeros((), device=t.device)
    if cfg['guidance']:
        derivative = vector_field(model, cfg, t, prediction)
        target = view.rhs_known(t, prediction, model.get_params_dict())
        guide = ((derivative - target) / network.derivative_scale).square().mean()
    total = data_loss + cfg['training']['guide_lambda'] * guide
    return (total, data_loss, guide)
