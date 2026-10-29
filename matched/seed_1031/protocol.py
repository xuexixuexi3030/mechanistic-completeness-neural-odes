from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
VERSION = 'matched_controls'
SEEDS = [1031]
NOISES = [0.0, 0.01, 0.06]
MODELS = ['MC', 'PG', 'MA']
SYSTEMS = {'gene': dict(name='gene_regulatory', t_end=100.0, test_points=100, hidden_units=32, common_params=['k_on', 'k_off'], param_init=dict(k_on=0.025, k_off=0.0125, alpha1=0.35, alpha2=0.2, alpha3=0.15, K_d=4.0, n_hill=1.5)), 'eco': dict(name='eco_epidemic', t_end=200.0, test_points=120, hidden_units=48, common_params=['r', 'K', 'gamma'], param_init=dict(r=0.4, K=800.0, beta=0.2, gamma=0.15, mu=0.02))}
TRAINING = dict(epochs=20000, lr=0.0003, phys_lr=0.0003, gradient_clip=1.0, step_size=1.0, hidden_layers=3, guide_lambda=0.1, log_every=100, checkpoint_every=1000, diagnostic_epochs=[1000, 4000, 8000, 16000, 20000], selection='last_epoch', derivative_scale_time=10.0)

def make_config(system, model, completeness, guidance, seed, noise, epochs=None):
    cfg = dict(version=VERSION, system=system, model=model, completeness=completeness, guidance=guidance, seed=seed, noise=noise, system_spec=SYSTEMS[system], training=dict(TRAINING))
    if epochs is not None:
        cfg['training']['epochs'] = int(epochs)
    return cfg

def task_id(cfg):
    return f'{cfg['system']}_{cfg['model']}_{cfg['completeness']}_g{cfg['guidance']}_noise{round(cfg['noise'] * 100):02d}_seed{cfg['seed']}'

def matrix(stage='pilot', epochs=None):
    seeds = SEEDS[:1] if stage == 'pilot' else SEEDS
    noises = [0.0] if stage == 'pilot' else NOISES
    return [make_config(s, m, c, g, seed, noise, epochs) for noise in noises for seed in seeds for s in SYSTEMS for c in ['partial', 'full'] for g in [0, 1] for m in MODELS]

def validate(cfg):
    assert cfg['version'] == VERSION
    assert cfg['system'] in SYSTEMS and cfg['model'] in MODELS
    assert cfg['completeness'] in ['partial', 'full']
    assert cfg['guidance'] in [0, 1]
    assert cfg['noise'] in NOISES and cfg['seed'] in SEEDS
    assert cfg['system_spec'] == SYSTEMS[cfg['system']]
    assert cfg['training']['epochs'] > 0
    assert cfg['training']['step_size'] > 0

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)
