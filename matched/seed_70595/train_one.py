import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
import protocol

def fingerprint(cfg):
    return protocol.digest(dict(config=cfg, code=protocol.code_digest()))

def evaluate(model, view, net, cfg, data, source, device):
    import numpy as np
    import torch
    from matched_models import vector_field, active_names
    t = data['t_full'].to(device)
    truth = data['y_full'].to(device)
    y0 = data['y0'].to(device)
    test = data['test_indices'].to(device)

    def rel(a, b):
        return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))
    with torch.no_grad():
        pred = model(y0, t)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Nonfinite evaluation trajectory')
        f = vector_field(model, cfg, t, pred)
        true_p = {k: pred.new_tensor(v) for k, v in source.true_params.items()}
        true_rhs = source.rhs_parametric(t, pred, true_p)
        phys = view.rhs_known(t, pred, model.get_params_dict())
        delta = f - phys
        fd = (pred[1:] - pred[:-1]) / (t[1:] - t[:-1]).unsqueeze(1)
        fm = source.rhs_parametric((t[1:] + t[:-1]) / 2, (pred[1:] + pred[:-1]) / 2, true_p)
        metrics = dict(full_relative_l2=rel(pred, truth), test_relative_l2_clean=rel(pred[test], truth[test]), test_scaled_rmse=float(((pred[test] - truth[test]) / net.state_scale).square().mean().sqrt()), physics_residual_true_param_norm=float((fd - fm).square().mean().sqrt()), vector_field_true_rhs_scaled_rms=float(((f - true_rhs) / net.derivative_scale).square().mean().sqrt()), learned_prior_residual_scaled_rms=float((delta / net.derivative_scale).square().mean().sqrt()), correction_to_prior_norm=float(delta.norm() / phys.norm().clamp_min(1e-12)), correction_prior_cosine=float((delta * phys).sum() / (delta.norm() * phys.norm()).clamp_min(1e-12)), negative_state_fraction=float((pred < 0).float().mean()))
        old_step = model.options['step_size']
        try:
            model.options['step_size'] = old_step / 2
            half = model(y0, t)
        finally:
            model.options['step_size'] = old_step
        metrics['solver_half_step_relative_gap'] = rel(half, pred)
        metrics['half_step_test_relative_l2_clean'] = rel(half[test], truth[test])
        parameters = []
        active = active_names(cfg)
        for name, value in model.get_params_dict().items():
            err = abs(float(value) - source.true_params[name]) / abs(source.true_params[name])
            parameters.append(dict(name=name, estimated=float(value), truth=source.true_params[name], active=name in active, relative_error=err if name in active else None))
        common_err = [r['relative_error'] for r in parameters if r['active'] and r['name'] in cfg['system_spec']['common_params']]
        all_err = [r['relative_error'] for r in parameters if r['active']]
        metrics['common_parameter_relative_error'] = float(np.mean(common_err)) if common_err else None
        metrics['active_parameter_relative_error'] = float(np.mean(all_err)) if all_err else None
        name = cfg['system_spec']['common_params'][0]
        raw = model.phys_params[name]
        backup = raw.clone()
        import torch.nn.functional as F
        raw.copy_(torch.log(torch.expm1(F.softplus(raw) * 1.01)))
        try:
            changed = model(y0, t)
        finally:
            raw.copy_(backup)
        metrics['inference_parameter_1pct_relative_change'] = rel(changed, pred)
        metrics['intervened_parameter'] = name
    return (metrics, parameters, pred.detach().cpu().numpy())

def gradient_diagnostic(model, view, net, cfg, data, device):
    import torch
    from data_protocol import loss
    args = [data[k].to(device) for k in ['t_train', 'y_train', 'y0']]
    total, ld, lg = loss(model, view, net, cfg, *args)
    params = list(model.phys_params.parameters())

    def norm(term):
        if not term.requires_grad:
            return 0.0
        gradients = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        return float(sum((g.square().sum() for g in gradients if g is not None), torch.zeros((), device=device)).sqrt())
    return dict(data_to_physics_grad_norm=norm(ld), guidance_to_physics_grad_norm=norm(lg), total_loss=float(total.detach()))

def train(cfg, output, device='cpu', purpose='scientific'):
    import torch
    import numpy as np
    import torchdiffeq
    from data_protocol import make_data, loss
    from matched_models import build, active_names
    protocol.validate(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_hash = fingerprint(cfg)
    result_path = output / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != run_hash or previous['purpose'] != purpose:
            raise ValueError(f'Refusing to overwrite incompatible completed run: {output}')
        return previous
    saved_config = output / 'config.json'
    if saved_config.exists() and json.loads(saved_config.read_text(encoding='utf-8')) != cfg:
        raise ValueError('Refusing to mix configurations in an existing run directory')
    lock = output / 'running.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), hostname=platform.node(), started=time.time()), handle)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        np.random.seed(cfg['seed'])
        data, source = make_data(cfg)
        model, view, net = build(cfg, data['state_scale'], device)
        optimizer = torch.optim.Adam([dict(params=list(net.parameters()), lr=cfg['training']['lr']), dict(params=list(model.phys_params.parameters()), lr=cfg['training']['phys_lr'])])
        args = [data[k].to(device) for k in ['t_train', 'y_train', 'y0']]
        protocol.dump(saved_config, cfg)
        environment = dict(python=platform.python_version(), torch=torch.__version__, torchdiffeq=torchdiffeq.__version__, numpy=np.__version__, device=device, device_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else platform.processor(), hostname=platform.node(), code_sha256=protocol.code_digest())
        protocol.dump(output / 'environment.json', environment)
        initial = gradient_diagnostic(model, view, net, cfg, data, device)
        protocol.dump(output / 'initial_gradient_paths.json', initial)
        with (output / 'observations.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            d = source.state_dim
            writer.writerow(['time', 'is_train'] + [f'clean_{j}' for j in range(d)] + [f'observed_{j}' for j in range(d)])
            train_idx = set(data['train_indices'].tolist())
            for i, t in enumerate(data['t_full'].tolist()):
                writer.writerow([t, int(i in train_idx)] + data['y_full'][i].tolist() + data['y_observed_full'][i].tolist())
        checkpoint = output / 'checkpoint.pt'
        start_epoch, previous_seconds = (0, 0.0)
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            if saved['fingerprint'] != run_hash or saved['purpose'] != purpose:
                raise ValueError('Checkpoint/configuration/code fingerprint mismatch')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            start_epoch, previous_seconds = (saved['epoch'], saved['training_seconds'])
        history_path = output / 'training_history.csv'
        rows = []
        if history_path.exists() and start_epoch:
            with history_path.open(encoding='utf-8') as handle:
                rows = [r for r in csv.DictReader(handle) if int(r['epoch']) <= start_epoch]
        sync = (lambda: torch.cuda.synchronize(device)) if device.startswith('cuda') else lambda: None
        sync()
        started = time.perf_counter()
        steps = cfg['training']['epochs']
        with history_path.open('w', newline='', encoding='utf-8') as handle:
            columns = ['epoch', 'total', 'data', 'guidance', 'training_seconds']
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            for epoch in range(start_epoch + 1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                total, ld, lg = loss(model, view, net, cfg, *args)
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
                total.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not all((torch.isfinite(g).all() for g in grads)):
                    raise FloatingPointError(f'Nonfinite gradient at epoch {epoch}')
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if epoch == 1 or epoch % cfg['training']['log_every'] == 0 or epoch == steps:
                    sync()
                    seconds = previous_seconds + time.perf_counter() - started
                    row = dict(epoch=epoch, total=float(total.detach()), data=float(ld.detach()), guidance=float(lg.detach()), training_seconds=seconds)
                    writer.writerow(row)
                    handle.flush()
                    print(json.dumps(dict(task=protocol.task_id(cfg), **row)), flush=True)
                if epoch % cfg['training']['checkpoint_every'] == 0 or epoch == steps:
                    sync()
                    tmp = output / 'checkpoint.tmp.pt'
                    torch.save(dict(fingerprint=run_hash, purpose=purpose, model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, training_seconds=previous_seconds + time.perf_counter() - started), tmp)
                    tmp.replace(checkpoint)
        sync()
        training_seconds = previous_seconds + time.perf_counter() - started
        diagnostics = gradient_diagnostic(model, view, net, cfg, data, device)
        metrics, parameters, prediction = evaluate(model, view, net, cfg, data, source, device)
        np.savetxt(output / 'predictions.csv', np.column_stack([data['t_full'].numpy(), prediction]), delimiter=',', header='time,' + ','.join((f'prediction_{i}' for i in range(source.state_dim))), comments='')
        protocol.dump(output / 'parameters.json', parameters)
        protocol.dump(output / 'final_gradient_paths.json', diagnostics)
        result = dict(status='completed', purpose=purpose, task_id=protocol.task_id(cfg), fingerprint=run_hash, config=cfg, metrics=metrics, training_seconds=training_seconds, environment=environment, observations_sha256=data['observations_sha256'], neural_parameter_count=sum((p.numel() for p in net.parameters())), registered_physics_count=len(model.phys_params), active_physics_parameters=active_names(cfg), interpretation='MC and MA are matched-equivalent; do not pool labels as independent seeds')
        protocol.dump(result_path, result)
        failure = output / 'failure.json'
        if failure.exists():
            failure.unlink()
        return result
    except BaseException as error:
        protocol.dump(output / 'failure.json', dict(status='failed_or_interrupted', error_type=type(error).__name__, error=str(error), fingerprint=run_hash, config=cfg, purpose=purpose))
        raise
    finally:
        lock.unlink(missing_ok=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--purpose', choices=['scientific', 'smoke', 'benchmark'], default='scientific')
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding='utf-8')), args.output, args.device, args.purpose)
