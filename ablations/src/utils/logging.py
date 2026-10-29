import csv
import json
import os
from pathlib import Path

class ExperimentLogger:

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        self._history_path = self.output_dir / 'training_history.csv'
        self._history_file = None
        self._history_writer = None
        self._history_fields = None
        self._params_path = self.output_dir / 'parameters.csv'
        self._params_file = None
        self._params_writer = None
        self._params_fields = None
        self.history: list[dict] = []
        self.param_history: list[dict] = []

    def _init_history_writer(self, fields: list[str]):
        self._history_fields = ['epoch'] + fields
        self._history_file = open(self._history_path, 'w', newline='')
        self._history_writer = csv.DictWriter(self._history_file, fieldnames=self._history_fields)
        self._history_writer.writeheader()

    def _init_params_writer(self, param_names: list[str]):
        self._params_fields = ['epoch'] + param_names
        self._params_file = open(self._params_path, 'w', newline='')
        self._params_writer = csv.DictWriter(self._params_file, fieldnames=self._params_fields)
        self._params_writer.writeheader()

    def log_losses(self, epoch: int, losses: dict[str, float]):
        if self._history_writer is None:
            self._init_history_writer(sorted(losses.keys()))
        row = {'epoch': epoch, **losses}
        self._history_writer.writerow(row)
        self._history_file.flush()
        self.history.append(row)

    def log_parameters(self, epoch: int, params: dict[str, float]):
        if self._params_writer is None:
            self._init_params_writer(sorted(params.keys()))
        row = {'epoch': epoch, **params}
        self._params_writer.writerow(row)
        self._params_file.flush()
        self.param_history.append(row)

    def log_weights(self, epoch: int, weights: dict[str, float]):
        pass

    def save_metrics(self, metrics: dict):
        path = self.output_dir / 'metrics.json'
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, allow_nan=False)

    def save_config(self, config):
        import yaml
        from dataclasses import asdict
        path = self.output_dir / 'config.yaml'
        with open(path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)

    def get_checkpoint_path(self, step: int) -> str:
        return str(self.output_dir / 'checkpoints' / f'step_{step}.pt')

    def get_figure_path(self, name: str) -> str:
        return str(self.output_dir / 'figures' / name)

    def close(self):
        if self._history_file:
            self._history_file.close()
        if self._params_file:
            self._params_file.close()
