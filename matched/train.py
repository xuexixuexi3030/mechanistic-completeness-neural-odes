import argparse
import copy
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[1031, 70595, 250333], required=True)
    parser.add_argument('--system', choices=['gene', 'eco'])
    parser.add_argument('--model', choices=['MC', 'PG', 'MA'])
    parser.add_argument('--completeness', choices=['partial', 'full'])
    parser.add_argument('--guidance', type=int, choices=[0, 1])
    parser.add_argument('--noise', type=float, choices=[0.0, 0.01, 0.06])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / f'seed_{args.seed}'))
    protocol = importlib.import_module('protocol')
    training = importlib.import_module('train_one')
    for cfg in protocol.matrix(stage='full'):
        if any(getattr(args, key) is not None and cfg[key] != getattr(args, key) for key in ['system', 'model', 'completeness', 'guidance', 'noise']):
            continue
        cfg = copy.deepcopy(cfg)
        name = protocol.task_id(cfg)
        if args.dry_run:
            print(name)
        else:
            training.train(cfg, args.output.resolve() / name, device=args.device)

if __name__ == '__main__':
    main()
