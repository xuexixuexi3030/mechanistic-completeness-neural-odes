import argparse
import os
from pathlib import Path
from scripts.run_experiment import run_experiment

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 3407, 7777]

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=Path)
    group.add_argument('--all', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--include-continuations', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--override', nargs='*', default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    files = sorted((ROOT / 'configs').glob('*.yaml')) if args.all else [args.config.resolve()]
    seeds = ([args.seed] if args.seed is not None else SEEDS) if args.all else [args.seed if args.seed is not None else 42]
    os.chdir(ROOT)
    for config in files:
        if args.all and not args.include_continuations and 'refine_' in config.stem:
            continue
        for seed in seeds:
            name = f'{config.stem}_seed{seed}'
            overrides = [f'training.seed={seed}', f'output_dir={output}', f'experiment_name={name}']
            if args.device is not None:
                overrides.append(f'device={args.device}')
            overrides.extend(args.override)
            if args.dry_run:
                print(config.name, *overrides)
            else:
                run_experiment(str(config), overrides)

if __name__ == '__main__':
    main()
