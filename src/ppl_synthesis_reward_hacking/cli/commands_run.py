from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Run experiments")
    run_sub = parser.add_subparsers(dest="run_command", required=True)

    baseline = run_sub.add_parser("baseline", help="Run baseline experiment")
    _add_common_run_args(baseline)
    baseline.set_defaults(func=_run_baseline)

    attack = run_sub.add_parser("attack", help="Run attack experiment")
    _add_common_run_args(attack)
    attack.set_defaults(func=_run_attack)

    sstan = run_sub.add_parser("sstan", help="Run SStan-checked experiment")
    _add_common_run_args(sstan)
    sstan.set_defaults(func=_run_sstan)

    emergent = run_sub.add_parser(
        "emergent",
        help="Run emergent reward hacking experiment",
    )
    _add_emergent_args(emergent)
    emergent.set_defaults(func=_run_emergent)


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--backend",
        required=False,
        help="Backend: toy/stan/sstan/pymc/pymc_safe",
    )
    parser.add_argument("--set", action="append", default=[], help="Override config key=value")
    parser.add_argument("--seed", type=int, default=None, help="Override run seed")


def _add_emergent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--set", action="append", default=[], help="Override config key=value")
    parser.add_argument("--seed", type=int, default=None, help="Override run seed")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of optimization steps",
    )


def _run_baseline(args: argparse.Namespace) -> int:
    from ppl_synthesis_reward_hacking.experiments.lh_protocol import run_lh_protocol

    run_ids = run_lh_protocol(
        config_path=args.config,
        backend_override=args.backend,
        overrides=args.set,
        seed_override=args.seed,
    )
    for run_id in run_ids:
        print(run_id)
    return 0


def _run_attack(args: argparse.Namespace) -> int:
    from ppl_synthesis_reward_hacking.experiments.lh_protocol import run_lh_protocol

    run_ids = run_lh_protocol(
        config_path=args.config,
        backend_override=args.backend,
        overrides=args.set,
        seed_override=args.seed,
    )
    for run_id in run_ids:
        print(run_id)
    return 0


def _run_sstan(args: argparse.Namespace) -> int:
    from ppl_synthesis_reward_hacking.experiments.lh_protocol import run_lh_protocol

    run_ids = run_lh_protocol(
        config_path=args.config,
        backend_override=args.backend,
        overrides=args.set,
        seed_override=args.seed,
    )
    for run_id in run_ids:
        print(run_id)
    return 0


def _run_emergent(args: argparse.Namespace) -> int:
    from ppl_synthesis_reward_hacking.experiments.emergent_hacking import run_emergent_hacking

    run_id = run_emergent_hacking(
        config_path=args.config,
        overrides=args.set,
        seed_override=args.seed,
        steps_override=args.steps,
    )
    print(run_id)
    return 0
