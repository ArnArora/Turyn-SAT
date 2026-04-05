"""
Rollout evaluation for Turyn policy models at partial-prefix cutoffs.

For each requested completion percentage, this script runs multiple model rollouts
starting from the seeded initial prefix [(1,), (1,), (1,), (1,)], then SAT-checks
the generated partial sequence and reports successful/unsuccessful counts.
"""

import argparse
import copy
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from cnf_automator import generate_encoding, verify_assignment
from turyn_policy_transformer import (
    PolicyOnlyTransformer,
    _add_bit_to_seq,
    predict_next_bit_from_prefix,
)


def parse_percentages(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one percentage is required.")
    values = []
    for part in parts:
        value = float(part)
        if value < 0.0 or value > 100.0:
            raise ValueError(f"Percentage must be in [0, 100], got {value}.")
        values.append(value)
    return values


def parse_runs_by_percentage(raw: Optional[str]) -> Dict[str, int]:
    if raw is None:
        return {}

    payload: object
    candidate_path = Path(raw)
    if candidate_path.exists():
        payload = json.loads(candidate_path.read_text())
    else:
        payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError("runs-by-percentage must be a JSON object.")

    parsed: Dict[str, int] = {}
    for k, v in payload.items():
        key = str(k)
        runs = int(v)
        if runs <= 0:
            raise ValueError(f"Run count for {key} must be positive, got {runs}.")
        parsed[key] = runs
    return parsed


def percentage_to_steps(total_steps: int, percentage: float) -> int:
    return max(0, min(total_steps, math.ceil(total_steps * (percentage / 100.0))))


def build_random_prefix(random_steps: int) -> Sequence[tuple]:
    seq = [(1,), (1,), (1,), (1,)]
    for _ in range(random_steps):
        seq = _add_bit_to_seq(copy.deepcopy(seq), random.randint(0, 1))
    return seq


def sample_next_bit_from_ratio(one_ratio: float) -> int:
    return random.choices([0, 1], weights=[1.0 - one_ratio, one_ratio], k=1)[0]


def load_one_ratio_from_data(data_path: Path) -> float:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Data file must contain a non-empty list of (seq, label) samples.")

    count_ones = 0
    count_total = 0
    for sample in samples:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            continue
        label = sample[1]
        label01 = 1 if float(label) > 0 else 0
        count_ones += label01
        count_total += 1

    if count_total == 0:
        raise ValueError("Could not extract labels from data file.")
    return count_ones / count_total


def rollout_until_steps(
    model: PolicyOnlyTransformer,
    steps: int,
    device: str,
    temperature: float,
    strategy: str,
    random_prefix_steps: int,
) -> Sequence[tuple]:
    # Start each run from a randomized prefix while keeping the seeded initial bits.
    warmup_steps = min(random_prefix_steps, steps)
    seq = build_random_prefix(warmup_steps)
    for _ in range(steps - warmup_steps):
        result = predict_next_bit_from_prefix(
            model,
            seq,
            device=device,
            temperature=temperature,
            strategy=strategy,
        )
        seq = _add_bit_to_seq(copy.deepcopy(seq), result["pred"])
    return seq


def random_policy_rollout_until_steps(
    steps: int,
    random_prefix_steps: int,
    one_ratio: float,
) -> Sequence[tuple]:
    warmup_steps = min(random_prefix_steps, steps)
    seq = build_random_prefix(warmup_steps)
    for _ in range(steps - warmup_steps):
        seq = _add_bit_to_seq(copy.deepcopy(seq), sample_next_bit_from_ratio(one_ratio))
    return seq


def sat_check_partial(seq: Sequence[tuple], cnf, varmap) -> bool:
    sat, core = verify_assignment(seq, varmap, cnf)
    # Existing code treats empty/None core as SAT.
    return sat and not core


def evaluate_percentage(
    model: PolicyOnlyTransformer,
    cnf,
    varmap,
    *,
    steps: int,
    runs: int,
    device: str,
    temperature: float,
    strategy: str,
    random_prefix_steps: int,
) -> Dict[str, object]:
    successful = 0
    unsuccessful = 0
    for _ in range(runs):
        partial_seq = rollout_until_steps(
            model=model,
            steps=steps,
            device=device,
            temperature=temperature,
            strategy=strategy,
            random_prefix_steps=random_prefix_steps,
        )
        if sat_check_partial(partial_seq, cnf, varmap):
            successful += 1
        else:
            unsuccessful += 1

    success_rate = successful / runs if runs > 0 else 0.0
    return {
        "runs": runs,
        "successful": successful,
        "unsuccessful": unsuccessful,
        "success_rate": success_rate,
    }


def evaluate_random_policy_percentage(
    cnf,
    varmap,
    *,
    steps: int,
    runs: int,
    random_prefix_steps: int,
    one_ratio: float,
) -> Dict[str, object]:
    successful = 0
    unsuccessful = 0
    for _ in range(runs):
        partial_seq = random_policy_rollout_until_steps(
            steps=steps,
            random_prefix_steps=random_prefix_steps,
            one_ratio=one_ratio,
        )
        if sat_check_partial(partial_seq, cnf, varmap):
            successful += 1
        else:
            unsuccessful += 1

    success_rate = successful / runs if runs > 0 else 0.0
    return {
        "runs": runs,
        "successful": successful,
        "unsuccessful": unsuccessful,
        "success_rate": success_rate,
    }


def load_model(model_path: Path, device: str) -> PolicyOnlyTransformer:
    model = PolicyOnlyTransformer()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval().to(device)
    return model


def run_rollout_evaluation(args: argparse.Namespace) -> Dict[str, object]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model_path), device=device)
    cnf, varmap, _, _ = generate_encoding(args.r)
    total_steps = 4 * (args.r - 1)
    random_prefix_steps = percentage_to_steps(total_steps, 25.0)
    random_policy_data = args.random_policy_data or f"data/data_{args.r}.pkl"
    random_one_ratio = load_one_ratio_from_data(Path(random_policy_data))

    percentages = parse_percentages(args.percentages)
    runs_override = parse_runs_by_percentage(args.runs_by_percentage)

    results: Dict[str, object] = {
        "model_path": str(args.model_path),
        "r": args.r,
        "total_steps": total_steps,
        "random_prefix_steps": random_prefix_steps,
        "random_policy_data": str(random_policy_data),
        "random_policy_one_ratio": random_one_ratio,
        "temperature": args.temperature,
        "strategy": args.strategy,
        "model_percentages": [],
        "random_policy_percentages": [],
    }

    for percentage in percentages:
        key = str(percentage)
        runs = runs_override.get(key, args.runs_per_percentage)
        steps = percentage_to_steps(total_steps, percentage)
        stats = evaluate_percentage(
            model,
            cnf,
            varmap,
            steps=steps,
            runs=runs,
            device=device,
            temperature=args.temperature,
            strategy=args.strategy,
            random_prefix_steps=random_prefix_steps,
        )
        random_stats = evaluate_random_policy_percentage(
            cnf,
            varmap,
            steps=steps,
            runs=runs,
            random_prefix_steps=random_prefix_steps,
            one_ratio=random_one_ratio,
        )

        results["model_percentages"].append(
            {
                "percentage": percentage,
                "steps": steps,
                **stats,
            }
        )
        results["random_policy_percentages"].append(
            {
                "percentage": percentage,
                "steps": steps,
                **random_stats,
            }
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SAT success of partial model rollouts by percentage."
    )
    parser.add_argument("model_path", type=str, help="Path to a model .pt checkpoint")
    parser.add_argument("--r", type=int, default=9, help="Turyn sequence length")
    parser.add_argument(
        "--percentages",
        type=str,
        default="25,50,75,100",
        help="Comma-separated rollout percentages (0..100)",
    )
    parser.add_argument(
        "--runs-percentage",
        type=int,
        default=10,
        dest="runs_per_percentage",
        help="Default number of runs per percentage",
    )
    parser.add_argument(
        "--runs-by-percentage",
        type=str,
        default=None,
        help='Optional JSON object/string override, e.g. \'{"25":20,"50":10}\'',
    )
    parser.add_argument(
        "--strategy",
        choices=["argmax", "sample"],
        default="argmax",
        help="Bit selection strategy",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (used by predictor)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, e.g. "cpu" or "cuda"',
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full JSON results",
    )
    parser.add_argument(
        "--random-policy-data",
        type=str,
        default=None,
        help=(
            "Path to pickle with (seq, label) samples used to estimate random-policy "
            "P(next_bit=1). Defaults to data/data_<r>.pkl if omitted."
        ),
    )
    return parser


def print_summary(results: Dict[str, object]) -> None:
    print("\n--- Rollout Percentage SAT Evaluation ---")
    print(f"model_path: {results['model_path']}")
    print(f"r:          {results['r']}")
    print(f"steps:      {results['total_steps']}")
    print(f"random25:   {results['random_prefix_steps']}")
    print(f"rand_data:  {results['random_policy_data']}")
    print(f"rand_p1:    {results['random_policy_one_ratio']:.4f}")
    print(f"strategy:   {results['strategy']}")
    print(f"temp:       {results['temperature']}")
    print("")
    print("Model policy:")
    for row in results["model_percentages"]:
        print(
            f"{row['percentage']:>6.2f}% | steps={row['steps']:>3} | "
            f"runs={row['runs']:>4} | success={row['successful']:>4} | "
            f"fail={row['unsuccessful']:>4} | rate={row['success_rate']:.3f}"
        )
    print("")
    print("Random policy:")
    for row in results["random_policy_percentages"]:
        print(
            f"{row['percentage']:>6.2f}% | steps={row['steps']:>3} | "
            f"runs={row['runs']:>4} | success={row['successful']:>4} | "
            f"fail={row['unsuccessful']:>4} | rate={row['success_rate']:.3f}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.r < 2:
        raise ValueError(f"r must be >= 2, got {args.r}")
    if args.runs_per_percentage <= 0:
        raise ValueError(
            f"--runs-percentage must be positive, got {args.runs_per_percentage}"
        )
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be positive, got {args.temperature}")

    results = run_rollout_evaluation(args)
    print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved JSON results to {output_path}")


if __name__ == "__main__":
    main()
