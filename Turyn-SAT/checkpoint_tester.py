from pathlib import Path
import argparse
import pickle
import statistics

import torch

from turyn_policy_transformer import (
    PolicyOnlyTransformer,
    evaluate_model_with_sat,
    evaluate_random_policy_with_sat,
)

METRIC_KEYS = (
    "count_right",
    "count_wrong",
    "first_incorrect",
    "num_unsat",
    "sum_unsat",
    "pred_one",
    "count_one",
    "count_total",
)


def _mean_std(values):
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, 0.0
    return mean, statistics.stdev(values)


def load_label_one_ratio_from_pickle(data_path: Path) -> float:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with open(data_path, "rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Dataset must be a non-empty list of (seq, label) samples.")

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
        raise ValueError("Could not extract labels from dataset.")
    return count_ones / count_total


def _print_metric_block(title, trial_results, runs, temperature, strategy):
    print(f"\n--- {title} ---")
    if runs == 1:
        m = trial_results[0]
        for key in METRIC_KEYS:
            print(f"  {key:16} {m[key]}")
    else:
        print(
            f"  ({runs} trials, temperature={temperature}, strategy={strategy})"
        )
        for key in METRIC_KEYS:
            vals = [t[key] for t in trial_results]
            mu, sigma = _mean_std(vals)
            print(f"  {key:16} mean={mu:.4g}  std={sigma:.4g}")


def evaluate_all_models_with_sat(
    models_dir="models",
    r=9,
    runs=1,
    temperature=1.0,
    random_policy_data: Path | None = None,
):
    """Run SAT evaluation for every checkpoint in models_dir recursively."""
    model_paths = sorted(Path(models_dir).rglob("*.pt"))
    # model_paths = ["model_13_dpo.pt"]

    strategy = "sample" if runs > 1 else "argmax"
    step_verbose = runs == 1

    if random_policy_data is not None:
        label_one_ratio = load_label_one_ratio_from_pickle(random_policy_data)
        print(
            f"\nRandom policy baseline (P(next_bit=1)={label_one_ratio:.6f} "
            f"from labels in {random_policy_data})"
        )
        random_trials = []
        for k in range(runs):
            if runs > 1:
                print(f"  trial {k + 1}/{runs}")
            random_trials.append(
                evaluate_random_policy_with_sat(
                    r=r,
                    one_ratio=label_one_ratio,
                    verbose=step_verbose,
                )
            )
        _print_metric_block(
            "SAT Evaluation (random policy, dataset label ratio)",
            random_trials,
            runs,
            temperature,
            "label-weighted sample",
        )

    if not model_paths:
        print(f"No model checkpoints found in {models_dir}")
        return

    for model_path in model_paths:
        print(f"\nEvaluating: {model_path}")
        model = PolicyOnlyTransformer()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"])

        trial_results = []
        for k in range(runs):
            if runs > 1:
                print(f"  trial {k + 1}/{runs}")
            trial_results.append(
                evaluate_model_with_sat(
                    model,
                    r=r,
                    verbose=step_verbose,
                    temperature=temperature,
                    strategy=strategy,
                )
            )

        _print_metric_block(
            "SAT Evaluation (model policy)",
            trial_results,
            runs,
            temperature,
            strategy,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Turyn policy checkpoints with SAT rollout; optional multi-run stats."
    )
    parser.add_argument(
        "sequence_length",
        type=int,
        help="Turyn sequence length (r) used for dataset generation and evaluation.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory to recursively search for .pt checkpoints (default: models).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation trials per checkpoint (uses sampling when > 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for next-bit prediction (affects stochastic trials).",
    )
    parser.add_argument(
        "--random-policy-data",
        type=str,
        default=None,
        help=(
            "Optional pickle path: list of (seq, label) samples; run a random-policy "
            "SAT baseline using P(1) = fraction of labels > 0 in the file."
        ),
    )

    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    models_dir = args.dir if args.dir is not None else "models"
    random_path = (
        Path(args.random_policy_data).resolve()
        if args.random_policy_data is not None
        else None
    )
    evaluate_all_models_with_sat(
        models_dir=models_dir,
        r=args.sequence_length,
        runs=args.runs,
        temperature=args.temperature,
        random_policy_data=random_path,
    )

if __name__ == "__main__":
    main()
