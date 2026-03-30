"""
DPO training for Turyn policy: preferences from SAT vs UNSAT next-bit branches
on random partial sequences.

Each example is a parent partial (random interleaved bits) and labels (y_w, y_l)
for the two possible next bits such that exactly one extension is SAT.
"""

from __future__ import annotations

import argparse
import copy
import pickle
import random
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cnf_automator import generate_encoding, verify_assignment
from sequence_generator_sat import (
    next_row_for_bit,
    random_partial_sequence,
    seq_to_tuples,
)
from turyn_policy_transformer import (
    PAD_ID,
    PolicyOnlyTransformer,
    create_and_filter_dataset,
    evaluate_model_with_sat,
    evaluate_policy,
    flatten_state,
    load_policy_checkpoint,
    PrefixNextBitPaddedDataset,
    save_policy_checkpoint,
)

RolloutPartial = Tuple[
    Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]
]


def build_supervised_test_dataset(
    r: int,
    *,
    data_path: Optional[str] = None,
    limit: Optional[int] = None,
    partial_flat_len: int = 4,
    test_frac: float = 0.2,
    split_seed: int = 42,
) -> Tuple[Dataset, dict]:
    """
    Same train/test split convention as ``train_with_split`` in
    ``turyn_policy_transformer`` (``random_split`` on ``PrefixNextBitPaddedDataset``).

    Returns (test subset dataset, metadata dict for logging/checkpoints).
    """
    samples = create_and_filter_dataset(
        r,
        data_path=data_path,
        limit=limit,
        partial_flat_len=partial_flat_len,
    )
    if not samples:
        raise ValueError(
            "No supervised samples after filtering; check r, --sup-data, or --sup-limit."
        )
    full_ds = PrefixNextBitPaddedDataset(samples)
    n_total = len(full_ds)
    n_test = int(round(test_frac * n_total))
    n_train = n_total - n_test
    if n_test < 1 or n_train < 1:
        raise ValueError(
            f"Supervised split yields empty train or test (n_total={n_total}, "
            f"test_frac={test_frac})."
        )
    g = torch.Generator().manual_seed(split_seed)
    _train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=g)
    meta = {
        "n_supervised_total": n_total,
        "n_supervised_train": n_train,
        "n_supervised_test": n_test,
        "test_frac": test_frac,
        "split_seed": split_seed,
        "partial_flat_len": partial_flat_len,
        "data_path": data_path,
        "limit": limit,
    }
    return test_ds, meta


def try_sample_dpo_triple(
    r: int,
    cnf: Any,
    varmap: Any,
    rng: random.Random,
) -> Optional[Tuple[RolloutPartial, int, int]]:
    """
    Sample a random partial (fixed flat length in [4, 4*r)), then look at the
    next append position. If exactly one of {-1, +1} keeps the assignment SAT,
    return (parent_partial, y_w, y_l) with y_* in {0,1} matching flatten_state.
    Otherwise return None.
    """
    partial_flat_len = rng.randint(4, 4 * r - 1)
    parent = random_partial_sequence(r, partial_flat_len, rng=rng)
    seq = [list(row) for row in parent]
    row = next_row_for_bit(seq)
    sat_by: dict[int, bool] = {}
    for bit in (-1, 1):
        trial = copy.deepcopy(seq)
        trial[row].append(bit)
        if any(len(x) > r for x in trial):
            sat_by[bit] = False
        else:
            sat_by[bit], _ = verify_assignment(seq_to_tuples(trial), varmap, cnf)
    if sat_by.get(-1) == sat_by.get(1):
        return None
    sat_bit = -1 if sat_by[-1] else 1
    unsat_bit = 1 if sat_by[-1] else -1
    y_w = 1 if sat_bit > 0 else 0
    y_l = 1 if unsat_bit > 0 else 0
    return (parent, y_w, y_l)


def generate_dpo_dataset(
    r: int,
    n_samples: int,
    seed: int,
    max_tries_factor: int = 200,
) -> List[Tuple[RolloutPartial, int, int]]:
    """
    Build up to n_samples unique preference triples (parent, y_w, y_l).
    Skips random partials where both or neither next-bit extension is SAT.
    """
    rng = random.Random(seed)
    cnf, varmap, _, _ = generate_encoding(r)
    out: List[Tuple[RolloutPartial, int, int]] = []
    seen: set[str] = set()
    tries = 0
    max_tries = max(n_samples * max_tries_factor, n_samples + 10)
    while len(out) < n_samples and tries < max_tries:
        tries += 1
        t = try_sample_dpo_triple(r, cnf, varmap, rng)
        if t is None:
            continue
        parent, y_w, y_l = t
        key = repr((parent, y_w, y_l))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) %  100 == 0:
            print("Collected", len(out), "samples")
    if len(out) < n_samples:
        raise RuntimeError(
            f"Only collected {len(out)}/{n_samples} DPO triples after {tries} tries "
            f"(r={r}). Try larger max_tries_factor or smaller r."
        )
    return out


def _default_dpo_dataset_path(
    cache_dir: str,
    *,
    r: int,
    n_samples: int,
    seed: int,
    split: str,
) -> Path:
    base = Path(cache_dir)
    return base / f"dpo_{split}_r{r}_n{n_samples}_seed{seed}.pkl"


def _save_dpo_triples(path: Path, *, triples: List[Tuple[RolloutPartial, int, int]], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"triples": triples, "meta": meta}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[dpo-data] saved {len(triples)} samples -> {path}")


def _load_dpo_triples(path: Path) -> Tuple[List[Tuple[RolloutPartial, int, int]], dict]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "triples" not in payload:
        raise ValueError(f"Invalid DPO dataset file format: {path}")
    triples = payload["triples"]
    meta = payload.get("meta", {})
    if not isinstance(triples, list):
        raise ValueError(f"DPO dataset file has non-list triples: {path}")
    return triples, meta


def get_or_create_dpo_split(
    *,
    r: int,
    n_samples: int,
    seed: int,
    split: str,
    cache_dir: str,
    explicit_path: Optional[str] = None,
    force_regen: bool = False,
) -> Tuple[List[Tuple[RolloutPartial, int, int]], str]:
    out_path = Path(explicit_path) if explicit_path else _default_dpo_dataset_path(
        cache_dir,
        r=r,
        n_samples=n_samples,
        seed=seed,
        split=split,
    )
    if out_path.exists() and not force_regen:
        triples, meta = _load_dpo_triples(out_path)
        if (
            meta.get("r") != r
            or meta.get("n_samples") != n_samples
            or meta.get("seed") != seed
            or meta.get("split") != split
        ):
            raise ValueError(
                f"Cached {split} data at {out_path} does not match requested "
                f"(expected r={r}, n_samples={n_samples}, seed={seed}). "
                "Use --dpo-force-regen to overwrite, or pass a matching file."
            )
        print(f"[dpo-data] loaded {len(triples)} {split} samples <- {out_path}")
        return triples, str(out_path)

    triples = generate_dpo_dataset(r, n_samples, seed)
    meta = {"r": r, "n_samples": n_samples, "seed": seed, "split": split}
    _save_dpo_triples(out_path, triples=triples, meta=meta)
    return triples, str(out_path)


class DPOPreferenceDataset(Dataset):
    """Each item: (x [T], y_w int, y_l int) with x = flatten_state(parent)."""

    def __init__(self, triples: Sequence[Tuple[RolloutPartial, int, int]]):
        self.data = []
        for parent, y_w, y_l in triples:
            x = flatten_state(parent)
            self.data.append((x, int(y_w), int(y_l)))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


def collate_dpo(
    batch: List[Tuple[torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return X, Y_w, Y_l, M (pad mask True at PAD)."""
    xs, y_ws, y_ls = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    padded, masks = [], []
    for x in xs:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x_pad = torch.cat(
                [x, torch.full((pad_len,), PAD_ID, dtype=torch.long)]
            )
            m_pad = torch.cat(
                [
                    torch.zeros(x.size(0), dtype=torch.bool),
                    torch.ones(pad_len, dtype=torch.bool),
                ]
            )
        else:
            x_pad = x
            m_pad = torch.zeros_like(x, dtype=torch.bool)
        padded.append(x_pad)
        masks.append(m_pad)

    X = torch.stack(padded)
    Y_w = torch.tensor(y_ws, dtype=torch.long)
    Y_l = torch.tensor(y_ls, dtype=torch.long)
    M = torch.stack(masks)
    return X, Y_w, Y_l, M


def log_prob_actions(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """logits [B,2], y [B] -> log pi(y|x) [B]."""
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(1, y.unsqueeze(1)).squeeze(1)


def dpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    y_w: torch.Tensor,
    y_l: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    logp_w = log_prob_actions(policy_logits, y_w)
    logp_l = log_prob_actions(policy_logits, y_l)
    logr_w = log_prob_actions(ref_logits, y_w)
    logr_l = log_prob_actions(ref_logits, y_l)
    z = beta * ((logp_w - logp_l) - (logr_w - logr_l))
    return -F.logsigmoid(z).mean()


@torch.no_grad()
def evaluate_dpo_loader(
    policy: nn.Module,
    ref: nn.Module,
    loader: DataLoader,
    device: str,
    beta: float,
) -> dict:
    policy.eval()
    ref.eval()
    tot_loss = 0.0
    tot_pref = 0.0
    n = 0
    for X, Y_w, Y_l, M in loader:
        X, Y_w, Y_l, M = X.to(device), Y_w.to(device), Y_l.to(device), M.to(device)
        lp = policy(X, pad_mask=M)
        lr = ref(X, pad_mask=M)
        loss = dpo_loss(lp, lr, Y_w, Y_l, beta)
        logp_w = log_prob_actions(lp, Y_w)
        logp_l = log_prob_actions(lp, Y_l)
        pref = (logp_w > logp_l).float()
        bs = X.size(0)
        tot_loss += loss.item() * bs
        tot_pref += pref.sum().item()
        n += bs
    return {
        "dpo_loss": tot_loss / max(n, 1),
        "pref_acc": tot_pref / max(n, 1),
        "n": n,
    }


def train_dpo(
    r: int,
    train_triples: List[Tuple[RolloutPartial, int, int]],
    test_triples: List[Tuple[RolloutPartial, int, int]],
    *,
    policy: Optional[PolicyOnlyTransformer] = None,
    ref_policy: Optional[PolicyOnlyTransformer] = None,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 3e-4,
    beta: float = 0.1,
    device: Optional[str] = None,
    grad_clip: float = 1.0,
    max_sat_steps: Optional[int] = None,
    supervised_test_ds: Optional[Dataset] = None,
    supervised_eval_batch_size: int = 256,
) -> Tuple[PolicyOnlyTransformer, List[dict]]:
    """
    Train with DPO; after **each** training batch, evaluate on the test DPO loader
    and run ``evaluate_model_with_sat`` (verbose off). After **each** epoch,
    optionally run supervised cross-entropy eval on ``supervised_test_ds``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if policy is None:
        policy = PolicyOnlyTransformer(
            vocab_size=3, pad_id=PAD_ID, max_len=4096
        ).to(device)
    else:
        policy = policy.to(device)

    if ref_policy is None:
        ref_policy = copy.deepcopy(policy)
    else:
        ref_policy = copy.deepcopy(ref_policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.to(device)

    train_ds = DPOPreferenceDataset(train_triples)
    test_ds = DPOPreferenceDataset(test_triples)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_dpo,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dpo,
    )

    opt = AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    history: List[dict] = []

    global_batch = 0
    last_sup_eval: Optional[dict] = None
    for ep in range(1, epochs + 1):
        policy.train()
        for X, Y_w, Y_l, M in train_loader:
            global_batch += 1
            X = X.to(device)
            Y_w, Y_l = Y_w.to(device), Y_l.to(device)
            M = M.to(device)

            logits_pi = policy(X, pad_mask=M)
            with torch.no_grad():
                logits_ref = ref_policy(X, pad_mask=M)
            loss = dpo_loss(logits_pi, logits_ref, Y_w, Y_l, beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()

            te = evaluate_dpo_loader(policy, ref_policy, test_loader, device, beta)
            sat = evaluate_model_with_sat(
                policy,
                r=r,
                max_steps=max_sat_steps,
                device=device,
                verbose=False,
            )
            row = {
                "epoch": ep,
                "train_batch": global_batch,
                "train_dpo_loss": float(loss.item()),
                "test_dpo_loss": te["dpo_loss"],
                "test_pref_acc": te["pref_acc"],
                "sat_count_right": sat["count_right"],
                "sat_count_wrong": sat["count_wrong"],
            }
            if last_sup_eval is not None:
                row["sup_test_loss"] = last_sup_eval["loss"]
                row["sup_test_acc"] = last_sup_eval["acc"]
            history.append(row)
            sup_suffix = ""
            if last_sup_eval is not None:
                sup_suffix = (
                    f" | sup_test loss {last_sup_eval['loss']:.4f} "
                    f"acc {last_sup_eval['acc']:.3f} (end ep {ep - 1})"
                )
            print(
                f"ep {ep} batch {len(history)} | train_dpo {loss.item():.4f} | "
                f"test_dpo {te['dpo_loss']:.4f} test_pref {te['pref_acc']:.3f} | "
                f"sat right/wrong {sat['count_right']}/{sat['count_wrong']}"
                f"{sup_suffix}"
            )

        if supervised_test_ds is not None and len(supervised_test_ds) > 0:
            sup = evaluate_policy(
                policy,
                supervised_test_ds,
                batch_size=supervised_eval_batch_size,
                device=device,
            )
            last_sup_eval = {"loss": sup["loss"], "acc": sup["acc"], "n": sup["n"]}
            if history:
                history[-1]["sup_test_loss"] = sup["loss"]
                history[-1]["sup_test_acc"] = sup["acc"]
            print(
                f"--- end epoch {ep} | supervised test CE loss {sup['loss']:.4f} "
                f"acc {sup['acc']:.4f} (n={sup['n']}) ---"
            )

    return policy, history


def main():
    parser = argparse.ArgumentParser(description="Turyn policy DPO (SAT vs UNSAT next bit).")
    parser.add_argument("r", type=int, help="Sequence length (Turyn r).")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=12345)
    parser.add_argument(
        "--dpo-cache-dir",
        type=str,
        default="data/dpo_samples",
        help=(
            "Directory for persisted DPO train/test triples. Used when explicit "
            "--dpo-train-data/--dpo-test-data are not provided."
        ),
    )
    parser.add_argument(
        "--dpo-train-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional explicit path for persisted DPO train triples (.pkl).",
    )
    parser.add_argument(
        "--dpo-test-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional explicit path for persisted DPO test triples (.pkl).",
    )
    parser.add_argument(
        "--dpo-force-regen",
        action="store_true",
        help="Regenerate DPO train/test triples even if persisted files exist.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional supervised policy .pt; reference is a frozen copy of initial weights.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save final policy checkpoint.",
    )
    parser.add_argument(
        "--supervised-eval",
        action="store_true",
        help=(
            "After each DPO epoch, evaluate cross-entropy on the supervised test split "
            "(same construction as turyn_policy_transformer.train_with_split)."
        ),
    )
    parser.add_argument(
        "--sup-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional pickle of (partial, next_bit) samples (see turyn_policy_transformer --data).",
    )
    parser.add_argument(
        "--sup-limit",
        type=int,
        default=None,
        help="Optional cap when generating supervised samples (matches create_and_filter_dataset).",
    )
    parser.add_argument(
        "--sup-partial-flat-len",
        type=int,
        default=4,
        metavar="L",
        help="partial_flat_len when generating supervised data (ignored if --sup-data is set).",
    )
    parser.add_argument(
        "--sup-test-frac",
        type=float,
        default=0.2,
        help="Fraction held out as supervised test set (default: 0.2).",
    )
    parser.add_argument(
        "--sup-split-seed",
        type=int,
        default=42,
        help="Torch Generator seed for supervised train/test split (default: 42, matches supervised training).",
    )
    parser.add_argument(
        "--sup-eval-batch-size",
        type=int,
        default=256,
        help="Batch size for supervised test evaluation.",
    )
    args = parser.parse_args()

    supervised_test_ds = None
    supervised_meta: Optional[dict] = None
    if args.supervised_eval:
        print("[supervised] building test split for cross-entropy eval…")
        supervised_test_ds, supervised_meta = build_supervised_test_dataset(
            args.r,
            data_path=args.sup_data,
            limit=args.sup_limit,
            partial_flat_len=args.sup_partial_flat_len,
            test_frac=args.sup_test_frac,
            split_seed=args.sup_split_seed,
        )
        print(
            f"[supervised] test n={supervised_meta['n_supervised_test']} "
            f"(total filtered n={supervised_meta['n_supervised_total']})"
        )

    print(
        f"[dpo-data] preparing train ({args.train_samples}) and test ({args.test_samples})…"
    )
    train_triples, train_data_path = get_or_create_dpo_split(
        r=args.r,
        n_samples=args.train_samples,
        seed=args.train_seed,
        split="train",
        cache_dir=args.dpo_cache_dir,
        explicit_path=args.dpo_train_data,
        force_regen=args.dpo_force_regen,
    )
    test_triples, test_data_path = get_or_create_dpo_split(
        r=args.r,
        n_samples=args.test_samples,
        seed=args.test_seed,
        split="test",
        cache_dir=args.dpo_cache_dir,
        explicit_path=args.dpo_test_data,
        force_regen=args.dpo_force_regen,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = None
    if args.init_checkpoint:
        policy, _payload = load_policy_checkpoint(args.init_checkpoint, device=device)

    policy, hist = train_dpo(
        args.r,
        train_triples,
        test_triples,
        policy=policy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        device=device,
        supervised_test_ds=supervised_test_ds,
        supervised_eval_batch_size=args.sup_eval_batch_size,
    )

    if args.supervised_eval and hist:
        final_row = next(
            (row for row in reversed(hist) if "sup_test_loss" in row),
            None,
        )
        if final_row is not None:
            print(
                "\n--- Final supervised test (last epoch) ---\n"
                f"  loss: {final_row['sup_test_loss']:.4f}\n"
                f"  acc:  {final_row['sup_test_acc']:.4f}"
            )

    if args.save:
        train_cfg = {
            "dpo": True,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "beta": args.beta,
            "supervised_eval": bool(args.supervised_eval),
            "dpo_train_data": train_data_path,
            "dpo_test_data": test_data_path,
            "hist_tail": hist[-3:] if len(hist) > 3 else hist,
        }
        if supervised_meta is not None:
            train_cfg["supervised_meta"] = supervised_meta
        save_policy_checkpoint(
            policy,
            args.save,
            r=args.r,
            train_config=train_cfg,
        )


if __name__ == "__main__":
    main()
