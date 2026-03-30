"""
Turyn policy transformer: dataset creation, supervised learning with train/test split,
and evaluation. Run with a sequence length (r) to get supervised learning and evaluation results.
"""

import argparse
import copy
from collections import deque
import pickle
import random
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure local imports work when run from repo root or script dir
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sequence_generator_sat import generate_dataset_sat
from cnf_automator import verify_assignment, generate_encoding

PAD_ID = 2

# (partial as four rows of tuples, next_bit in {-1, 1}) — same as create_partials output
Sample = Tuple[Any, int]


def flatten_state(cur_seq):
    flat = []
    for seq in cur_seq:
        for b in seq:
            flat.append(1 if b > 0 else 0)
    return torch.tensor(flat, dtype=torch.long)


class PrefixNextBitPaddedDataset(Dataset):
    def __init__(self, samples):
        self.data = []
        for seq, label in samples:
            x = flatten_state(seq)
            y = 1 if label > 0 else 0
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_pad(batch):
    """
    batch: list of (x, y) where x is [T_i] of 0/1 tokens
    returns:
      X: [B, T] int64 in {0,1,PAD_ID}
      Y: [B]     int64 in {0,1}
      M: [B, T]  bool   True where PAD
    """
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    padded, masks = [], []
    for x in xs:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x_pad = torch.cat([x, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
            m_pad = torch.cat(
                [
                    torch.zeros(x.size(0), dtype=torch.bool),
                    torch.ones(pad_len, dtype=torch.bool),
                ]
            )  # True = PAD
        else:
            x_pad = x
            m_pad = torch.zeros_like(x, dtype=torch.bool)
        padded.append(x_pad)
        masks.append(m_pad)

    X = torch.stack(padded)  # [B, T]
    Y = torch.tensor(ys, dtype=torch.long)  # [B]
    M = torch.stack(masks)  # [B, T] (True=PAD)
    return X, Y, M


class PolicyOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=3,  # tokens {0,1,PAD_ID}
        pad_id=PAD_ID,
        max_len=4096,
        d_model=256,
        nhead=8,
        nlayers=4,
        dim_ff=512,
        dropout=0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)  # logits for {0,1}

    def forward(self, x, pad_mask):
        """
        x:        [B, T] int64 in {0,1,pad_id}
        pad_mask: [B, T] bool, True where PAD (ignored by attention)
        """
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)

        h = self.encoder(h, src_key_padding_mask=pad_mask)

        valid = ~pad_mask
        lengths = valid.long().sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(B, 1, 1).expand(B, 1, h.size(-1))
        last_h = h.gather(1, idx).squeeze(1)

        logits = self.head(self.norm(last_h))
        return logits


def save_samples_pickle(samples: List[Sample], path: str) -> None:
    """Write filtered (partial, next_bit) samples with pickle (compatible with --data)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save] wrote {len(samples)} samples to {p}")


def save_policy_checkpoint(
    model: nn.Module,
    path: str,
    *,
    r: int,
    hist: Optional[dict] = None,
    train_config: Optional[dict] = None,
) -> None:
    """Save state_dict and kwargs needed to rebuild PolicyOnlyTransformer."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "r": r,
        "policy_kwargs": {
            "vocab_size": 3,
            "pad_id": PAD_ID,
            "max_len": getattr(model.pos_emb, "num_embeddings", 4096),
        },
    }
    if hist is not None:
        payload["hist"] = hist
    if train_config is not None:
        payload["train_config"] = train_config
    torch.save(payload, p)
    print(f"[save] wrote model checkpoint to {p}")


def load_policy_checkpoint(
    path: str, device: Optional[str] = None
) -> Tuple[PolicyOnlyTransformer, dict]:
    """Load a checkpoint written by save_policy_checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(path), map_location=device)
    kw = dict(payload["policy_kwargs"])
    model = PolicyOnlyTransformer(**kw).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, payload


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()
    tot_loss = 0.0
    tot_correct = 0
    n = 0
    for X, Y, M in loader:
        X, Y, M = X.to(device), Y.to(device), M.to(device)
        logits = model(X, pad_mask=M)
        loss = F.cross_entropy(logits, Y, reduction="sum")
        tot_loss += loss.item()
        tot_correct += (logits.argmax(-1) == Y).sum().item()
        n += X.size(0)
    return (tot_loss / max(n, 1), tot_correct / max(n, 1))


def _epochs_for_midtraining_saves(total_epochs: int, n_saves: int = 5) -> List[int]:
    """
    Epoch indices (1-based) at which to save checkpoints: n_saves saves starting from
    the halfway epoch. If fewer than n_saves epochs remain from halfway to the end,
    epochs are chosen evenly from halfway through the final epoch (may repeat).
    """
    if total_epochs < 1 or n_saves < 1:
        return []
    first = max(1, (total_epochs + 1) // 2)
    last = total_epochs
    span = last - first + 1
    if span >= n_saves:
        return [first + i for i in range(n_saves)]
    if span <= 1:
        return [last] * n_saves
    out: List[int] = []
    for k in range(n_saves):
        t = k / (n_saves - 1) if n_saves > 1 else 0.0
        ep = int(round(first + (last - first) * t))
        ep = max(first, min(last, ep))
        out.append(ep)
    return out


def train_with_split(
    samples,
    test_frac=0.2,
    test_samples=None,
    seed=42,
    epochs=30,
    batch_size=64,
    lr=3e-4,
    device=None,
    mid_checkpoint_base: Optional[str] = None,
    mid_checkpoint_r: Optional[int] = None,
    mid_checkpoint_train_config: Optional[dict] = None,
    n_mid_checkpoints: int = 5,
):
    """Train with train/test split. No plotting."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if test_samples is None:
        full_ds = PrefixNextBitPaddedDataset(samples)
        n_total = len(full_ds)
        n_test = int(round(test_frac * n_total))
        n_train = n_total - n_test
        g = torch.Generator().manual_seed(seed)
        train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=g)
        print(f"[split] train={n_train} test={n_test}")
    else:
        train_ds = PrefixNextBitPaddedDataset(samples)
        test_ds = PrefixNextBitPaddedDataset(test_samples)
        print(f"[explicit] train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pad,
    )

    model = PolicyOnlyTransformer(
        vocab_size=3, pad_id=PAD_ID, max_len=4096
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    mid_save_schedule: deque[int] = deque()
    if mid_checkpoint_base is not None and mid_checkpoint_r is not None:
        mid_save_schedule = deque(
            _epochs_for_midtraining_saves(epochs, n_mid_checkpoints)
        )

    for ep in range(1, epochs + 1):
        model.train()
        for X, Y, M in train_loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            logits = model(X, pad_mask=M)
            loss = F.cross_entropy(logits, Y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        tr_loss, tr_acc = evaluate_epoch(model, train_loader, device)
        te_loss, te_acc = evaluate_epoch(model, test_loader, device)

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss)
        hist["test_acc"].append(te_acc)

        print(
            f"epoch {ep:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"test loss {te_loss:.4f} acc {te_acc:.3f}"
        )

        while mid_save_schedule and mid_save_schedule[0] == ep:
            mid_save_schedule.popleft()
            base = Path(mid_checkpoint_base)
            stem = base.stem
            suffix = base.suffix if base.suffix else ".pt"
            # Ordinal prefix so five files are always distinct (even if epochs repeat).
            n_done = n_mid_checkpoints - len(mid_save_schedule)
            out_path = base.parent / f"{stem}_mid_{n_done:02d}_ep{ep:03d}{suffix}"
            hist_so_far = {k: list(v) for k, v in hist.items()}
            save_policy_checkpoint(
                model,
                out_path,
                r=mid_checkpoint_r,
                hist=hist_so_far,
                train_config=mid_checkpoint_train_config,
            )

    return model, (train_ds, test_ds), hist


@torch.no_grad()
def evaluate_policy(model, dataset, batch_size=256, device=None, return_mistakes=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pad
    )

    tot_loss = tot_correct = n = 0
    mistakes = []
    for X, Y, M in loader:
        X, Y, M = X.to(device), Y.to(device), M.to(device)
        logits = model(X, pad_mask=M)
        loss = F.cross_entropy(logits, Y, reduction="sum")
        pred = logits.argmax(dim=-1)

        tot_loss += loss.item()
        tot_correct += (pred == Y).sum().item()
        n += X.size(0)

        if return_mistakes:
            wrong = (pred != Y).nonzero(as_tuple=False).squeeze(1)
            for i in wrong.tolist():
                mistakes.append(
                    {
                        "idx": n - X.size(0) + i,
                        "true": int(Y[i].item()),
                        "pred": int(pred[i].item()),
                        "prob": float(
                            torch.softmax(logits[i], -1)[pred[i]].item()
                        ),
                    }
                )

    return {
        "loss": tot_loss / max(n, 1),
        "acc": tot_correct / max(n, 1),
        "n": n,
        "mistakes": mistakes if return_mistakes else None,
    }

def evaluate_random_policy(train_dataset, test_dataset):
    train_count_1 = sum(1 for i in range(len(train_dataset)) if train_dataset[i][1] == 1)
    train_1_ratio = train_count_1 / max(len(train_dataset), 1)
    tot_correct = 0
    n = 0

    for item in test_dataset:
        x, y = item
        pred = random.choices([0, 1], weights=[1 - train_1_ratio, train_1_ratio], k=1)[0]
        if pred == y:
            tot_correct += 1
        n += 1

    return {
        "acc": tot_correct / max(n, 1),
        "n": n,
    }


@torch.no_grad()
def predict_next_bit_from_prefix(model, cur_seq, device=None, temperature=1.0, strategy="argmax"):
    """
    cur_seq: list of sequences, e.g. [X, Y, Z, W], each a list/1D tensor of ±1
    returns: dict(pred in {0,1}, conf, probs[2], logits[2])
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    bits = flatten_state(cur_seq)
    T = bits.numel()
    max_len = model.pos_emb.num_embeddings

    if T > max_len:
        raise ValueError(f"prefix length {T} exceeds model max_len {max_len}")

    x = torch.full((1, T if T > 0 else 1), PAD_ID, dtype=torch.long)
    if T > 0:
        x[0, :T] = bits
    pad_mask = x == PAD_ID

    x, pad_mask = x.to(device), pad_mask.to(device)
    logits = model(x, pad_mask=pad_mask).squeeze(0)

    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if strategy == "sample":
        pred = int(torch.multinomial(probs, 1).item())
    else:
        pred = int(probs.argmax().item())

    return {
        "pred": pred,
        "conf": float(probs[pred].item()),
        "probs": probs.detach().cpu(),
        "logits": logits.detach().cpu(),
    }


def _add_bit_to_seq(seq, next_bit):
    """
    Match the notebook's add(seq, next_bit): keep rows as equal-length as possible.
    """
    first_len = len(seq[0])
    added = False
    for i in range(4):
        if len(seq[i]) < first_len:
            seq[i] = seq[i] + (next_bit,)
            added = True
            break
    if not added:
        seq[0] = seq[0] + (next_bit,)
    return seq


def evaluate_model_with_sat(
    model, r=11, max_steps=None, device=None, verbose=True
):
    """
    Load a trained model, iteratively predict next bits, check each choice with the SAT solver,
    and return final metrics (mirroring the notebook cell).

    If ``verbose`` is True, print per-step "ONLY ONE CHOICE WORKS" diagnostics (original notebook behavior).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Initial all-ones prefix as in the notebook
    seq = [(1,), (1,), (1,), (1,)]

    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    rev_varmap = {v: k for k, v in varmap.items()}

    model.eval()

    num_unsat = 0
    sum_unsat = 0
    count_right = 0
    count_wrong = 0
    pred_one = 0
    count_one = 0
    count_total = 0
    first_incorrect = -1

    steps = max_steps if max_steps is not None else 4 * (r - 1)

    for i in range(steps):
        res = predict_next_bit_from_prefix(model, seq, device=device)
        next_bit = res["pred"]
        prev_seq = copy.deepcopy(seq)
        seq = _add_bit_to_seq(seq, next_bit)

        sat, core = verify_assignment(seq, varmap, cnf)

        if core:
            if first_incorrect == -1:
                first_incorrect = i + 4
            num_unsat += 1
            sum_unsat += len(core)
            seq = copy.deepcopy(prev_seq)
            seq = _add_bit_to_seq(seq, 1 - next_bit)

        alt_seq = _add_bit_to_seq(copy.deepcopy(prev_seq), 1 - next_bit)
        sat_alt, core_alt = verify_assignment(alt_seq, varmap, cnf)

        if core or core_alt:
            if verbose:
                print(i, "ONLY ONE CHOICE WORKS")
                if core:
                    print("WRONG")
                else:
                    print("RIGHT")
            if core:
                count_wrong += 1
            else:
                count_right += 1

    return {
        "count_right": count_right,
        "count_wrong": count_wrong,
        "pred_one": pred_one,
        "count_one": count_one,
        "count_total": count_total,
        "final_seq": seq,
        "first_incorrect": first_incorrect,
        "num_unsat": num_unsat,
        "sum_unsat": sum_unsat,
    }


def evaluate_random_policy_with_sat(train_dataset, r=11, max_steps=None):
    """
    Sequential SAT-guided evaluation using a random policy.

    The random policy samples bits with probabilities equal to the label
    distribution in the training dataset (0/1 labels), analogous to
    evaluate_random_policy, but applied in the sequential SAT setting.
    """
    train_count_1 = sum(1 for i in range(len(train_dataset)) if train_dataset[i][1] == 1)
    train_1_ratio = train_count_1 / max(len(train_dataset), 1)

    seq = [(1,), (1,), (1,), (1,)]

    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    rev_varmap = {v: k for k, v in varmap.items()}

    num_unsat = 0
    sum_unsat = 0
    count_right = 0
    count_wrong = 0
    pred_one = 0
    count_one = 0
    count_total = 0
    first_incorrect = -1

    steps = max_steps if max_steps is not None else 4 * (r - 1)

    for i in range(steps):
        next_bit = random.choices(
            [0, 1],
            weights=[1.0 - train_1_ratio, train_1_ratio],
            k=1,
        )[0]

        prev_seq = copy.deepcopy(seq)
        seq = _add_bit_to_seq(seq, next_bit)

        sat, core = verify_assignment(seq, varmap, cnf)

        if core:
            if first_incorrect == -1:
                first_incorrect = i + 4
            num_unsat += 1
            sum_unsat += len(core)
            seq = copy.deepcopy(prev_seq)
            seq = _add_bit_to_seq(seq, 1 - next_bit)

        alt_seq = _add_bit_to_seq(copy.deepcopy(prev_seq), 1 - next_bit)
        sat_alt, core_alt = verify_assignment(alt_seq, varmap, cnf)

        if core or core_alt:
            print(i, "ONLY ONE CHOICE WORKS")
            if core:
                print("WRONG")
                count_wrong += 1
            else:
                print("RIGHT")
                count_right += 1

    return {
        "count_right": count_right,
        "count_wrong": count_wrong,
        "pred_one": pred_one,
        "count_one": count_one,
        "count_total": count_total,
        "final_seq": seq,
        "first_incorrect": first_incorrect,
        "num_unsat": num_unsat,
        "sum_unsat": sum_unsat,
    }


def create_and_filter_dataset(
    sequence_length, data_path=None, limit=None, partial_flat_len=4
):
    """
    Create dataset for the given sequence length (r).
    If data_path is set, load from pickle; otherwise call generate_dataset_sat
    with the given partial_flat_len (total bits in the random partial before SAT extension).
    Drops samples where the first row has length <= min_prefix_len - 1 (notebook used 3 → keep len >= 4).
    """
    if data_path and Path(data_path).exists():
        import pickle

        with open(data_path, "rb") as f:
            samples = pickle.load(f)
        print(f"[data] loaded {len(samples)} samples from {data_path}")
    else:
        if partial_flat_len < 4 or partial_flat_len >= 4 * sequence_length:
            raise ValueError(
                f"partial_flat_len must be in [4, 4*r), got {partial_flat_len} for r={sequence_length}"
            )
        print(
            f"[data] generating SAT dataset for sequence_length={sequence_length}, "
            f"partial_flat_len={partial_flat_len}..."
        )
        samples = generate_dataset_sat(
            sequence_length,
            partial_flat_len=partial_flat_len,
            limit=limit,
            max_partial_tries_per_seq=10,
            verbose=False
        )
        print(f"[data] generated {len(samples)} samples")

    # Filter short prefixes (same as notebook: len(item[0][0]) <= 3 removed)
    min_len = sequence_length // 2
    filtered = [item for item in samples if len(item[0][0]) >= min_len]
    dropped = len(samples) - len(filtered)
    if dropped:
        print(f"[data] dropped {dropped} samples with prefix length < {min_len}")
    return filtered
    

def run(
    sequence_length,
    data_path=None,
    limit=None,
    partial_flat_len=4,
    test_frac=0.2,
    epochs=30,
    batch_size=64,
    save_model_path=None,
    save_data_path=None,
):
    """
    Run full pipeline: create dataset, train with split, evaluate on test set.
    Returns (model, train_ds, test_ds, hist, eval_results, random_policy_results).

    If ``save_data_path`` is set, writes the filtered training samples (pickle).
    If ``save_model_path`` is set, writes five checkpoints from the halfway epoch
    onward (names ``<stem>_mid_01_epXXX.pt``, …) during training, then the final
    checkpoint at ``save_model_path`` after training.
    """
    samples = create_and_filter_dataset(
        sequence_length,
        data_path=data_path,
        limit=limit,
        partial_flat_len=partial_flat_len,
    )
    if not samples:
        raise ValueError(
            "No samples after filtering. Try a different sequence_length or data_path."
        )

    if save_data_path:
        save_samples_pickle(samples, save_data_path)

    train_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "test_frac": test_frac,
        "partial_flat_len": partial_flat_len,
        "limit": limit,
        "data_path": data_path,
    }
    model, (train_ds, test_ds), hist = train_with_split(
        samples,
        test_frac=test_frac,
        seed=42,
        epochs=epochs,
        batch_size=batch_size,
        mid_checkpoint_base=save_model_path,
        mid_checkpoint_r=sequence_length if save_model_path else None,
        mid_checkpoint_train_config=train_config if save_model_path else None,
    )

    eval_results = evaluate_policy(model, test_ds)
    random_policy_results = evaluate_random_policy(train_ds, test_ds)

    if save_model_path:
        save_policy_checkpoint(
            model,
            save_model_path,
            r=sequence_length,
            hist=hist,
            train_config=train_config,
        )

    return model, train_ds, test_ds, hist, eval_results, random_policy_results


def run_sat_evaluations(train_dataset, model, r):
    """
    Run both SAT-based evaluations:
      - evaluate_model_with_sat: model policy
      - evaluate_random_policy_with_sat: random policy using train label ratios

    Returns (model_sat_results, random_sat_results), each a dict.
    """
    model_sat_results = evaluate_model_with_sat(model, r=r)
    random_sat_results = evaluate_random_policy_with_sat(train_dataset, r=r)
    return model_sat_results, random_sat_results


def main():
    parser = argparse.ArgumentParser(
        description="Turyn policy transformer: train and evaluate with a given sequence length."
    )
    parser.add_argument(
        "sequence_length",
        type=int,
        help="Turyn sequence length (r) used for dataset generation and evaluation.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to pickle file with pre-generated samples (skips generate_dataset_sat).",
    )
    parser.add_argument(
        "--partial-flat-len",
        type=int,
        default=4,
        metavar="L",
        help=(
            "Total bits in the random partial before SAT extension; must satisfy 4 <= L < 4*r "
            "(default: 4). Ignored when loading from --data."
        ),
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on generated dataset size (unique partial+label samples).",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Save trained policy checkpoint (.pt) with state_dict and r for reload.",
    )
    parser.add_argument(
        "--save-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Save filtered (partial, next_bit) samples as pickle (same format as --data).",
    )
    args = parser.parse_args()

    model, train_ds, test_ds, hist, eval_results, random_policy_results = run(
        sequence_length=args.sequence_length,
        data_path=args.data,
        limit=args.limit,
        partial_flat_len=args.partial_flat_len,
        test_frac=args.test_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_model_path=args.save_model,
        save_data_path=args.save_data,
    )

    print("\n--- Evaluation (test set) ---")
    print(f"  loss: {eval_results['loss']:.4f}")
    print(f"  acc:  {eval_results['acc']:.4f}")
    print(f"  n:    {eval_results['n']}")

    print("\n--- Evaluation (random policy) ---")
    print(f"  acc: {random_policy_results['acc']:.4f}")
    print(f"  n:   {random_policy_results['n']}")

    model_sat_results, random_sat_sat_results = run_sat_evaluations(
            train_ds, model, r=args.sequence_length
    )

    print("\n--- SAT Evaluation (model policy) ---")
    print(f"  count_right:     {model_sat_results['count_right']}")
    print(f"  count_wrong:     {model_sat_results['count_wrong']}")
    print(f"  first_incorrect: {model_sat_results['first_incorrect']}")
    print(f"  num_unsat:       {model_sat_results['num_unsat']}")
    print(f"  sum_unsat:       {model_sat_results['sum_unsat']}")

    print("\n--- SAT Evaluation (random policy) ---")
    print(f"  count_right:     {random_sat_sat_results['count_right']}")
    print(f"  count_wrong:     {random_sat_sat_results['count_wrong']}")
    print(f"  first_incorrect: {random_sat_sat_results['first_incorrect']}")
    print(f"  num_unsat:       {random_sat_sat_results['num_unsat']}")
    print(f"  sum_unsat:       {random_sat_sat_results['sum_unsat']}")

if __name__ == "__main__":
    main()
