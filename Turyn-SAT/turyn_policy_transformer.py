"""
Turyn policy transformer: dataset creation, supervised learning with train/test split,
and evaluation. Run with a sequence length (r) to get supervised learning and evaluation results.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure local imports work when run from repo root or script dir
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sequence_generator import generate_dataset

PAD_ID = 2


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


def train_with_split(
    samples,
    test_frac=0.2,
    test_samples=None,
    seed=42,
    epochs=30,
    batch_size=64,
    lr=3e-4,
    device=None,
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


def create_and_filter_dataset(sequence_length, data_path=None):
    """
    Create dataset for the given sequence length (r).
    If data_path is set, load from pickle; otherwise call generate_dataset(sequence_length).
    Drops samples where the first row has length <= min_prefix_len - 1 (notebook used 3 → keep len >= 4).
    """
    if data_path and Path(data_path).exists():
        import pickle

        with open(data_path, "rb") as f:
            samples = pickle.load(f)
        print(f"[data] loaded {len(samples)} samples from {data_path}")
    else:
        print(f"[data] generating dataset for sequence_length={sequence_length}...")
        samples = generate_dataset(sequence_length)
        print(f"[data] generated {len(samples)} samples")

    # Filter short prefixes (same as notebook: len(item[0][0]) <= 3 removed)
    min_len = sequence_length // 2
    filtered = [item for item in samples if len(item[0][0]) >= min_len]
    dropped = len(samples) - len(filtered)
    if dropped:
        print(f"[data] dropped {dropped} samples with prefix length < {min_len}")
    return filtered


def run(sequence_length, data_path=None, test_frac=0.2, epochs=30, batch_size=64):
    """
    Run full pipeline: create dataset, train with split, evaluate on test set.
    Returns (model, train_ds, test_ds, hist, eval_results).
    """
    samples = create_and_filter_dataset(sequence_length, data_path=data_path)
    if not samples:
        raise ValueError(
            "No samples after filtering. Try a different sequence_length or data_path."
        )

    model, (train_ds, test_ds), hist = train_with_split(
        samples,
        test_frac=test_frac,
        seed=42,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Evaluation on the test split (test_ds is a Subset from random_split)
    eval_results = evaluate_policy(model, test_ds)

    return model, train_ds, test_ds, hist, eval_results


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
        help="Optional path to pickle file with pre-generated samples (skips generate_dataset).",
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
    args = parser.parse_args()

    model, train_ds, test_ds, hist, eval_results = run(
        sequence_length=args.sequence_length,
        data_path=args.data,
        test_frac=args.test_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n--- Evaluation (test set) ---")
    print(f"  loss: {eval_results['loss']:.4f}")
    print(f"  acc:  {eval_results['acc']:.4f}")
    print(f"  n:    {eval_results['n']}")


if __name__ == "__main__":
    main()
