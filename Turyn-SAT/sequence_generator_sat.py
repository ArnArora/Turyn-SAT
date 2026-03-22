"""
SAT-guided Turyn dataset generation: same (partial, next_bit) format as
sequence_generator.generate_dataset, but completions come from a SAT solver.

Partials from create_partials exclude the rollout seed state: four rows each
length 1 with value +1 (matches the fixed initial column in random_partial /
_add_bit_to_seq).
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Sequence, Tuple

from cnf_automator import generate_encoding, verify_assignment
from sequence_generator import (
    create_str,
    create_partials,
    create_partial_str_next_bit,
)

# Rollout seed: one +1 per row before any free choices (see turyn_policy_transformer).
RolloutPartial = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]
PartialNext = Tuple[RolloutPartial, int]


def next_row_for_bit(seq: Sequence[Sequence[int]]) -> int:
    first_len = len(seq[0])
    for i in range(4):
        if len(seq[i]) < first_len:
            return i
    return 0


def add_bit_to_seq(seq: List[List[int]], bit: int) -> None:
    row = next_row_for_bit(seq)
    seq[row].append(bit)


def flat_len(seq: Sequence[Sequence[int]]) -> int:
    return sum(len(row) for row in seq)


def seq_to_tuples(seq: Sequence[Sequence[int]]) -> RolloutPartial:
    return tuple(tuple(row) for row in seq)


def canonicalize_seq(seq):
    # reverse c and d
    reversed_index_one = -1
    reversed_index_two = -1
    for i in range(4):
        subseq = seq[i]
        if subseq[0] == subseq[-1]:
            if reversed_index_one == -1:
                reversed_index_one = i
            else:
                reversed_index_two = i
                break
    if reversed_index_one != -1 and reversed_index_two != -1:
        reversed_subseq_one = seq[reversed_index_one][::-1]
        reversed_subseq_two = seq[reversed_index_two][::-1]
        if create_str(reversed_subseq_one) < create_str(seq[reversed_index_one]):
            seq[reversed_index_one] = reversed_subseq_one
        if create_str(reversed_subseq_two) < create_str(seq[reversed_index_two]):
            seq[reversed_index_two] = reversed_subseq_two
    # sort subsequences
    seq = sorted(seq, key=lambda x: abs(sum(x)))
    return seq


def random_partial_sequence(
    r: int,
    partial_flat_len: int,
    rng: random.Random | None = None,
) -> RolloutPartial:
    """
    Random partial: four rows each start with +1, then random ±1 bits until
    sum(len(row)) == partial_flat_len.
    """
    rng = rng or random
    if partial_flat_len < 4 or partial_flat_len >= 4 * r:
        raise ValueError(f"partial_flat_len must be in [4, 4*r), got {partial_flat_len}")
    seq = [[1], [1], [1], [1]]
    while flat_len(seq) < partial_flat_len:
        add_bit_to_seq(seq, 1 if rng.random() < 0.5 else -1)
    return seq_to_tuples(seq)


def extend_to_full_turyn_sat(
    partial_tuples: RolloutPartial,
    r: int,
    varmap,
    cnf,
    rng: random.Random | None = None,
    verbose: bool = False,
) -> Optional[RolloutPartial]:
    """
    Greedy SAT extension using the same append order as random_partial / rollout.
    Returns four tuples of length r, or None.
    """
    rng = rng or random
    seq = [list(t) for t in partial_tuples]
    if any(len(row) > r for row in seq):
        return None
    step = 0
    while any(len(row) < r for row in seq):
        row = next_row_for_bit(seq)
        sat_by_bit: dict[int, bool] = {}
        for bit in (-1, 1):
            trial = copy.deepcopy(seq)
            trial[row].append(bit)
            if any(len(x) > r for x in trial):
                sat_by_bit[bit] = False
            else:
                sat_by_bit[bit], _ = verify_assignment(
                    seq_to_tuples(trial), varmap, cnf
                )
        if verbose:
            print(
                f"step {step}: append to row {row} | "
                f"sat(-1)={sat_by_bit[-1]} sat(+1)={sat_by_bit[1]}"
            )
            step += 1
        order = [-1, 1]
        rng.shuffle(order)
        chosen = None
        for bit in order:
            if sat_by_bit.get(bit):
                chosen = bit
                break
        if chosen is None:
            return None
        seq[row].append(chosen)
    return seq_to_tuples(seq)


def generate_dataset_sat(
    r: int,
    partial_flat_len: int,
    max_partial_tries_per_seq: int = 80,
    limit: Optional[int] = None,
    rng: random.Random | None = None,
    verbose: bool = False,
) -> List[PartialNext]:
    """
    SAT-complete random partials -> canonical full sequences -> partials
    (excluding seed-only partial), deduped like generate_dataset.

    If ``limit`` is set, stop once the dataset contains that many unique
    partial+next-bit samples (may exit before ``num_successful_sequences``).
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    rng = rng or random
    cnf, varmap, _, _ = generate_encoding(r)
    dataset: List[PartialNext] = []
    partial_set: set[str] = set()
    successes = 0
    sum_count = {}

    while limit is None or len(dataset) < limit:
        full: Optional[RolloutPartial] = None
        for _ in range(max_partial_tries_per_seq):
            p0 = random_partial_sequence(r, partial_flat_len, rng=rng)
            full = extend_to_full_turyn_sat(p0, r, varmap, cnf, rng=rng)
            if full is not None:
                break
        if full is None:
            if verbose:
                print(
                    f"Stopped: no completion after {max_partial_tries_per_seq} tries "
                    f"(successes={successes})"
                )
            break

        canon = [list(row) for row in full]
        canon_seq = canonicalize_seq(canon)
        if verbose:
            print("Canonicalizing sequence:", canon_seq)
            if tuple([sum(subseq) for subseq in canon_seq]) not in sum_count:
                sum_count[tuple([sum(subseq) for subseq in canon_seq])] = 0
            sum_count[tuple([sum(subseq) for subseq in canon_seq])] += 1
        canon_tuples = seq_to_tuples(canon_seq)
        new_pairs = create_partials(canon_tuples, r)
        added = 0
        for partial in new_pairs:
            partial_seq, next_bit = partial
            key = create_partial_str_next_bit(partial_seq, next_bit)
            if key not in partial_set:
                partial_set.add(key)
                dataset.append(partial)
                added += 1
                if limit is not None and len(dataset) >= limit:
                    break
        successes += 1
        if verbose:
            print(
                f"sequence {successes}: new_unique_pairs={added}, "
                f"dataset_size={len(dataset)}"
            )
            print(sum_count)
            print(len(sum_count))
            print(max(sum_count.values()))
    return dataset
