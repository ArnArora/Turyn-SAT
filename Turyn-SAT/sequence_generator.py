import itertools
import numpy as np
import random

# generate all possible sum arrays of length 4 equal to r^2
def generate_sums(r):
    target = 4 * r
    # OPTIMIZATION: precompute odd candidates and square->value map; remove inner d-loop
    odds = list(range(1, r + 1, 2))
    sq_to_val = {x * x: x for x in odds}

    sums = []
    for ai, a in enumerate(odds):
        a_s = a * a
        for bi, b in enumerate(odds[ai:], start=ai):  # ensures a <= b
            b_s = b * b
            for c in odds[bi:]:                        # ensures b <= c
                needed = target - (a_s + b_s + c * c)
                d = sq_to_val.get(needed)
                if d is not None and d >= c and d <= r:
                    sums.append((a, b, c, d))
    return sums

def is_turyn(r, arr):
    # OPTIMIZATION: one-time conversion to int32 arrays
    A0 = np.asarray(arr[0], dtype=np.int32)
    A1 = np.asarray(arr[1], dtype=np.int32)
    A2 = np.asarray(arr[2], dtype=np.int32)
    A3 = np.asarray(arr[3], dtype=np.int32)

    for lag in range(1, r):
        n = r - lag
        total = int(
            np.dot(A0[:n], A0[lag:]) +
            np.dot(A1[:n], A1[lag:]) +
            np.dot(A2[:n], A2[lag:]) +
            np.dot(A3[:n], A3[lag:])
        )
        if total != 0:
            return False
    return True

# OPTIMIZATION: generate unique permutations lazily without materializing n! or using set
def _unique_permutations(seq):
    seq = sorted(seq)
    used = [False] * len(seq)
    perm = []

    def backtrack():
        if len(perm) == len(seq):
            yield tuple(perm)
            return
        prev = None
        for i, val in enumerate(seq):
            if used[i] or val == prev:
                continue
            used[i] = True
            perm.append(val)
            yield from backtrack()
            perm.pop()
            used[i] = False
            prev = val

    yield from backtrack()


# generate distinct sequences for specified length with given sums
def generate_turyns_sum(r, sums):
    # create ordered array with correct number of positives and negatives
    arr = []
    for i in range(4):
        negs = (r - sums[i]) // 2
        # OPTIMIZATION: list multiply instead of repeated extends
        cur = [-1] * negs + [1] * (r - negs)
        arr.append(cur)
    
    # OPTIMIZATION: cheap necessary-condition prune at lag=1
    def _lag1_zero(aA, bA, cA, dA):
        n = len(aA) - 1
        if n <= 0:
            return True
        return int(
            np.dot(aA[:n], aA[1:]) +
            np.dot(bA[:n], bA[1:]) +
            np.dot(cA[:n], cA[1:]) +
            np.dot(dA[:n], dA[1:])
        ) == 0

    # go through all possible combinations lazily
    seqs = []
    for a in _unique_permutations(arr[0]):
        aA = np.asarray(a, dtype=np.int32)
        for b in _unique_permutations(arr[1]):
            bA = np.asarray(b, dtype=np.int32)
            for c in _unique_permutations(arr[2]):
                cA = np.asarray(c, dtype=np.int32)
                for d in _unique_permutations(arr[3]):
                    dA = np.asarray(d, dtype=np.int32)

                    if not _lag1_zero(aA, bA, cA, dA):
                        continue

                    if is_turyn(r, (aA, bA, cA, dA)):
                        seqs.append([a, b, c, d])
    return seqs

# generate distinct sequences for specified length
def generate_all_turyns(r):
    # generate all possible sums
    sums = generate_sums(r)
    seqs = []
    
    # generate sequences for each sum
    for sum_arr in sums:
        seqs.extend(generate_turyns_sum(r, sum_arr))
    
    return seqs

def generate_invalid_turyn(r):
    count = 0
    while count < 10:
        seq = []
        for i in range(4):
            cur_seq = []
            for j in range(r):
                if random.random() < 0.5:
                    cur_seq.append(1)
                else:
                    cur_seq.append(-1)
            seq.append(tuple(cur_seq))
        if not is_turyn(r, seq):
            return seq
        count += 1