import itertools
import numpy as np
import random
from cnf_automator import generate_encoding, verify_assignment

# generate all possible sum arrays of length 4 equal to r^2
def generate_sums(r):
    target = 4 * r
    sums = []
    for a in range(1, r + 1, 2):
        a_s = a * a
        if a_s > target:
            break
        for b in range(a, r + 1, 2):
            b_s = b * b
            if a_s + b_s > target:
                break
            for c in range(b, r + 1, 2):
                c_s = c * c
                if a_s + b_s + c_s > target:
                    break
                for d in range(c, r + 1, 2):
                    d_s = d * d
                    if a_s + b_s + c_s + d_s == target:
                        sums.append((a, b, c, d))
                    if a_s + b_s + c_s + d_s > target:
                        break
    return sums

def is_turyn(r, arr):
    for lag in range(1, r):
        total = 0
        for i in range(4):
            total += np.dot(arr[i][:r - lag], arr[i][lag:])
        if (total != 0):
            return False
    return True


def filter(arrs, last_element_eq):
    new_arrs = []
    for arr in arrs:
        if arr[0] == 1 and (arr[0] == arr[-1]) == last_element_eq:
            new_arrs.append(arr)
    return new_arrs

# generate distinct sequences for specified length with given sums
def generate_turyns_sum(r, sums):
    # create ordered array with correct number of positives and negatives
    arr = []
    for i in range(4):
        cur = []
        negs = (r - sums[i]) // 2
        cur.extend([-1 for j in range(negs)])
        cur.extend([1 for j in range(r - negs)])
        arr.append(cur)
    
    # go through all possible combinations
    seqs = []
    a_list = list(set(itertools.permutations(arr[0])))
    b_list = list(set(itertools.permutations(arr[1])))
    c_list = list(set(itertools.permutations(arr[2])))
    d_list = list(set(itertools.permutations(arr[3])))

    # canonical form
    a_list = filter(a_list, False)
    b_list = filter(b_list, False)
    c_list = filter(c_list, True)
    d_list = filter(d_list, True)

    for a in a_list:
        for b in b_list:
            for c in c_list:
                for d in d_list:
                    new_arr = [a, b, c, d]
                    if is_turyn(r, new_arr):
                        seqs.append(new_arr)
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

def create_str(tup):
    s = ""
    for i in range(len(tup)):
        if tup[i] == 1:
            s += "1"
        else:
            s += "0"
    return s

def canonicalize_seq(seq):
    # reverse c and d
    reversed_c = seq[2][::-1]
    reversed_d = seq[3][::-1]
    if (create_str(reversed_c) < create_str(seq[2])):
        seq[2] = reversed_c
    if (create_str(reversed_d) < create_str(seq[3])):
        seq[3] = reversed_d
    # sort subsequences
    sorted(seq, key=create_str)
    return seq

def canonicalize_all_seqs(seqs):
    new_seqs = []
    for seq in seqs:
        new_seqs.append(canonicalize_seq(seq))
    return new_seqs

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

def get_partial(arr, r, row, col):
    cur_partial = []
    for i in range(4):
        end = col + 1
        if (i >= row):
            end = col
        cur_partial.append(arr[i][:end])
    return (cur_partial, arr[row][col])

def create_partials(arr, r):
    partials = []
    for i in range(r):
        for j in range(4):
            partials.append(get_partial(arr, r, j, i))
    return partials

def generate_partial_sequences(r):
    seqs = generate_all_turyns(r)
    new_seqs = canonicalize_all_seqs(seqs)
    partials = []
    for seq in new_seqs:
        partials.extend(create_partials(seq, r))
    return partials

def create_partial_str_next_bit(seq, next_bit):
    s = ""
    for i in range(4):
        cur_seq = seq[i]
        for j in range(len(cur_seq)):
            if cur_seq[j] == 1:
                s += "1"
            else:
                s += "0"
    if next_bit == 1:
        s += "1"
    else:
        s += "0"
    return s

def generate_dataset(r, limit=None):
    """
    Generate dataset of (partial_seq, next_bit) pairs for length r.

    - Iterates over all sums from generate_sums(r) via generate_all_turyns(r).
    - Deduplicates by partial prefix + next_bit.
    - If limit is set, stops once limit unique samples have been generated.

    Each full Turyn sequence contributes at most 4*r partials before deduplication.
    """
    dataset = []
    partial_set = set()

    seqs = generate_all_turyns(r)
    new_seqs = canonicalize_all_seqs(seqs)

    for seq in new_seqs:
        # Each full sequence yields up to 4*r partials (row-major over columns)
        for col in range(r):
            for row in range(4):
                partial_seq, next_bit = get_partial(seq, r, row, col)
                partial_str = create_partial_str_next_bit(partial_seq, next_bit)
                if partial_str in partial_set:
                    continue
                partial_set.add(partial_str)
                dataset.append((partial_seq, next_bit))
                if limit is not None and len(dataset) >= limit:
                    return dataset

    return dataset

def get_partial_only(arr, r, row, col):
    cur_partial = []
    for i in range(4):
        end = col + 1
        if (i > row):
            end = col
        cur_partial.append(arr[i][:end])
    return cur_partial

def create_partials_only(arr, r):
    partials = []
    for i in range(r):
        for j in range(4):
            partials.append(get_partial_only(arr, r, j, i))
    return partials

def generate_partial_sequences_only(r):
    seqs = generate_all_turyns(r)
    new_seqs = canonicalize_all_seqs(seqs)
    partials = []
    for seq in new_seqs:
        partials.extend(create_partials_only(seq, r))
    return partials

def create_partial_str(seq):
    s = ""
    for i in range(4):
        s += create_str(seq[i])
    return s

def write_seq(seqs, file):
    with open(file, 'w') as fin:
        for seq in seqs:
            for i in range(4):
                for item in seq[i]:
                    fin.write(str(item) + " ")
                fin.write("; ")
            fin.write("\n")
    
def read_seq(file):
    with open(file, 'r') as fin:
        seqs = []
        line = fin.readline()
        while line:
            cur_line = line.strip().split(";")[:-1]
            cur_seq = []
            for item in cur_line:
                cur_seq.append(tuple(map(int, item.strip().split())))
            seqs.append(cur_seq)
            line = fin.readline()
    return seqs

def generate_dataset_only(r):
    ## Get rid of empty sequence, account for repeat sequences
    dataset = []
    partials = generate_partial_sequences_only(r)
    partial_set = set()
    for partial_seq in partials:
        partial_str = create_partial_str(partial_seq)
        if partial_str not in partial_set:
            partial_set.add(partial_str)
            dataset.append(partial_seq)
    return dataset

def get_invalid_from_partial(partial, r, cnf, varmap, limit):
    count = 0
    invalid_partials = []
    for i in range(1, r - 1):
        for j in range(4):
            if (len(partial[j]) <= i):
                continue
            invalid_partial = list(map(list, partial))
            invalid_partial[j][i] *= -1
            invalid_partial = list(map(tuple, invalid_partial))
            sat = verify_assignment(invalid_partial, varmap, cnf)
            if not sat:
                count += 1
                invalid_partials.append(invalid_partial)
                if (count == limit):
                    return invalid_partials
    return None

def generate_partial_values(r, sats):
    limit = 2
    all_invalid_partials = []
    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    for partial in sats:
        invalid_partials = get_invalid_from_partial(partial, r, cnf, varmap, limit)
        if invalid_partials:
            all_invalid_partials.extend(invalid_partials)
    return all_invalid_partials
    

# generate values of partial sequences
def generate_turyn_values(r, sat_file, unsat_file):
    sats = generate_dataset_only(r)
    unsats = generate_partial_values(r, sats)
    write_seq(sats, sat_file)
    write_seq(unsats, unsat_file)