import itertools
import numpy as np
import random

# generate all possible sum arrays of length 4 equal to r^2
def generate_sums(r):
    target = 4 * r
    sums = []
    for a in range(1, r + 1, 2):
        a_s = a * a
        for b in range(a, r + 1, 2):
            b_s = b * b
            for c in range(b, r + 1, 2):
                c_s = c * c
                for d in range(c, r + 1, 2):
                    d_s = d * d
                    if a_s + b_s + c_s + d_s == target:
                        sums.append((a, b, c, d))
    return sums

def is_turyn(r, arr):
    for lag in range(1, r):
        total = 0
        for i in range(4):
            total += np.dot(arr[i][:r - lag], arr[i][lag:])
        if (total != 0):
            return False
    return True


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