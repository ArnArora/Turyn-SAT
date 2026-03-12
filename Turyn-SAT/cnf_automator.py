from pysat.formula import CNF, IDPool
from pysat.solvers import Solver, Minisat22
from pysat.card import CardEnc, EncType

# generate CNF encoding given length and specific lag
def generate_encoding_lag(r, lag, cnf, pool, varmap, pmap_lags, seqs):
    # add variable to map
    def seq_id(seq, i):
        key = f'{seq}_{i}'
        if key not in varmap:
            varmap[key] = pool.id(key)
        return varmap[key]

    # create auxiliary variable for x * y
    def add_reified_xnor(p, x, y):
        cnf.append([-p, -x,  y])
        cnf.append([-p,  x, -y])
        cnf.append([ p, -x, -y])
        cnf.append([ p,  x,  y])

    # for each sequence, add clause for new combination
    pmap = {}
    p_lits = []
    for seq in seqs:
        for i in range(r - lag):
            j = i + lag
            seq_i = seq_id(seq, i)
            seq_j = seq_id(seq, j)
            p_key = f'p_{seq}_{i}_{j}'
            p_id = pool.id(p_key)
            pmap[p_key] = p_id
            p_lits.append(p_id)
            add_reified_xnor(p_id, seq_i, seq_j)

    # verify sum is 0
    eq = CardEnc.equals(lits=p_lits, bound=2 * (r - lag), encoding=EncType.totalizer, vpool=pool)
    cnf.extend(eq.clauses)
    pmap_lags[lag] = pmap

# generate CNF encoding given length
def generate_encoding(r):
    cnf = CNF()
    pool = IDPool()
    varmap = {}
    pmap_lags = {}
    seqs = ['X', 'Y', 'Z', 'W']

    for lag in range(1, r):
        generate_encoding_lag(r, lag, cnf, pool, varmap, pmap_lags, seqs)
        
    return cnf, varmap, pmap_lags, pool

def make_assumptions(varmap, assign):
    lits = []
    for (seq, i), val in assign.items():
        v = varmap[f'{seq}_{i}']
        lits.append(v if val == 1 else -v)
    return lits

def verify_assignment(assignment, varmap, cnf):
    assign = {}
    seqs = ['X', 'Y', 'Z', 'W']

    seq_num = 0
    for seq in seqs:
        for i in range(len(assignment[seq_num])):
            assign[(seq, i)] = assignment[seq_num][i]
        seq_num += 1

    with Minisat22(bootstrap_with=cnf) as solver:
        sat = solver.solve(assumptions=make_assumptions(varmap, assign))
        core = solver.get_core()
        return sat, core
