from pysat.solvers import Minisat22
from sequence_generator import read_seq, create_partial_str
from cnf_automator import generate_encoding, make_assumptions
import json

def evaluate_prefix_with_core(prefix, r, cnf, varmap):
    """
    Evaluate a prefix and return SAT result and UNSAT core if applicable
    """
    # Build assignment dictionary
    assign = {}
    seqs = ['X', 'Y', 'Z', 'W']

    for seq_idx, seq_name in enumerate(seqs):
        for i in range(len(prefix[seq_idx])):
            assign[(seq_name, i)] = prefix[seq_idx][i]

    # Create assumptions
    assumptions = make_assumptions(varmap, assign)

    # Run solver with UNSAT core extraction enabled
    with Minisat22(bootstrap_with=cnf) as solver:
        sat = solver.solve(assumptions=assumptions)

        result = {
            'sat': sat,
            'unsat_core': None,
            'unsat_core_assumptions': None
        }

        if not sat:
            # Get UNSAT core (subset of assumptions that caused UNSAT)
            core = solver.get_core()
            if core:
                result['unsat_core'] = core
                # Map back to variable assignments
                core_assignments = []
                for lit in core:
                    abs_lit = abs(lit)
                    # Find which variable this corresponds to
                    for key, var_id in varmap.items():
                        if var_id == abs_lit:
                            value = 1 if lit > 0 else -1
                            core_assignments.append((key, value))
                            break
                result['unsat_core_assumptions'] = core_assignments

        return result

def format_prefix_compact(prefix):
    """Format prefix in a compact string format"""
    parts = []
    for seq in prefix:
        if len(seq) == 0:
            parts.append("[]")
        else:
            seq_str = "[" + ",".join(str(x) for x in seq) + "]"
            parts.append(seq_str)
    return " | ".join(parts)

def evaluate_all_candidates(r, input_file, output_file):
    """Evaluate all candidate prefixes from file"""

    print(f"Loading candidates from {input_file}...")
    candidates = read_seq(input_file)
    print(f"Loaded {len(candidates)} candidates\n")

    print("Generating CNF encoding...")
    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    print(f"CNF has {len(cnf.clauses)} clauses and {cnf.nv} variables\n")

    results = []

    print("="*80)
    print(f"Evaluating {len(candidates)} LLM-Generated Prefixes (r={r})")
    print("="*80)

    for idx, candidate in enumerate(candidates, 1):
        print(f"\nCandidate {idx}:")
        print(f"  Prefix: {format_prefix_compact(candidate)}")

        # Evaluate with SAT solver
        result = evaluate_prefix_with_core(candidate, r, cnf, varmap)

        sat_status = "SAT" if result['sat'] else "UNSAT"
        print(f"  Result: {sat_status}")

        if not result['sat']:
            core = result['unsat_core']
            core_assignments = result['unsat_core_assumptions']

            if core:
                print(f"  UNSAT Core Size: {len(core)} assumptions")
                print(f"  UNSAT Core Variables: {core_assignments[:10]}")  # Show first 10
                if len(core_assignments) > 10:
                    print(f"    ... and {len(core_assignments) - 10} more")

        results.append({
            'id': idx,
            'prefix': candidate,
            'prefix_str': format_prefix_compact(candidate),
            'sat': result['sat'],
            'unsat_core_size': len(result['unsat_core']) if result['unsat_core'] else 0,
            'unsat_core': result['unsat_core'],
            'unsat_core_assignments': result['unsat_core_assumptions']
        })

    # Write results to file
    print("\n" + "="*80)
    print("Writing results to file...")
    print("="*80 + "\n")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"LLM-Generated Prefix Evaluation Results (r={r})\n")
        f.write("="*80 + "\n\n")

        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'ID':<4} {'Result':<8} {'Core Size':<12} {'Prefix'}\n")
        f.write("-"*80 + "\n")

        for res in results:
            core_size_str = str(res['unsat_core_size']) if not res['sat'] else "N/A"
            f.write(f"{res['id']:<4} {('SAT' if res['sat'] else 'UNSAT'):<8} "
                   f"{core_size_str:<12} {res['prefix_str']}\n")

        # Detailed results
        f.write("\n\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")

        for res in results:
            f.write(f"Candidate {res['id']}\n")
            f.write(f"  Prefix: {res['prefix_str']}\n")
            f.write(f"  Full:   {res['prefix']}\n")
            f.write(f"  Result: {'SAT' if res['sat'] else 'UNSAT'}\n")

            if not res['sat']:
                f.write(f"  UNSAT Core Size: {res['unsat_core_size']} assumptions\n")
                if res['unsat_core']:
                    f.write(f"  UNSAT Core Literals: {res['unsat_core']}\n")
                if res['unsat_core_assignments']:
                    f.write(f"  UNSAT Core Assignments:\n")
                    for var, val in res['unsat_core_assignments']:
                        f.write(f"    {var} = {val}\n")

            f.write("\n" + "-"*80 + "\n\n")

        # Statistics
        sat_count = sum(1 for r in results if r['sat'])
        unsat_count = len(results) - sat_count

        f.write("="*80 + "\n")
        f.write("STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Total Candidates: {len(results)}\n")
        f.write(f"SAT:              {sat_count} ({sat_count/len(results)*100:.1f}%)\n")
        f.write(f"UNSAT:            {unsat_count} ({unsat_count/len(results)*100:.1f}%)\n")

        if unsat_count > 0:
            avg_core_size = sum(r['unsat_core_size'] for r in results if not r['sat']) / unsat_count
            f.write(f"Avg UNSAT Core Size: {avg_core_size:.1f} assumptions\n")

    print(f"Results written to {output_file}")

    # Also save as JSON for easier processing
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        # Convert tuples to lists for JSON serialization
        json_results = []
        for res in results:
            json_res = res.copy()
            json_res['prefix'] = [list(seq) for seq in res['prefix']]
            json_results.append(json_res)
        json.dump(json_results, f, indent=2)

    print(f"JSON results written to {json_file}\n")

    # Print summary
    print("SUMMARY:")
    print(f"  SAT:   {sat_count}/{len(results)}")
    print(f"  UNSAT: {unsat_count}/{len(results)}")

    return results

if __name__ == "__main__":
    r = 5
    input_file = f"llm_generated_prefixes_{r}.txt"
    output_file = f"llm_prefix_eval_results_{r}.txt"

    results = evaluate_all_candidates(r, input_file, output_file)
