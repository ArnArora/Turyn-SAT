"""
Evaluate Claude-Generated Prefixes with SAT Solver

Tests real LLM (Claude) ability to generate valid Turyn prefixes.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from cnf_automator import generate_encoding, verify_assignment, make_assumptions
from pysat.solvers import Minisat22


def evaluate_prefix_with_core(prefix, r, cnf, varmap):
    """Evaluate a prefix and extract UNSAT core if invalid."""
    # Build assignment from prefix
    assign = {}
    for seq_idx, seq_name in enumerate(['X', 'Y', 'Z', 'W']):
        for i in range(len(prefix[seq_idx])):
            assign[(seq_name, i)] = prefix[seq_idx][i]

    # Create assumptions for SAT solver
    assumptions = make_assumptions(varmap, assign)

    # Test satisfiability
    with Minisat22(bootstrap_with=cnf) as solver:
        sat = solver.solve(assumptions=assumptions)

        result = {
            'prefix': prefix,
            'sat': sat
        }

        if not sat:
            # Extract UNSAT core
            core = solver.get_core()

            # Map core back to assignments
            core_assignments = []
            for lit in core:
                # Find which variable this corresponds to
                for key, var_id in varmap.items():
                    if var_id == abs(lit):
                        value = 1 if lit > 0 else -1
                        core_assignments.append((key, value))
                        break

            result['unsat_core_size'] = len(core)
            result['unsat_core_assignments'] = core_assignments

        return result


def main():
    print("="*80)
    print("EVALUATING CLAUDE-GENERATED PREFIXES")
    print("="*80)

    # Load Claude-generated prefixes
    input_file = "../intermediate_results/claude_generated_prefixes_5.json"

    with open(input_file, 'r') as f:
        data = json.load(f)

    prefixes = data['prefixes']
    r = data['r']

    print(f"\nLoaded {len(prefixes)} Claude-generated prefixes (N={r})")

    # Generate CNF encoding
    print("\nGenerating SAT encoding...")
    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    print(f"  CNF has {cnf.nv} variables")

    # Evaluate each prefix
    print("\nEvaluating prefixes with SAT solver...")
    results = []

    for i, prefix in enumerate(prefixes, 1):
        print(f"\n  Evaluating Prefix {i}...")
        result = evaluate_prefix_with_core(prefix, r, cnf, varmap)
        results.append(result)

        if result['sat']:
            print(f"    Result: SAT [+]")
        else:
            core_size = result['unsat_core_size']
            print(f"    Result: UNSAT (core size: {core_size})")

    # Summary statistics
    sat_count = sum(1 for r in results if r['sat'])
    unsat_count = len(results) - sat_count
    sat_rate = (sat_count / len(results)) * 100

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total prefixes: {len(results)}")
    print(f"SAT: {sat_count} ({sat_rate:.1f}%)")
    print(f"UNSAT: {unsat_count} ({100-sat_rate:.1f}%)")

    if unsat_count > 0:
        avg_core_size = sum(r.get('unsat_core_size', 0) for r in results if not r['sat']) / unsat_count
        print(f"Average UNSAT core size: {avg_core_size:.1f}")

    # Save results
    output_file = "../intermediate_results/claude_prefix_eval_results_5.json"
    with open(output_file, 'w') as f:
        json.dump({
            'r': r,
            'total': len(results),
            'sat_count': sat_count,
            'unsat_count': unsat_count,
            'sat_rate': sat_rate,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Save detailed text report
    output_txt = "../intermediate_results/claude_prefix_eval_results_5.txt"
    with open(output_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLAUDE-GENERATED PREFIX EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"Prefix {i}:\n")
            prefix = result['prefix']
            for j, seq in enumerate(prefix):
                seq_name = ['A', 'B', 'C', 'D'][j]
                if len(seq) > 0:
                    formatted = ' '.join(['+1' if x == 1 else '-1' for x in seq])
                else:
                    formatted = '(empty)'
                f.write(f"  {seq_name}: [{formatted}]\n")

            if result['sat']:
                f.write("  Result: SAT [+]\n\n")
            else:
                f.write(f"  Result: UNSAT\n")
                f.write(f"  UNSAT Core Size: {result['unsat_core_size']}\n")
                f.write(f"  Core Assignments:\n")
                for key, value in result.get('unsat_core_assignments', [])[:10]:  # Show first 10
                    f.write(f"    {key} = {value}\n")
                f.write("\n")

        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total: {len(results)}\n")
        f.write(f"SAT: {sat_count} ({sat_rate:.1f}%)\n")
        f.write(f"UNSAT: {unsat_count}\n")

    print(f"Detailed report saved to: {output_txt}")

    print("\n" + "="*80)
    print("NEXT STEP: If there are UNSAT prefixes, run Claude-based repair")
    print("="*80)


if __name__ == "__main__":
    main()
