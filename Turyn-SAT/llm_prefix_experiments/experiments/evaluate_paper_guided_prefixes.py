"""
Evaluate theory-guided prefixes from paper_guided_prefixes_5.txt and paper_guided_prefixes_7.txt
Using the same SAT solver pipeline as the LLM-generated prefix evaluation.
"""

from sequence_generator import read_seq
from cnf_automator import generate_encoding
from evaluate_llm_candidates import evaluate_prefix_with_core, format_prefix_compact
import json

def evaluate_paper_guided_prefixes(r, input_file, output_file):
    """Evaluate all paper-guided prefixes from file"""

    print(f"\n{'='*80}")
    print(f"Evaluating Paper-Guided Prefixes for N={r}")
    print(f"{'='*80}\n")

    print(f"Loading prefixes from {input_file}...")
    candidates = read_seq(input_file)
    print(f"Loaded {len(candidates)} candidates\n")

    print("Generating CNF encoding...")
    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    print(f"CNF has {len(cnf.clauses)} clauses and {cnf.nv} variables\n")

    results = []

    print("="*80)
    print(f"Evaluating {len(candidates)} Theory-Guided Prefixes (r={r})")
    print("="*80)

    for idx, candidate in enumerate(candidates, 1):
        print(f"\nPrefix {idx}:")
        print(f"  Sequences: {format_prefix_compact(candidate)}")

        # Evaluate with SAT solver
        result = evaluate_prefix_with_core(candidate, r, cnf, varmap)

        sat_status = "SAT" if result['sat'] else "UNSAT"
        print(f"  Result: {sat_status}")

        if not result['sat']:
            core = result['unsat_core']
            core_assignments = result['unsat_core_assumptions']

            if core:
                print(f"  UNSAT Core Size: {len(core)} assumptions")
                print(f"  UNSAT Core Variables: {core_assignments[:10]}")
                if len(core_assignments) > 10:
                    print(f"    ... and {len(core_assignments) - 10} more")

        results.append({
            'id': idx,
            'prefix': candidate,
            'prefix_str': format_prefix_compact(candidate),
            'sat': result['sat'],
            'unsat_core_size': len(result.get('unsat_core', [])) if result.get('unsat_core') else 0,
            'unsat_core': result.get('unsat_core'),
            'unsat_core_assignments': result.get('unsat_core_assumptions')
        })

    # Write results to file
    print("\n" + "="*80)
    print("Writing results to file...")
    print("="*80 + "\n")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Theory-Guided Prefix Evaluation Results (r={r})\n")
        f.write(f"Source: paper_guided_prefixes_{r}.txt\n")
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
            f.write(f"Prefix {res['id']}\n")
            f.write(f"  Sequences: {res['prefix_str']}\n")
            f.write(f"  Full:      {res['prefix']}\n")
            f.write(f"  Result:    {'SAT' if res['sat'] else 'UNSAT'}\n")

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
        f.write(f"Total Prefixes:   {len(results)}\n")
        f.write(f"SAT:              {sat_count} ({sat_count/len(results)*100:.1f}%)\n")
        f.write(f"UNSAT:            {unsat_count} ({unsat_count/len(results)*100:.1f}%)\n")

        if unsat_count > 0:
            avg_core_size = sum(r['unsat_core_size'] for r in results if not r['sat']) / unsat_count
            f.write(f"Avg UNSAT Core Size: {avg_core_size:.1f} assumptions\n")

    print(f"Results written to {output_file}\n")

    # Print summary
    sat_count = sum(1 for r in results if r['sat'])
    unsat_count = len(results) - sat_count
    print("SUMMARY:")
    print(f"  SAT:   {sat_count}/{len(results)} ({sat_count/len(results)*100:.1f}%)")
    print(f"  UNSAT: {unsat_count}/{len(results)} ({unsat_count/len(results)*100:.1f}%)")

    return results

def main():
    print("="*80)
    print("EVALUATING THEORY-GUIDED TURYN PREFIXES")
    print("="*80)

    # Evaluate N=5 prefixes
    results_n5 = evaluate_paper_guided_prefixes(
        r=5,
        input_file="paper_guided_prefixes_5.txt",
        output_file="paper_prefix_eval_results_5.txt"
    )

    # Evaluate N=7 prefixes
    results_n7 = evaluate_paper_guided_prefixes(
        r=7,
        input_file="paper_guided_prefixes_7.txt",
        output_file="paper_prefix_eval_results_7.txt"
    )

    # Save combined JSON results
    print("\n" + "="*80)
    print("Writing combined JSON results...")
    print("="*80 + "\n")

    combined_results = {
        "N5": [],
        "N7": []
    }

    # Convert N=5 results
    for res in results_n5:
        json_res = res.copy()
        json_res['prefix'] = [list(seq) for seq in res['prefix']]
        combined_results["N5"].append(json_res)

    # Convert N=7 results
    for res in results_n7:
        json_res = res.copy()
        json_res['prefix'] = [list(seq) for seq in res['prefix']]
        combined_results["N7"].append(json_res)

    with open("paper_prefix_eval_results.json", 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("JSON results written to paper_prefix_eval_results.json\n")

    # Overall statistics
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    total_sat_n5 = sum(1 for r in results_n5 if r['sat'])
    total_sat_n7 = sum(1 for r in results_n7 if r['sat'])
    total_sat = total_sat_n5 + total_sat_n7
    total_prefixes = len(results_n5) + len(results_n7)

    print(f"N=5: {total_sat_n5}/{len(results_n5)} SAT ({total_sat_n5/len(results_n5)*100:.1f}%)")
    print(f"N=7: {total_sat_n7}/{len(results_n7)} SAT ({total_sat_n7/len(results_n7)*100:.1f}%)")
    print(f"Overall: {total_sat}/{total_prefixes} SAT ({total_sat/total_prefixes*100:.1f}%)")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  - paper_prefix_eval_results_5.txt")
    print("  - paper_prefix_eval_results_7.txt")
    print("  - paper_prefix_eval_results.json")
    print("\nReady for UNSAT-core-guided repair in next experiment.")

if __name__ == "__main__":
    main()
