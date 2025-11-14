"""
Repair UNSAT theory-guided prefixes using UNSAT-core feedback.
Uses the same repair pipeline as correct_unsat_prefixes.py
"""

import json
from cnf_automator import generate_encoding
from correct_unsat_prefixes import smart_correct_prefix
from evaluate_llm_candidates import format_prefix_compact
from sequence_generator import write_seq

def load_unsat_prefixes_from_json(json_file):
    """Load UNSAT prefixes from evaluation results JSON"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    n5_unsat = []
    n7_unsat = []

    # Extract N=5 UNSAT prefixes
    for result in data['N5']:
        if not result['sat']:
            n5_unsat.append({
                'id': result['id'],
                'prefix': [tuple(seq) for seq in result['prefix']],
                'prefix_str': result['prefix_str'],
                'unsat_core_assignments': result['unsat_core_assignments']
            })

    # Extract N=7 UNSAT prefixes
    for result in data['N7']:
        if not result['sat']:
            n7_unsat.append({
                'id': result['id'],
                'prefix': [tuple(seq) for seq in result['prefix']],
                'prefix_str': result['prefix_str'],
                'unsat_core_assignments': result['unsat_core_assignments']
            })

    return n5_unsat, n7_unsat

def correct_paper_guided_prefixes(r, unsat_list, output_file):
    """Correct UNSAT paper-guided prefixes using UNSAT-core feedback"""

    print(f"\n{'='*80}")
    print(f"Correcting Paper-Guided UNSAT Prefixes for N={r}")
    print(f"{'='*80}\n")

    if len(unsat_list) == 0:
        print(f"No UNSAT prefixes found for N={r}")
        return []

    print(f"Found {len(unsat_list)} UNSAT prefixes to correct\n")

    # Generate CNF encoding
    print("Generating CNF encoding...")
    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    print(f"CNF has {len(cnf.clauses)} clauses\n")

    correction_results = []

    print("="*80)
    print("Attempting Corrections")
    print("="*80 + "\n")

    for candidate in unsat_list:
        cid = candidate['id']
        prefix = candidate['prefix']
        unsat_core_assignments = candidate['unsat_core_assignments']

        print(f"Prefix {cid}:")
        print(f"  Original: {format_prefix_compact(prefix)}")
        print(f"  UNSAT Core Size: {len(unsat_core_assignments)} assignments")

        # Attempt correction
        corrected_prefix, num_changes, result = smart_correct_prefix(
            prefix, unsat_core_assignments, r, cnf, varmap
        )

        print(f"  Corrected: {format_prefix_compact(corrected_prefix)}")
        print(f"  Changes: {num_changes}")
        print(f"  Result: {'SAT' if result['sat'] else 'UNSAT'}")
        print()

        correction_results.append({
            'original_id': cid,
            'original_prefix': prefix,
            'original_prefix_str': format_prefix_compact(prefix),
            'unsat_core_size': len(unsat_core_assignments),
            'unsat_core': unsat_core_assignments,
            'corrected_prefix': corrected_prefix,
            'corrected_prefix_str': format_prefix_compact(corrected_prefix),
            'num_changes': num_changes,
            'result_sat': result['sat'],
            'result_core_size': len(result.get('unsat_core', [])) if result.get('unsat_core') else 0
        })

    # Write results
    print("="*80)
    print("Writing results...")
    print("="*80 + "\n")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Paper-Guided Prefix Correction Results (r={r})\n")
        f.write("="*80 + "\n\n")

        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Orig ID':<8} {'Changes':<10} {'Result':<10} {'Original Prefix'}\n")
        f.write(f"{'':8} {'':10} {'':10} {'Corrected Prefix'}\n")
        f.write("-"*80 + "\n")

        for res in correction_results:
            f.write(f"{res['original_id']:<8} {res['num_changes']:<10} "
                   f"{'SAT' if res['result_sat'] else 'UNSAT':<10} "
                   f"{res['original_prefix_str']}\n")
            f.write(f"{'':8} {'':10} {'':10} {res['corrected_prefix_str']}\n\n")

        # Detailed results
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")

        for res in correction_results:
            f.write(f"Original Prefix {res['original_id']}\n")
            f.write(f"  Original:  {res['original_prefix_str']}\n")
            f.write(f"  UNSAT Core Size:  {res['unsat_core_size']}\n")
            f.write(f"  UNSAT Core (sample): {res['unsat_core'][:5]}\n")
            if len(res['unsat_core']) > 5:
                f.write(f"    ... and {len(res['unsat_core']) - 5} more\n")
            f.write(f"\n")
            f.write(f"  Corrected: {res['corrected_prefix_str']}\n")
            f.write(f"  Number of Changes: {res['num_changes']}\n")
            f.write(f"  Result: {'SAT' if res['result_sat'] else 'UNSAT'}\n")
            if not res['result_sat']:
                f.write(f"  New UNSAT Core Size: {res['result_core_size']}\n")
            f.write("\n" + "-"*80 + "\n\n")

        # Statistics
        success_count = sum(1 for r in correction_results if r['result_sat'])
        f.write("="*80 + "\n")
        f.write("STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Total UNSAT Prefixes: {len(correction_results)}\n")
        f.write(f"Successfully Corrected: {success_count} ({success_count/len(correction_results)*100:.1f}%)\n")
        f.write(f"Still UNSAT:            {len(correction_results) - success_count}\n")

        if success_count > 0:
            avg_changes = sum(r['num_changes'] for r in correction_results if r['result_sat']) / success_count
            f.write(f"Avg Changes (successful): {avg_changes:.1f}\n")

    print(f"Results written to {output_file}\n")

    # Save corrected prefixes in same format
    corrected_only = [res['corrected_prefix'] for res in correction_results if res['result_sat']]
    if corrected_only:
        corrected_file = output_file.replace('.txt', '_SAT_only.txt')
        write_seq(corrected_only, corrected_file)
        print(f"Successfully corrected prefixes written to {corrected_file}")

    # Print summary
    success_count = sum(1 for r in correction_results if r['result_sat'])
    print(f"\nSUCCESS RATE: {success_count}/{len(correction_results)} corrected to SAT")

    return correction_results

def main():
    print("="*80)
    print("UNSAT-CORE REPAIR OF THEORY-GUIDED PREFIXES")
    print("="*80)

    # Load UNSAT prefixes
    print("\nLoading UNSAT prefixes from paper_prefix_eval_results.json...")
    n5_unsat, n7_unsat = load_unsat_prefixes_from_json("paper_prefix_eval_results.json")

    print(f"  N=5: {len(n5_unsat)} UNSAT prefixes")
    print(f"  N=7: {len(n7_unsat)} UNSAT prefixes")
    print(f"  Total: {len(n5_unsat) + len(n7_unsat)} UNSAT prefixes\n")

    # Correct N=5 prefixes
    results_n5 = correct_paper_guided_prefixes(
        r=5,
        unsat_list=n5_unsat,
        output_file="paper_corrected_prefixes_5.txt"
    )

    # Correct N=7 prefixes
    results_n7 = correct_paper_guided_prefixes(
        r=7,
        unsat_list=n7_unsat,
        output_file="paper_corrected_prefixes_7.txt"
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
        json_res = {
            "original_id": res['original_id'],
            "original": [list(seq) for seq in res['original_prefix']],
            "original_str": res['original_prefix_str'],
            "core": res['unsat_core'],
            "corrected": [list(seq) for seq in res['corrected_prefix']],
            "corrected_str": res['corrected_prefix_str'],
            "changes": res['num_changes'],
            "status": "SAT" if res['result_sat'] else "UNSAT"
        }
        combined_results["N5"].append(json_res)

    # Convert N=7 results
    for res in results_n7:
        json_res = {
            "original_id": res['original_id'],
            "original": [list(seq) for seq in res['original_prefix']],
            "original_str": res['original_prefix_str'],
            "core": res['unsat_core'],
            "corrected": [list(seq) for seq in res['corrected_prefix']],
            "corrected_str": res['corrected_prefix_str'],
            "changes": res['num_changes'],
            "status": "SAT" if res['result_sat'] else "UNSAT"
        }
        combined_results["N7"].append(json_res)

    with open("paper_corrected_prefixes.json", 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("JSON results written to paper_corrected_prefixes.json\n")

    # Generate summary report
    print("="*80)
    print("Writing summary report...")
    print("="*80 + "\n")

    with open("paper_corrected_prefixes_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("PAPER-GUIDED PREFIX UNSAT-CORE REPAIR SUMMARY\n")
        f.write("="*80 + "\n\n")

        # N=5 Summary
        f.write("N=5 CORRECTIONS\n")
        f.write("-"*80 + "\n\n")

        if len(results_n5) > 0:
            for res in results_n5:
                f.write(f"Prefix {res['original_id']}:\n")
                f.write(f"  Original:      {res['original_prefix_str']}\n")
                f.write(f"  UNSAT Core:    {len(res['unsat_core'])} conflicting assignments\n")
                f.write(f"  Corrected:     {res['corrected_prefix_str']}\n")
                f.write(f"  Flips:         {res['num_changes']} positions changed\n")
                f.write(f"  Result:        {'SAT' if res['result_sat'] else 'UNSAT'}\n")
                f.write(f"\n")

            sat_count = sum(1 for r in results_n5 if r['result_sat'])
            f.write(f"N=5 Success Rate: {sat_count}/{len(results_n5)} "
                   f"({sat_count/len(results_n5)*100:.1f}%)\n")
            if sat_count > 0:
                avg_changes = sum(r['num_changes'] for r in results_n5 if r['result_sat']) / sat_count
                f.write(f"Average Changes (successful): {avg_changes:.1f}\n")
        else:
            f.write("No UNSAT prefixes to correct for N=5\n")

        f.write("\n\n")

        # N=7 Summary
        f.write("="*80 + "\n")
        f.write("N=7 CORRECTIONS\n")
        f.write("-"*80 + "\n\n")

        if len(results_n7) > 0:
            for res in results_n7:
                f.write(f"Prefix {res['original_id']}:\n")
                f.write(f"  Original:      {res['original_prefix_str']}\n")
                f.write(f"  UNSAT Core:    {len(res['unsat_core'])} conflicting assignments\n")
                f.write(f"  Corrected:     {res['corrected_prefix_str']}\n")
                f.write(f"  Flips:         {res['num_changes']} positions changed\n")
                f.write(f"  Result:        {'SAT' if res['result_sat'] else 'UNSAT'}\n")
                f.write(f"\n")

            sat_count = sum(1 for r in results_n7 if r['result_sat'])
            f.write(f"N=7 Success Rate: {sat_count}/{len(results_n7)} "
                   f"({sat_count/len(results_n7)*100:.1f}%)\n")
            if sat_count > 0:
                avg_changes = sum(r['num_changes'] for r in results_n7 if r['result_sat']) / sat_count
                f.write(f"Average Changes (successful): {avg_changes:.1f}\n")
        else:
            f.write("No UNSAT prefixes to correct for N=7\n")

        f.write("\n\n")

        # Overall statistics
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")

        total_unsat = len(results_n5) + len(results_n7)
        total_sat = sum(1 for r in results_n5 if r['result_sat']) + sum(1 for r in results_n7 if r['result_sat'])

        f.write(f"Total UNSAT Prefixes:      {total_unsat}\n")
        f.write(f"Successfully Corrected:    {total_sat} ({total_sat/total_unsat*100:.1f}%)\n")
        f.write(f"Still UNSAT:               {total_unsat - total_sat}\n")

        if total_sat > 0:
            all_results = results_n5 + results_n7
            avg_changes = sum(r['num_changes'] for r in all_results if r['result_sat']) / total_sat
            f.write(f"Average Changes:           {avg_changes:.2f} bit flips\n")

    print("Summary report written to paper_corrected_prefixes_summary.txt\n")

    # Overall statistics
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    total_n5_sat = sum(1 for r in results_n5 if r['result_sat'])
    total_n7_sat = sum(1 for r in results_n7 if r['result_sat'])
    total_sat = total_n5_sat + total_n7_sat
    total_unsat = len(results_n5) + len(results_n7)

    print(f"N=5: {total_n5_sat}/{len(results_n5)} corrected" if len(results_n5) > 0 else "N=5: No UNSAT prefixes")
    print(f"N=7: {total_n7_sat}/{len(results_n7)} corrected" if len(results_n7) > 0 else "N=7: No UNSAT prefixes")
    print(f"Overall: {total_sat}/{total_unsat} corrected ({total_sat/total_unsat*100:.1f}%)")

    print("\n" + "="*80)
    print("CORRECTION COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  - paper_corrected_prefixes_5.txt")
    print("  - paper_corrected_prefixes_7.txt")
    print("  - paper_corrected_prefixes.json")
    print("  - paper_corrected_prefixes_summary.txt")
    print("\nExperiment complete. Ready for next steps.")

if __name__ == "__main__":
    main()
