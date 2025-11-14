import json
from pysat.solvers import Minisat22
from sequence_generator import write_seq
from cnf_automator import generate_encoding, make_assumptions
from evaluate_llm_candidates import evaluate_prefix_with_core, format_prefix_compact

def parse_core_variable(var_str):
    """Parse 'X_2' into ('X', 2)"""
    parts = var_str.split('_')
    return parts[0], int(parts[1])

def correct_prefix_using_core(prefix, unsat_core_assignments, r):
    """
    Attempt to correct a prefix by modifying values indicated by UNSAT core.

    Strategy:
    1. Identify which positions in the prefix are in the UNSAT core
    2. Try flipping individual values to resolve conflicts
    3. Try small combinations of flips
    4. Maintain canonical form (first element must be 1)
    """

    # Parse UNSAT core to understand which positions conflict
    seq_map = {'X': 0, 'Y': 1, 'Z': 2, 'W': 3}
    core_positions = []

    for var_str, value in unsat_core_assignments:
        seq_name, pos = parse_core_variable(var_str)
        seq_idx = seq_map[seq_name]

        # Only consider positions that are actually in the prefix
        if pos < len(prefix[seq_idx]):
            core_positions.append((seq_idx, pos, value))

    # Create a mutable copy of the prefix
    corrected = [list(seq) for seq in prefix]

    # Strategy 1: Try flipping each core position one at a time
    # Start with positions that have the most impact (later positions, or non-first positions)
    # Sort by position to avoid flipping position 0 first (canonical form requirement)

    flip_candidates = []
    for seq_idx, pos, expected_val in core_positions:
        if pos > 0:  # Don't flip position 0 (must be 1 for canonical form)
            current_val = corrected[seq_idx][pos]
            # If the current value conflicts with what's in core, consider flipping
            if current_val == expected_val:
                # This assignment is in the core, so it's part of the conflict
                flip_candidates.append((seq_idx, pos))

    # Strategy: Try flipping positions in the UNSAT core
    # We'll try multiple combinations and return the first that works

    corrections_to_try = []

    # 1. Try single flips
    for seq_idx, pos in flip_candidates:
        corrections_to_try.append([(seq_idx, pos)])

    # 2. Try pairs of flips
    for i in range(len(flip_candidates)):
        for j in range(i + 1, min(i + 4, len(flip_candidates))):  # Limit combinations
            corrections_to_try.append([flip_candidates[i], flip_candidates[j]])

    # 3. Try removing elements from overspecified sequences
    # If a sequence is fully specified (length r) and in UNSAT core, try truncating
    for seq_idx in range(4):
        if len(prefix[seq_idx]) == r and any(s == seq_idx for s, _, _ in core_positions):
            # Try truncating this sequence
            corrections_to_try.append([('truncate', seq_idx)])

    return corrected, flip_candidates, corrections_to_try

def apply_correction(prefix, correction):
    """Apply a correction (list of (seq_idx, pos) tuples to flip) to a prefix"""
    corrected = [list(seq) for seq in prefix]

    for item in correction:
        if isinstance(item, tuple) and len(item) == 2:
            if item[0] == 'truncate':
                seq_idx = item[1]
                # Truncate to length r-1
                if len(corrected[seq_idx]) > 0:
                    corrected[seq_idx] = corrected[seq_idx][:-1]
            else:
                seq_idx, pos = item
                # Flip the value at this position
                if pos < len(corrected[seq_idx]):
                    corrected[seq_idx][pos] *= -1

    return [tuple(seq) for seq in corrected]

def smart_correct_prefix(prefix, unsat_core_assignments, r, cnf, varmap, max_attempts=50):
    """
    Intelligently correct a prefix using UNSAT core information.
    Returns (corrected_prefix, num_changes, sat_result)
    """

    corrected, flip_candidates, corrections_to_try = correct_prefix_using_core(
        prefix, unsat_core_assignments, r
    )

    print(f"  Identified {len(flip_candidates)} positions in UNSAT core that can be flipped")
    print(f"  Will try {min(len(corrections_to_try), max_attempts)} correction strategies")

    best_correction = None
    best_num_changes = float('inf')

    for idx, correction in enumerate(corrections_to_try[:max_attempts]):
        # Apply this correction
        candidate = apply_correction(prefix, correction)

        # Verify it still maintains canonical form (first element = 1)
        valid_canonical = True
        for seq in candidate:
            if len(seq) > 0 and seq[0] != 1:
                valid_canonical = False
                break

        if not valid_canonical:
            continue

        # Test with SAT solver
        result = evaluate_prefix_with_core(candidate, r, cnf, varmap)

        num_changes = len(correction)

        if result['sat']:
            if num_changes < best_num_changes:
                best_correction = (candidate, num_changes, result)
                best_num_changes = num_changes
                print(f"  [+] Found SAT correction with {num_changes} changes: {format_prefix_compact(candidate)}")
                # Return immediately if we found a solution with minimal changes
                if num_changes <= 2:
                    return best_correction

    if best_correction:
        return best_correction

    # If no correction worked, return the original with modifications and mark as still UNSAT
    # Try a simple heuristic: flip the most conflicting position
    if flip_candidates:
        fallback = apply_correction(prefix, [flip_candidates[0]])
        result = evaluate_prefix_with_core(fallback, r, cnf, varmap)
        print(f"  [-] No SAT correction found, returning best attempt")
        return (fallback, 1, result)

    # Last resort: return original unchanged
    result = evaluate_prefix_with_core(prefix, r, cnf, varmap)
    print(f"  [-] Could not generate any valid corrections")
    return (prefix, 0, result)

def correct_all_unsat_candidates(json_file, r, output_file):
    """Load UNSAT candidates and attempt corrections"""

    print("="*80)
    print("Loading UNSAT candidates from JSON...")
    print("="*80 + "\n")

    with open(json_file, 'r') as f:
        results = json.load(f)

    # Filter for UNSAT candidates
    unsat_candidates = [r for r in results if not r['sat']]
    print(f"Found {len(unsat_candidates)} UNSAT candidates\n")

    # Generate CNF encoding
    print("Generating CNF encoding...")
    cnf, varmap, pmap_lags, pool = generate_encoding(r)
    print(f"CNF has {len(cnf.clauses)} clauses\n")

    correction_results = []

    print("="*80)
    print("Attempting Corrections")
    print("="*80 + "\n")

    for candidate in unsat_candidates:
        cid = candidate['id']
        prefix = [tuple(seq) for seq in candidate['prefix']]
        unsat_core_assignments = candidate['unsat_core_assignments']

        print(f"Candidate {cid}:")
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
            'result_core_size': len(result['unsat_core']) if result['unsat_core'] else 0
        })

    # Write results
    print("="*80)
    print("Writing results...")
    print("="*80 + "\n")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"UNSAT Prefix Correction Results (r={r})\n")
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
            f.write(f"Original Candidate {res['original_id']}\n")
            f.write(f"  Original Prefix:  {res['original_prefix_str']}\n")
            f.write(f"  UNSAT Core Size:  {res['unsat_core_size']}\n")
            f.write(f"  UNSAT Core (sample): {res['unsat_core'][:5]}\n")
            if len(res['unsat_core']) > 5:
                f.write(f"    ... and {len(res['unsat_core']) - 5} more\n")
            f.write(f"\n")
            f.write(f"  Corrected Prefix: {res['corrected_prefix_str']}\n")
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
        f.write(f"Total UNSAT Candidates: {len(correction_results)}\n")
        f.write(f"Successfully Corrected: {success_count} ({success_count/len(correction_results)*100:.1f}%)\n")
        f.write(f"Still UNSAT:            {len(correction_results) - success_count}\n")

        if success_count > 0:
            avg_changes = sum(r['num_changes'] for r in correction_results if r['result_sat']) / success_count
            f.write(f"Avg Changes (successful): {avg_changes:.1f}\n")

    print(f"Results written to {output_file}")

    # Save corrected prefixes in same format as original generator
    corrected_only = [res['corrected_prefix'] for res in correction_results if res['result_sat']]
    if corrected_only:
        corrected_file = output_file.replace('.txt', '_SAT_only.txt')
        write_seq(corrected_only, corrected_file)
        print(f"Successfully corrected prefixes written to {corrected_file}")

    # Save JSON for further processing
    json_output = output_file.replace('.txt', '.json')
    with open(json_output, 'w') as f:
        json.dump(correction_results, f, indent=2)
    print(f"JSON results written to {json_output}")

    print(f"\nSUCCESS RATE: {success_count}/{len(correction_results)} corrected to SAT")

    return correction_results

if __name__ == "__main__":
    r = 5
    json_file = f"llm_prefix_eval_results_{r}.json"
    output_file = f"llm_corrected_prefixes_{r}.txt"

    results = correct_all_unsat_candidates(json_file, r, output_file)
