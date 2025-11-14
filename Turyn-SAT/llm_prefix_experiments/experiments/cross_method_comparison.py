"""
Cross-Method Comparison Experiment
Compares LLM-generated, Paper-guided, and Baseline valid prefixes
"""

import json
import numpy as np
from collections import defaultdict
from sequence_generator import generate_dataset_only, read_seq

def hamming_distance(seq1, seq2):
    """Compute Hamming distance between two sequences"""
    max_len = max(len(seq1), len(seq2))
    # Pad shorter sequence with zeros (treated as different)
    s1 = list(seq1) + [0] * (max_len - len(seq1))
    s2 = list(seq2) + [0] * (max_len - len(seq2))
    return sum(1 for a, b in zip(s1, s2) if a != b)

def prefix_hamming_distance(prefix1, prefix2):
    """Compute total Hamming distance between two prefixes (4 sequences)"""
    return sum(hamming_distance(prefix1[i], prefix2[i]) for i in range(4))

def edit_distance_to_nearest(prefix, valid_prefixes):
    """Find minimum edit distance to any valid prefix"""
    if len(valid_prefixes) == 0:
        return float('inf')
    distances = [prefix_hamming_distance(prefix, valid) for valid in valid_prefixes]
    return min(distances)

def count_transitions(sequence):
    """Count +1->-1 and -1->+1 transitions"""
    if len(sequence) <= 1:
        return 0
    transitions = sum(1 for i in range(len(sequence)-1) if sequence[i] != sequence[i+1])
    return transitions

def average_run_length(sequence):
    """Compute average run length"""
    if len(sequence) == 0:
        return 0
    runs = []
    current_run = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    return np.mean(runs) if runs else 0

def is_alternating(sequence):
    """Check if sequence is alternating (perfect or near-perfect)"""
    if len(sequence) <= 1:
        return False
    transitions = count_transitions(sequence)
    # Near-alternating: at least 80% of positions have transitions
    return transitions >= 0.8 * (len(sequence) - 1)

def check_canonical_violations(prefix):
    """Check if prefix violates canonical form (first element must be 1)"""
    violations = sum(1 for seq in prefix if len(seq) > 0 and seq[0] != 1)
    return violations

def check_forbidden_patterns(prefix):
    """Check for forbidden patterns based on theory rules"""
    violations = 0

    # Rule: Avoid long runs (>3) at early positions
    for seq in prefix:
        if len(seq) >= 4:
            for i in range(len(seq) - 3):
                if all(seq[i] == seq[i+j] for j in range(4)):
                    violations += 1

    # Rule: Avoid all sequences having +1 at same position (except position 0)
    max_len = max(len(seq) for seq in prefix)
    for pos in range(1, max_len):
        if all(len(seq) > pos and seq[pos] == 1 for seq in prefix if len(seq) > pos):
            if all(len(seq) > pos for seq in prefix):  # All sequences defined at this position
                violations += 1

    return violations

def analyze_structural_patterns(prefixes, label):
    """Analyze structural patterns in a set of prefixes"""
    print(f"\nAnalyzing structural patterns for {label}...")

    stats = {
        'total_prefixes': len(prefixes),
        'transitions_per_seq': defaultdict(list),
        'run_lengths': defaultdict(list),
        'alternating_count': defaultdict(int),
        'canonical_violations': 0,
        'forbidden_patterns': 0
    }

    for prefix in prefixes:
        # Canonical violations
        stats['canonical_violations'] += check_canonical_violations(prefix)

        # Forbidden patterns
        stats['forbidden_patterns'] += check_forbidden_patterns(prefix)

        # Per-sequence analysis
        for i, seq in enumerate(prefix):
            seq_name = ['A', 'B', 'C', 'D'][i]
            if len(seq) > 0:
                stats['transitions_per_seq'][seq_name].append(count_transitions(seq))
                stats['run_lengths'][seq_name].append(average_run_length(seq))
                if is_alternating(seq):
                    stats['alternating_count'][seq_name] += 1

    # Compute averages
    summary = {
        'total_prefixes': stats['total_prefixes'],
        'canonical_violations': stats['canonical_violations'],
        'forbidden_patterns': stats['forbidden_patterns'],
        'avg_transitions': {},
        'avg_run_length': {},
        'alternating_freq': {}
    }

    for seq_name in ['A', 'B', 'C', 'D']:
        if stats['transitions_per_seq'][seq_name]:
            summary['avg_transitions'][seq_name] = np.mean(stats['transitions_per_seq'][seq_name])
            summary['avg_run_length'][seq_name] = np.mean(stats['run_lengths'][seq_name])
            summary['alternating_freq'][seq_name] = stats['alternating_count'][seq_name] / len(prefixes) * 100
        else:
            summary['avg_transitions'][seq_name] = 0
            summary['avg_run_length'][seq_name] = 0
            summary['alternating_freq'][seq_name] = 0

    return summary

def analyze_unsat_cores(unsat_prefixes_with_cores):
    """Analyze UNSAT core patterns"""
    print("\nAnalyzing UNSAT cores...")

    if len(unsat_prefixes_with_cores) == 0:
        return {
            'total_unsat': 0,
            'avg_core_size': 0,
            'seq_frequency': {},
            'position_conflicts': {}
        }

    core_sizes = []
    seq_counts = defaultdict(int)
    position_counts = defaultdict(int)

    for prefix_data in unsat_prefixes_with_cores:
        core = prefix_data.get('unsat_core_assignments') or prefix_data.get('core', [])
        if core:
            core_sizes.append(len(core))

            for assignment in core:
                if isinstance(assignment, list) and len(assignment) == 2:
                    var_name = assignment[0]
                    # Parse variable name like "X_2" or "W_3"
                    parts = var_name.split('_')
                    if len(parts) == 2:
                        seq = parts[0]
                        pos = int(parts[1])
                        seq_counts[seq] += 1
                        position_counts[pos] += 1

    return {
        'total_unsat': len(unsat_prefixes_with_cores),
        'avg_core_size': np.mean(core_sizes) if core_sizes else 0,
        'seq_frequency': dict(seq_counts),
        'position_conflicts': dict(sorted(position_counts.items()))
    }

def load_all_data():
    """Load all prefix data from files"""
    print("="*80)
    print("LOADING ALL DATA")
    print("="*80)

    data = {}

    # Baseline valid prefixes
    print("\n1. Loading baseline valid prefixes...")
    data['baseline_prefixes'] = generate_dataset_only(5)
    print(f"   Loaded {len(data['baseline_prefixes'])} baseline prefixes")

    # LLM-generated raw
    print("\n2. Loading LLM-generated raw prefixes...")
    data['llm_raw_prefixes'] = read_seq("llm_generated_prefixes_5.txt")
    print(f"   Loaded {len(data['llm_raw_prefixes'])} LLM raw prefixes")

    # LLM evaluation results
    print("\n3. Loading LLM evaluation results...")
    with open("llm_prefix_eval_results_5.json", 'r') as f:
        data['llm_eval'] = json.load(f)
    print(f"   Loaded evaluation for {len(data['llm_eval'])} LLM prefixes")

    # LLM corrected
    print("\n4. Loading LLM corrected prefixes...")
    with open("llm_corrected_prefixes_5.json", 'r') as f:
        data['llm_corrected'] = json.load(f)
    print(f"   Loaded {len(data['llm_corrected'])} LLM corrections")

    # Paper-guided raw
    print("\n5. Loading paper-guided prefixes...")
    with open("paper_guided_prefixes.json", 'r') as f:
        paper_data = json.load(f)
        data['paper_raw_prefixes'] = [tuple(tuple(seq) for seq in p) for p in paper_data['n5_prefixes']]
    print(f"   Loaded {len(data['paper_raw_prefixes'])} paper-guided prefixes")

    # Paper evaluation results
    print("\n6. Loading paper evaluation results...")
    with open("paper_prefix_eval_results.json", 'r') as f:
        paper_eval = json.load(f)
        data['paper_eval'] = paper_eval['N5']
    print(f"   Loaded evaluation for {len(data['paper_eval'])} paper prefixes")

    # Paper corrected
    print("\n7. Loading paper corrected prefixes...")
    with open("paper_corrected_prefixes.json", 'r') as f:
        paper_corrected = json.load(f)
        data['paper_corrected'] = paper_corrected['N5']
    print(f"   Loaded {len(data['paper_corrected'])} paper corrections")

    return data

def compute_sat_rates(data):
    """Compute SAT rates for all groups"""
    print("\n" + "="*80)
    print("COMPUTING SAT RATES")
    print("="*80)

    results = {}

    # Baseline (all SAT by definition)
    results['baseline'] = {
        'total': len(data['baseline_prefixes']),
        'sat': len(data['baseline_prefixes']),
        'unsat': 0,
        'sat_rate': 100.0
    }

    # LLM raw
    llm_raw_sat = sum(1 for r in data['llm_eval'] if r['sat'])
    results['llm_raw'] = {
        'total': len(data['llm_eval']),
        'sat': llm_raw_sat,
        'unsat': len(data['llm_eval']) - llm_raw_sat,
        'sat_rate': llm_raw_sat / len(data['llm_eval']) * 100
    }

    # LLM corrected
    llm_corrected_sat = sum(1 for r in data['llm_corrected'] if r['result_sat'])
    results['llm_corrected'] = {
        'total': len(data['llm_corrected']),
        'sat': llm_corrected_sat,
        'unsat': len(data['llm_corrected']) - llm_corrected_sat,
        'sat_rate': llm_corrected_sat / len(data['llm_corrected']) * 100 if len(data['llm_corrected']) > 0 else 0
    }

    # Paper raw
    paper_raw_sat = sum(1 for r in data['paper_eval'] if r['sat'])
    results['paper_raw'] = {
        'total': len(data['paper_eval']),
        'sat': paper_raw_sat,
        'unsat': len(data['paper_eval']) - paper_raw_sat,
        'sat_rate': paper_raw_sat / len(data['paper_eval']) * 100
    }

    # Paper corrected
    paper_corrected_sat = sum(1 for r in data['paper_corrected'] if r['status'] == 'SAT')
    results['paper_corrected'] = {
        'total': len(data['paper_corrected']),
        'sat': paper_corrected_sat,
        'unsat': len(data['paper_corrected']) - paper_corrected_sat,
        'sat_rate': paper_corrected_sat / len(data['paper_corrected']) * 100 if len(data['paper_corrected']) > 0 else 0
    }

    return results

def compute_edit_distances(data):
    """Compute edit distances to nearest valid prefix"""
    print("\n" + "="*80)
    print("COMPUTING EDIT DISTANCES")
    print("="*80)

    baseline = data['baseline_prefixes']
    results = {}

    # LLM raw
    print("\nComputing distances for LLM raw...")
    llm_raw_distances = []
    for prefix in data['llm_raw_prefixes']:
        dist = edit_distance_to_nearest(prefix, baseline)
        llm_raw_distances.append(dist)

    results['llm_raw'] = {
        'mean': np.mean(llm_raw_distances),
        'median': np.median(llm_raw_distances),
        'min': np.min(llm_raw_distances),
        'max': np.max(llm_raw_distances)
    }

    # LLM corrected
    print("Computing distances for LLM corrected...")
    llm_corrected_distances = []
    for correction in data['llm_corrected']:
        prefix = tuple(tuple(seq) for seq in correction['corrected_prefix'])
        dist = edit_distance_to_nearest(prefix, baseline)
        llm_corrected_distances.append(dist)

    results['llm_corrected'] = {
        'mean': np.mean(llm_corrected_distances) if llm_corrected_distances else 0,
        'median': np.median(llm_corrected_distances) if llm_corrected_distances else 0,
        'min': np.min(llm_corrected_distances) if llm_corrected_distances else 0,
        'max': np.max(llm_corrected_distances) if llm_corrected_distances else 0
    }

    # Paper raw
    print("Computing distances for Paper raw...")
    paper_raw_distances = []
    for prefix in data['paper_raw_prefixes']:
        dist = edit_distance_to_nearest(prefix, baseline)
        paper_raw_distances.append(dist)

    results['paper_raw'] = {
        'mean': np.mean(paper_raw_distances),
        'median': np.median(paper_raw_distances),
        'min': np.min(paper_raw_distances),
        'max': np.max(paper_raw_distances)
    }

    # Paper corrected
    print("Computing distances for Paper corrected...")
    paper_corrected_distances = []
    for correction in data['paper_corrected']:
        prefix = tuple(tuple(seq) for seq in correction['corrected'])
        dist = edit_distance_to_nearest(prefix, baseline)
        paper_corrected_distances.append(dist)

    results['paper_corrected'] = {
        'mean': np.mean(paper_corrected_distances) if paper_corrected_distances else 0,
        'median': np.median(paper_corrected_distances) if paper_corrected_distances else 0,
        'min': np.min(paper_corrected_distances) if paper_corrected_distances else 0,
        'max': np.max(paper_corrected_distances) if paper_corrected_distances else 0
    }

    return results

def compute_structural_analysis(data):
    """Compute structural pattern analysis"""
    print("\n" + "="*80)
    print("STRUCTURAL PATTERN ANALYSIS")
    print("="*80)

    results = {}

    results['baseline'] = analyze_structural_patterns(data['baseline_prefixes'], "Baseline")
    results['llm_raw'] = analyze_structural_patterns(data['llm_raw_prefixes'], "LLM Raw")

    # LLM corrected
    llm_corrected_prefixes = [tuple(tuple(seq) for seq in r['corrected_prefix'])
                              for r in data['llm_corrected']]
    results['llm_corrected'] = analyze_structural_patterns(llm_corrected_prefixes, "LLM Corrected")

    results['paper_raw'] = analyze_structural_patterns(data['paper_raw_prefixes'], "Paper Raw")

    # Paper corrected
    paper_corrected_prefixes = [tuple(tuple(seq) for seq in r['corrected'])
                                for r in data['paper_corrected']]
    results['paper_corrected'] = analyze_structural_patterns(paper_corrected_prefixes, "Paper Corrected")

    return results

def compute_unsat_core_analysis(data):
    """Analyze UNSAT cores"""
    print("\n" + "="*80)
    print("UNSAT CORE ANALYSIS")
    print("="*80)

    results = {}

    # LLM raw UNSAT
    llm_unsat = [r for r in data['llm_eval'] if not r['sat']]
    results['llm_raw'] = analyze_unsat_cores(llm_unsat)

    # LLM corrected (still UNSAT)
    llm_still_unsat = [r for r in data['llm_corrected'] if not r['result_sat']]
    results['llm_corrected'] = analyze_unsat_cores(llm_still_unsat)

    # Paper raw UNSAT
    paper_unsat = [r for r in data['paper_eval'] if not r['sat']]
    results['paper_raw'] = analyze_unsat_cores(paper_unsat)

    # Paper corrected (still UNSAT)
    paper_still_unsat = [r for r in data['paper_corrected'] if r['status'] != 'SAT']
    results['paper_corrected'] = analyze_unsat_cores(paper_still_unsat)

    return results

def compute_improvement_metrics(data):
    """Compute improvement from raw to corrected"""
    print("\n" + "="*80)
    print("IMPROVEMENT METRICS")
    print("="*80)

    results = {}

    # LLM improvements
    llm_raw_sat_rate = sum(1 for r in data['llm_eval'] if r['sat']) / len(data['llm_eval']) * 100
    llm_corrected_sat_rate = sum(1 for r in data['llm_corrected'] if r['result_sat']) / len(data['llm_corrected']) * 100 if len(data['llm_corrected']) > 0 else 0

    llm_avg_flips = np.mean([r['num_changes'] for r in data['llm_corrected'] if r['result_sat']]) if any(r['result_sat'] for r in data['llm_corrected']) else 0

    results['llm'] = {
        'raw_sat_rate': llm_raw_sat_rate,
        'corrected_sat_rate': llm_corrected_sat_rate,
        'sat_improvement': llm_corrected_sat_rate - llm_raw_sat_rate,
        'avg_flips': llm_avg_flips,
        'correction_success_rate': sum(1 for r in data['llm_corrected'] if r['result_sat']) / len(data['llm_corrected']) * 100 if len(data['llm_corrected']) > 0 else 0
    }

    # Paper improvements
    paper_raw_sat_rate = sum(1 for r in data['paper_eval'] if r['sat']) / len(data['paper_eval']) * 100
    paper_corrected_sat_rate = sum(1 for r in data['paper_corrected'] if r['status'] == 'SAT') / len(data['paper_corrected']) * 100 if len(data['paper_corrected']) > 0 else 0

    paper_avg_flips = np.mean([r['changes'] for r in data['paper_corrected'] if r['status'] == 'SAT']) if any(r['status'] == 'SAT' for r in data['paper_corrected']) else 0

    results['paper'] = {
        'raw_sat_rate': paper_raw_sat_rate,
        'corrected_sat_rate': paper_corrected_sat_rate,
        'sat_improvement': paper_corrected_sat_rate - paper_raw_sat_rate,
        'avg_flips': paper_avg_flips,
        'correction_success_rate': sum(1 for r in data['paper_corrected'] if r['status'] == 'SAT') / len(data['paper_corrected']) * 100 if len(data['paper_corrected']) > 0 else 0
    }

    return results

def generate_ascii_heatmap(position_conflicts, title):
    """Generate ASCII heatmap for position conflicts"""
    if not position_conflicts:
        return "No conflicts"

    max_conflicts = max(position_conflicts.values())
    lines = [title]
    lines.append("-" * 40)

    for pos, count in sorted(position_conflicts.items()):
        bar_length = int(count / max_conflicts * 30)
        bar = "#" * bar_length
        lines.append(f"Pos {pos}: {bar} ({count})")

    return "\n".join(lines)

def write_summary_report(sat_rates, edit_distances, structural, unsat_cores, improvements):
    """Write comprehensive summary report"""
    print("\n" + "="*80)
    print("WRITING SUMMARY REPORT")
    print("="*80)

    with open("comparison_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-METHOD COMPARISON SUMMARY\n")
        f.write("Turyn Sequence Prefix Generation Methods\n")
        f.write("="*80 + "\n\n")

        # SAT Rates
        f.write("1. SAT RATES\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Method':<20} {'Total':<10} {'SAT':<10} {'UNSAT':<10} {'SAT Rate':<10}\n")
        f.write("-"*80 + "\n")
        for method, stats in sat_rates.items():
            f.write(f"{method:<20} {stats['total']:<10} {stats['sat']:<10} "
                   f"{stats['unsat']:<10} {stats['sat_rate']:.1f}%\n")
        f.write("\n\n")

        # Edit Distances
        f.write("2. EDIT DISTANCE TO NEAREST VALID PREFIX\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Method':<20} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}\n")
        f.write("-"*80 + "\n")
        for method, stats in edit_distances.items():
            f.write(f"{method:<20} {stats['mean']:.2f}{'':<6} {stats['median']:.2f}{'':<6} "
                   f"{stats['min']:<10} {stats['max']:<10}\n")
        f.write("\n\n")

        # Structural Patterns
        f.write("3. STRUCTURAL PATTERN ANALYSIS\n")
        f.write("-"*80 + "\n\n")

        for method, stats in structural.items():
            f.write(f"{method.upper()}:\n")
            f.write(f"  Total Prefixes: {stats['total_prefixes']}\n")
            f.write(f"  Canonical Violations: {stats['canonical_violations']}\n")
            f.write(f"  Forbidden Patterns: {stats['forbidden_patterns']}\n")
            f.write(f"  Average Transitions per Sequence:\n")
            for seq in ['A', 'B', 'C', 'D']:
                f.write(f"    {seq}: {stats['avg_transitions'][seq]:.2f}\n")
            f.write(f"  Average Run Length:\n")
            for seq in ['A', 'B', 'C', 'D']:
                f.write(f"    {seq}: {stats['avg_run_length'][seq]:.2f}\n")
            f.write(f"  Alternating Pattern Frequency:\n")
            for seq in ['A', 'B', 'C', 'D']:
                f.write(f"    {seq}: {stats['alternating_freq'][seq]:.1f}%\n")
            f.write("\n")

        # UNSAT Core Analysis
        f.write("\n4. UNSAT CORE SEVERITY\n")
        f.write("-"*80 + "\n\n")

        for method, stats in unsat_cores.items():
            f.write(f"{method.upper()}:\n")
            f.write(f"  Total UNSAT: {stats['total_unsat']}\n")
            f.write(f"  Average Core Size: {stats['avg_core_size']:.1f}\n")
            f.write(f"  Sequence Frequency in Cores:\n")
            for seq, count in stats['seq_frequency'].items():
                f.write(f"    {seq}: {count}\n")
            f.write(f"  Position Conflicts:\n")
            for pos, count in stats['position_conflicts'].items():
                f.write(f"    Position {pos}: {count}\n")
            f.write("\n")

            # ASCII heatmap
            f.write(generate_ascii_heatmap(stats['position_conflicts'],
                                          f"  Position Conflict Heatmap ({method})"))
            f.write("\n\n")

        # Improvements
        f.write("\n5. IMPROVEMENT MEASUREMENTS (Raw -> Corrected)\n")
        f.write("-"*80 + "\n\n")

        for method, stats in improvements.items():
            f.write(f"{method.upper()}:\n")
            f.write(f"  Raw SAT Rate: {stats['raw_sat_rate']:.1f}%\n")
            f.write(f"  Corrected SAT Rate: {stats['corrected_sat_rate']:.1f}%\n")
            f.write(f"  SAT Improvement: {stats['sat_improvement']:+.1f}%\n")
            f.write(f"  Average Flips per Correction: {stats['avg_flips']:.2f}\n")
            f.write(f"  Correction Success Rate: {stats['correction_success_rate']:.1f}%\n")
            f.write("\n")

        # Key Findings
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")

        # Best SAT rate
        best_method = max(sat_rates.items(), key=lambda x: x[1]['sat_rate'])
        f.write(f"1. Best SAT Rate: {best_method[0]} ({best_method[1]['sat_rate']:.1f}%)\n\n")

        # Closest to valid prefixes
        best_distance = min(edit_distances.items(), key=lambda x: x[1]['mean'])
        f.write(f"2. Closest to Valid Prefixes: {best_distance[0]} "
               f"(mean distance: {best_distance[1]['mean']:.2f})\n\n")

        # Best improvement
        best_improvement = max(improvements.items(), key=lambda x: x[1]['sat_improvement'])
        f.write(f"3. Best Improvement: {best_improvement[0]} "
               f"({best_improvement[1]['sat_improvement']:+.1f}% SAT rate increase)\n\n")

        f.write("="*80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*80 + "\n")

    print("Summary report written to comparison_summary.txt")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def write_json_tables(sat_rates, edit_distances, structural, unsat_cores, improvements):
    """Write machine-readable JSON tables"""
    print("Writing JSON tables...")

    tables = {
        'sat_rates': sat_rates,
        'edit_distance': edit_distances,
        'structural': structural,
        'unsat_core': unsat_cores,
        'improvement': improvements
    }

    # Convert numpy types to native Python types
    tables = convert_numpy_types(tables)

    with open("comparison_tables.json", 'w') as f:
        json.dump(tables, f, indent=2)

    print("JSON tables written to comparison_tables.json")

def main():
    print("="*80)
    print("CROSS-METHOD COMPARISON EXPERIMENT")
    print("="*80)

    # Load all data
    data = load_all_data()

    # Compute metrics
    sat_rates = compute_sat_rates(data)
    edit_distances = compute_edit_distances(data)
    structural = compute_structural_analysis(data)
    unsat_cores = compute_unsat_core_analysis(data)
    improvements = compute_improvement_metrics(data)

    # Write outputs
    write_summary_report(sat_rates, edit_distances, structural, unsat_cores, improvements)
    write_json_tables(sat_rates, edit_distances, structural, unsat_cores, improvements)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  - comparison_summary.txt")
    print("  - comparison_tables.json")
    print("\nExperiment complete. Ready for final report.")

if __name__ == "__main__":
    main()
