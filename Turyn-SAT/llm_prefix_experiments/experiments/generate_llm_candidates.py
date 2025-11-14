import random
from sequence_generator import generate_dataset_only, create_partial_str, write_seq

def generate_llm_candidates(r, num_candidates=10):
    """Generate candidate prefixes that don't exist in the valid dataset"""

    # Get all valid prefixes
    valid_prefixes = generate_dataset_only(r)
    valid_set = set()
    for prefix in valid_prefixes:
        prefix_str = create_partial_str(prefix)
        valid_set.add(prefix_str)

    print(f"Total valid prefixes for r={r}: {len(valid_prefixes)}")

    # Generate candidate prefixes
    candidates = []
    attempts = 0
    max_attempts = 10000

    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1

        # Generate a random partial sequence
        # Randomly choose how many elements to fill (from 1 to 4*r)
        total_positions = 4 * r
        num_positions = random.randint(1, total_positions)

        # Create partial sequence structure
        partial = [[], [], [], []]

        # Fill positions row by row
        for seq_idx in range(4):
            if num_positions > 0:
                # Determine how many elements for this sequence
                if seq_idx < 3:
                    # For first 3 sequences, can fill up to r elements
                    max_for_this_seq = min(r, num_positions)
                    num_elements = random.randint(0, max_for_this_seq)
                else:
                    # For last sequence, fill remaining
                    num_elements = min(r, num_positions)

                # Generate random +1/-1 values
                # But respect canonical form: first element should be 1
                for i in range(num_elements):
                    if i == 0:
                        partial[seq_idx].append(1)
                    else:
                        partial[seq_idx].append(random.choice([1, -1]))

                num_positions -= num_elements

        # Convert to tuples
        partial = [tuple(seq) for seq in partial]

        # Check if this candidate is not in the valid set
        candidate_str = create_partial_str(partial)

        # Skip empty sequences
        if all(len(seq) == 0 for seq in partial):
            continue

        if candidate_str not in valid_set:
            # Check if we haven't already added this candidate
            already_added = False
            for existing in candidates:
                if create_partial_str(existing) == candidate_str:
                    already_added = True
                    break

            if not already_added:
                candidates.append(partial)
                print(f"Generated candidate {len(candidates)}: {partial}")

    print(f"\nGenerated {len(candidates)} unique candidates in {attempts} attempts")
    return candidates

if __name__ == "__main__":
    r = 5
    candidates = generate_llm_candidates(r, num_candidates=10)

    # Write to file
    output_file = f"llm_generated_prefixes_{r}.txt"
    write_seq(candidates, output_file)
    print(f"\nWrote {len(candidates)} candidates to {output_file}")
