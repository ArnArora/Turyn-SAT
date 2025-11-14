"""
LLM-Powered Prefix Generation Using Anthropic Claude API

Calls actual LLM to generate novel Turyn sequence prefixes given:
- Examples of valid prefixes
- Examples of invalid prefixes
- Theory constraints

Tests whether real LLMs can understand and generate valid Turyn prefixes.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sequence_generator import generate_dataset_only, generate_partial_values
from anthropic import Anthropic


def format_prefix(prefix):
    """Format prefix as readable string."""
    lines = []
    for i, seq in enumerate(prefix):
        seq_name = ['A', 'B', 'C', 'D'][i]
        formatted = ' '.join(['+1' if x == 1 else '-1' for x in seq])
        lines.append(f"  {seq_name}: [{formatted}]")
    return '\n'.join(lines)


def parse_llm_prefix(response_text):
    """
    Parse LLM response to extract prefix.
    Expected format:
    A: [+1 -1 +1 -1 +1]
    B: [+1 +1 -1 -1 +1]
    C: [+1 -1 -1 +1 +1]
    D: [+1 +1 +1 -1 -1]
    """
    import re

    sequences = {'A': None, 'B': None, 'C': None, 'D': None}

    # Try to find lines with A:, B:, C:, D: followed by values
    for seq_name in ['A', 'B', 'C', 'D']:
        pattern = rf'{seq_name}:\s*\[([^\]]+)\]'
        match = re.search(pattern, response_text, re.IGNORECASE)

        if match:
            values_str = match.group(1)
            # Parse +1/-1 values
            values = []
            for token in values_str.split():
                token = token.strip(',')
                if '+1' in token or '1' == token:
                    values.append(1)
                elif '-1' in token:
                    values.append(-1)

            if values:
                sequences[seq_name] = tuple(values)

    # Check if we got all sequences
    if all(v is not None for v in sequences.values()):
        return [sequences['A'], sequences['B'], sequences['C'], sequences['D']]

    return None


def create_generation_prompt(valid_examples, invalid_examples, num_to_generate=10):
    """Create prompt for LLM to generate new prefixes."""

    # Select subset of examples
    valid_sample = valid_examples[:5]  # First 5 valid
    invalid_sample = invalid_examples[:3]  # First 3 invalid

    prompt = f"""You are helping with mathematical sequence generation. Turyn sequences are special 4-tuples (A, B, C, D) of binary sequences using values +1 and -1.

A valid Turyn sequence prefix must satisfy the autocorrelation property:
TA(s) + TB(s) + 2*TC(s) + 2*TD(s) = 0 for all shifts s != 0

Where TA(s) is the autocorrelation of sequence A at shift s (dot product of sequence with itself shifted by s positions).

Note: Sequences C and D have weight 2 (they appear with coefficient 2 in the formula).

## VALID PREFIX EXAMPLES (these satisfy the Turyn property):

"""

    for i, prefix in enumerate(valid_sample, 1):
        prompt += f"Example {i}:\n{format_prefix(prefix)}\n\n"

    prompt += "\n## INVALID PREFIX EXAMPLES (these violate the Turyn property):\n\n"

    for i, prefix in enumerate(invalid_sample, 1):
        prompt += f"Invalid {i}:\n{format_prefix(prefix)}\n\n"

    prompt += f"""
## YOUR TASK:

Generate {num_to_generate} NEW Turyn sequence prefixes of length N=5 (5 elements per sequence).

CONSTRAINTS:
1. Each sequence must have exactly 5 elements
2. Each element must be +1 or -1
3. First element of each sequence MUST be +1 (canonical form)
4. Try to satisfy the Turyn autocorrelation property
5. Do NOT duplicate the examples shown above

For each prefix, output in this exact format:

Prefix 1:
A: [+1 ... ... ... ...]
B: [+1 ... ... ... ...]
C: [+1 ... ... ... ...]
D: [+1 ... ... ... ...]

Continue for all {num_to_generate} prefixes.
"""

    return prompt


def call_claude_api(prompt, api_key):
    """Call Claude API to generate prefixes."""

    client = Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        temperature=1.0,  # Higher temperature for more creative generation
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


def extract_all_prefixes(llm_response):
    """Extract all prefixes from LLM response."""
    prefixes = []

    # Split by "Prefix N:" markers
    import re
    sections = re.split(r'Prefix\s+\d+:', llm_response)

    for section in sections[1:]:  # Skip first split (before first Prefix)
        prefix = parse_llm_prefix(section)
        if prefix:
            prefixes.append(prefix)

    return prefixes


def main():
    print("="*80)
    print("LLM-POWERED TURYN PREFIX GENERATION")
    print("="*80)

    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    print("\n[1/5] Generating valid prefix examples...")
    r = 5
    valid_prefixes = generate_dataset_only(r)
    print(f"  Generated {len(valid_prefixes)} valid prefixes")

    print("\n[2/5] Generating invalid prefix examples...")
    invalid_prefixes = generate_partial_values(r, valid_prefixes[:10])
    print(f"  Generated {len(invalid_prefixes)} invalid prefixes")

    print("\n[3/5] Creating LLM prompt with examples...")
    num_to_generate = 10
    prompt = create_generation_prompt(valid_prefixes, invalid_prefixes, num_to_generate)
    print(f"  Prompt created ({len(prompt)} characters)")

    print("\n[4/5] Calling Claude API to generate prefixes...")
    print("  (This may take 10-30 seconds...)")

    try:
        llm_response = call_claude_api(prompt, api_key)
        print("  [+] Claude API call successful")

        print("\n[5/5] Parsing LLM response...")
        prefixes = extract_all_prefixes(llm_response)
        print(f"  Successfully parsed {len(prefixes)} prefixes")

        # Save results
        output_dir = "../intermediate_results"
        os.makedirs(output_dir, exist_ok=True)

        # Save raw response
        with open(f"{output_dir}/llm_api_response_raw.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("RAW CLAUDE API RESPONSE\n")
            f.write("="*80 + "\n\n")
            f.write(llm_response)

        print(f"\n  Saved raw response to: {output_dir}/llm_api_response_raw.txt")

        # Save parsed prefixes
        with open(f"{output_dir}/llm_api_generated_prefixes.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"LLM-GENERATED TURYN PREFIXES (N={r})\n")
            f.write(f"Generated by: Claude 3.5 Sonnet\n")
            f.write(f"Total prefixes: {len(prefixes)}\n")
            f.write("="*80 + "\n\n")

            for i, prefix in enumerate(prefixes, 1):
                f.write(f"Prefix {i}:\n")
                f.write(format_prefix(prefix) + "\n\n")

        print(f"  Saved prefixes to: {output_dir}/llm_api_generated_prefixes.txt")

        # Save as JSON for evaluation
        prefixes_serializable = [
            [[int(x) for x in seq] for seq in prefix]
            for prefix in prefixes
        ]

        with open(f"{output_dir}/llm_api_generated_prefixes.json", 'w') as f:
            json.dump({
                'r': r,
                'num_prefixes': len(prefixes),
                'prefixes': prefixes_serializable,
                'generation_method': 'claude_api',
                'model': 'claude-3-5-sonnet-20241022'
            }, f, indent=2)

        print(f"  Saved JSON to: {output_dir}/llm_api_generated_prefixes.json")

        print("\n" + "="*80)
        print("LLM GENERATION COMPLETE")
        print("="*80)
        print(f"\nSuccessfully generated {len(prefixes)} prefixes using Claude API")
        print("\nNext step: Run evaluate_llm_api_prefixes.py to test with SAT solver")

    except Exception as e:
        print(f"\n  [!] ERROR calling Claude API: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
