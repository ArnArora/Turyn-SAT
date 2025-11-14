# Experimental Scripts

This folder contains all experimental Python scripts used in the study.

## Scripts

### Generation
- `generate_llm_candidates.py` - Generate LLM prefixes without theory
- `generate_theory_guided_prefixes.py` - Generate theory-guided prefixes

### Evaluation
- `evaluate_llm_candidates.py` - Evaluate LLM-generated prefixes with SAT
- `evaluate_paper_guided_prefixes.py` - Evaluate theory-guided prefixes

### Repair
- `correct_unsat_prefixes.py` - Repair LLM prefixes using UNSAT cores
- `correct_paper_guided_prefixes.py` - Repair theory-guided prefixes

### Analysis
- `cross_method_comparison.py` - Compare all methods
- `consolidate_final_deliverables.py` - Generate final deliverables

### Utility
- `cleanup_repository.py` - Repository organization script

## Usage

All scripts are self-contained and can be run with:
```
python script_name.py
```

Results are saved to ../intermediate_results/
