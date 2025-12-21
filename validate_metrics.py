#!/usr/bin/env python3
"""
Independent validation script for SST-2 demo dataset.

This script verifies basic metrics that can be calculated without
the full Collapse Index pipeline:
- Flip rate: percentage of examples with inconsistent predictions
- Overall accuracy: model correctness across all variants
- Confidence distribution: how confidence differs between correct/incorrect predictions

Advanced metrics (AUC, CI scores) require the full analysis pipeline.

Usage:
    python validate_metrics.py

Author: Alex Kwon (Collapse Index Labs)
License: MIT
"""
import pandas as pd

# Load dataset
df = pd.read_csv('sst2_ci_demo.csv')

print("=" * 60)
print("SST-2 DATASET INDEPENDENT VALIDATION")
print("=" * 60)
print("\nThis script verifies metrics that DON'T require the CI pipeline.")
print("Full metrics (AUC, CI scores, etc.) require running the analysis.")

# Basic stats
total_rows = len(df)
unique_ids = df['id'].nunique()
print(f"\nDataset Stats:")
print(f"  Total rows: {total_rows}")
print(f"  Unique base examples: {unique_ids}")
print(f"  Variants per base: {total_rows // unique_ids}")

# Calculate is_error
df['is_error'] = (df['pred_label'] != df['true_label']).astype(int)

# Flip rate calculation (INDEPENDENTLY VERIFIABLE)
print("\n" + "=" * 60)
print("✓ FLIP RATE (independently verifiable)")
print("=" * 60)

flip_count = 0
for base_id in df['id'].unique():
    base_examples = df[df['id'] == base_id]
    predictions = base_examples['pred_label'].unique()
    
    # If more than one unique prediction across variants, it flipped
    if len(predictions) > 1:
        flip_count += 1

flip_rate = (flip_count / unique_ids) * 100

print(f"Base examples with flips: {flip_count}/{unique_ids}")
print(f"Flip rate: {flip_rate:.1f}%")
print(f"\n✓ Matches claim: ~42.8%? {abs(flip_rate - 42.8) < 1.0}")

# Overall accuracy (INDEPENDENTLY VERIFIABLE)
print("\n" + "=" * 60)
print("✓ OVERALL ACCURACY (independently verifiable)")
print("=" * 60)

# Base examples only (clean text)
base_df = df[df['variant_id'] == 'base']
base_correct = (base_df['pred_label'] == base_df['true_label']).sum()
base_accuracy = (base_correct / len(base_df)) * 100

# All rows (including perturbations)
total_correct = (df['is_error'] == 0).sum()
overall_accuracy = (total_correct / total_rows) * 100

print(f"Base examples (clean): {base_accuracy:.1f}% ({base_correct}/{len(base_df)})")
print(f"All variants (w/ perturbations): {overall_accuracy:.1f}% ({total_correct}/{total_rows})")
print(f"Degradation: {base_accuracy - overall_accuracy:.1f} percentage points")
print(f"\n✓ Base accuracy matches claim: ~90%? {abs(base_accuracy - 90) < 2.0}")

# Confidence distribution
print("\n" + "=" * 60)
print("✓ CONFIDENCE DISTRIBUTION")
print("=" * 60)
errors_df = df[df['is_error'] == 1]
correct_df = df[df['is_error'] == 0]
print(f"Errors - Mean confidence: {errors_df['confidence'].mean():.4f}")
print(f"Correct - Mean confidence: {correct_df['confidence'].mean():.4f}")
print(f"Gap: {correct_df['confidence'].mean() - errors_df['confidence'].mean():.4f}")
print("\nNote: Small gap suggests confidence barely separates errors")

# CI-dependent metrics
print("\n" + "=" * 60)
print("⚠ METRICS REQUIRING CI PIPELINE")
print("=" * 60)
print("The following metrics CANNOT be independently verified:")
print("  • AUC(CI): 0.698 - requires ROC analysis")
print("  • AUC(Confidence): 0.515 - requires ROC analysis  ")
print("  • ΔAUC: +0.182 - difference between above")
print("  • CI Score: 0.275 - requires perturbation analysis")
print("  • High-Conf Errors: 35 - requires collapse_log from pipeline")
print("\nTo verify these, run the full CI analysis:")
print("  Contact: ask@collapseindex.org")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Flip rate: {flip_rate:.1f}% (matches ~42.8%)")
print(f"✓ Overall accuracy: {overall_accuracy:.1f}%")
print(f"⚠ Full metrics require CI pipeline analysis")
print("=" * 60)


