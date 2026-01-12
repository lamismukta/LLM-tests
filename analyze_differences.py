#!/usr/bin/env python3
"""Analyze how each CV is treated differently across models and pipelines."""
import json
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List
import argparse


def load_mapping() -> Dict[str, Dict]:
    """Load CV ID mapping."""
    mapping_path = Path(__file__).parent / "data" / "cv_id_mapping.json"
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_original_info(sanitized_id: str, mapping: Dict) -> Dict:
    """Get original ID and name from sanitized ID."""
    if sanitized_id in mapping:
        return mapping[sanitized_id]
    return {"original_id": sanitized_id, "original_name": "Unknown"}


def analyze_experiment(experiment_dir: Path, output_file: str = None):
    """Analyze differences in how CVs are treated."""
    mapping = load_mapping()
    
    # Load comparison CSV
    comparison_path = experiment_dir / "comparison.csv"
    if not comparison_path.exists():
        print(f"Error: comparison.csv not found in {experiment_dir}")
        return
    
    df = pd.read_csv(comparison_path)
    
    # Add original IDs and names
    df['original_id'] = df['cv_id'].apply(
        lambda x: get_original_info(x, mapping)['original_id']
    )
    df['original_name'] = df['cv_id'].apply(
        lambda x: get_original_info(x, mapping)['original_name']
    )
    
    # Create analysis output
    analysis = []
    
    # Group by CV to see variation
    for cv_id in df['cv_id'].unique():
        cv_data = df[df['cv_id'] == cv_id]
        original_info = get_original_info(cv_id, mapping)
        
        rankings = cv_data['ranking'].tolist()
        unique_rankings = sorted(set(rankings))
        ranking_counts = cv_data['ranking'].value_counts().to_dict()
        
        # Get breakdown by pipeline
        pipeline_rankings = {}
        for pipeline in cv_data['pipeline'].unique():
            pipeline_data = cv_data[cv_data['pipeline'] == pipeline]
            pipeline_rankings[pipeline] = {
                'rankings': pipeline_data['ranking'].tolist(),
                'models': pipeline_data['model'].tolist(),
                'avg_ranking': pipeline_data['ranking'].mean()
            }
        
        # Get breakdown by model
        model_rankings = {}
        for model in cv_data['model'].unique():
            model_data = cv_data[cv_data['model'] == model]
            model_rankings[model] = {
                'rankings': model_data['ranking'].tolist(),
                'pipelines': model_data['pipeline'].tolist(),
                'avg_ranking': model_data['ranking'].mean()
            }
        
        # Calculate variance metrics
        ranking_variance = cv_data['ranking'].var()
        ranking_std = cv_data['ranking'].std()
        min_ranking = cv_data['ranking'].min()
        max_ranking = cv_data['ranking'].max()
        ranking_range = max_ranking - min_ranking
        
        analysis.append({
            'cv_id': cv_id,
            'original_id': original_info['original_id'],
            'name': original_info['original_name'],
            'total_evaluations': len(rankings),
            'unique_rankings': unique_rankings,
            'ranking_distribution': ranking_counts,
            'min_ranking': int(min_ranking),
            'max_ranking': int(max_ranking),
            'ranking_range': int(ranking_range),
            'avg_ranking': round(cv_data['ranking'].mean(), 2),
            'ranking_variance': round(ranking_variance, 2),
            'ranking_std': round(ranking_std, 2),
            'by_pipeline': pipeline_rankings,
            'by_model': model_rankings
        })
    
    # Sort by variance (most disagreement first)
    analysis.sort(key=lambda x: x['ranking_variance'], reverse=True)
    
    # Print analysis
    print("=" * 80)
    print("CV ANALYSIS: How Each Candidate is Treated Differently")
    print("=" * 80)
    print()
    
    for item in analysis:
        print(f"CV: {item['name']} ({item['original_id']})")
        print(f"  Sanitized ID: {item['cv_id']}")
        print(f"  Total Evaluations: {item['total_evaluations']}")
        print(f"  Rankings Received: {sorted(item['unique_rankings'])}")
        print(f"  Ranking Distribution: {item['ranking_distribution']}")
        print(f"  Range: {item['min_ranking']} - {item['max_ranking']} (span: {item['ranking_range']})")
        print(f"  Average: {item['avg_ranking']:.2f}")
        print(f"  Variance: {item['ranking_variance']:.2f} (Std Dev: {item['ranking_std']:.2f})")
        print()
        
        print("  By Pipeline:")
        for pipeline, data in item['by_pipeline'].items():
            print(f"    {pipeline}: {data['rankings']} (avg: {data['avg_ranking']:.2f})")
        print()
        
        print("  By Model:")
        for model, data in item['by_model'].items():
            print(f"    {model}: {data['rankings']} (avg: {data['avg_ranking']:.2f})")
        print()
        print("-" * 80)
        print()
    
    # Create summary table
    summary_df = pd.DataFrame([
        {
            'Original ID': item['original_id'],
            'Name': item['name'],
            'Sanitized ID': item['cv_id'],
            'Min': item['min_ranking'],
            'Max': item['max_ranking'],
            'Range': item['ranking_range'],
            'Avg': item['avg_ranking'],
            'Std Dev': item['ranking_std'],
            'Variance': item['ranking_variance']
        }
        for item in analysis
    ])
    
    print("\nSUMMARY TABLE (sorted by variance - most disagreement first):")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()
    
    # Save detailed analysis
    if output_file:
        output_path = experiment_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed analysis saved to: {output_path}")
    
    # Save summary CSV
    summary_csv_path = experiment_dir / "cv_differences_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary table saved to: {summary_csv_path}")
    
    # Create pivot table: CV vs Pipeline/Model combinations
    print("\n" + "=" * 80)
    print("PIVOT TABLE: Ranking by CV and Pipeline-Model Combination")
    print("=" * 80)
    
    df['pipeline_model'] = df['pipeline'] + ' / ' + df['model']
    pivot = df.pivot_table(
        values='ranking',
        index=['original_id', 'original_name'],
        columns='pipeline_model',
        aggfunc='first'
    )

    # Reorder columns: one_shot, chain_of_thought, decomposed_algorithmic, multi_layer
    pipeline_order = ['one_shot', 'chain_of_thought', 'decomposed_algorithmic', 'multi_layer']
    ordered_cols = []
    for pipeline in pipeline_order:
        pipeline_cols = [col for col in pivot.columns if col.startswith(pipeline + ' /')]
        ordered_cols.extend(sorted(pipeline_cols))

    # Reindex with ordered columns (keep any columns not matched)
    remaining_cols = [col for col in pivot.columns if col not in ordered_cols]
    pivot = pivot.reindex(columns=ordered_cols + remaining_cols)

    print(pivot.to_string())
    
    pivot_csv_path = experiment_dir / "cv_rankings_pivot.csv"
    pivot.to_csv(pivot_csv_path)
    print(f"\nPivot table saved to: {pivot_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how CVs are treated differently across models/pipelines"
    )
    parser.add_argument(
        "experiment_name",
        help="Name of the experiment directory in results/",
        nargs="?",
        default=None
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory (default: results)"
    )
    parser.add_argument(
        "--output",
        default="cv_differences_analysis.json",
        help="Output JSON file name (default: cv_differences_analysis.json)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # If no experiment name provided, use the most recent one
    if args.experiment_name is None:
        experiments = sorted(results_dir.glob("experiment_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not experiments:
            # Try looking for any directory
            experiments = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime if x.is_dir() else 0, reverse=True)
            experiments = [e for e in experiments if e.is_dir() and (e / "comparison.csv").exists()]
        
        if not experiments:
            print(f"Error: No experiments found in {results_dir}")
            print("Please specify an experiment name or ensure results directory exists.")
            sys.exit(1)
        
        experiment_dir = experiments[0]
        print(f"Using most recent experiment: {experiment_dir.name}\n")
    else:
        experiment_dir = results_dir / args.experiment_name
        if not experiment_dir.exists():
            print(f"Error: Experiment '{args.experiment_name}' not found in {results_dir}")
            sys.exit(1)
    
    analyze_experiment(experiment_dir, args.output)


if __name__ == "__main__":
    main()

