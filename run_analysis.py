#!/usr/bin/env python3
"""Main script to run CV analysis pipelines and compare results."""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.providers.openai_provider import OpenAIProvider
from src.pipelines.one_shot import OneShotPipeline
from src.pipelines.chain_of_thought import ChainOfThoughtPipeline
from src.pipelines.multi_layer import MultiLayerPipeline
from src.comparison import ComparisonFramework
from src.pipelines.base import PipelineResult


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_cv_data(data_path: str) -> List[Dict[str, Any]]:
    """Load CV data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)


async def run_pipeline(
    pipeline: Pipeline,
    cv_data: Dict[str, Any],
    progress_callback=None
) -> PipelineResult:
    """Run a pipeline on a single CV."""
    try:
        result = await pipeline.analyze(cv_data)
        if progress_callback:
            progress_callback(f"Completed {pipeline.name} for {cv_data['id']}")
        return result
    except Exception as e:
        print(f"Error running {pipeline.name} on {cv_data['id']}: {e}")
        raise


async def run_experiment(
    config: Dict[str, Any],
    cv_data: List[Dict[str, Any]],
    models: List[str] = None,
    pipelines: List[str] = None,
    cv_ids: List[str] = None
) -> List[PipelineResult]:
    """Run a complete experiment across models, pipelines, and CVs."""
    load_dotenv()
    
    # Filter CVs if specified
    if cv_ids:
        cv_data = [cv for cv in cv_data if cv['id'] in cv_ids]
    
    # Default to all models and pipelines if not specified
    if models is None:
        models = config['llm_providers']['openai']['models']
    
    if pipelines is None:
        pipelines = [name for name, settings in config['pipelines'].items() if settings.get('enabled', True)]
    
    results = []
    total_tasks = len(models) * len(pipelines) * len(cv_data)
    completed = 0
    
    print(f"Running experiment:")
    print(f"  Models: {models}")
    print(f"  Pipelines: {pipelines}")
    print(f"  CVs: {len(cv_data)}")
    print(f"  Total tasks: {total_tasks}\n")
    
    for model in models:
        # Create provider for this model
        provider = OpenAIProvider(
            model=model,
            temperature=config['llm_providers']['openai']['temperature'],
            max_tokens=config['llm_providers']['openai']['max_tokens']
        )
        
        for pipeline_name in pipelines:
            # Create pipeline
            if pipeline_name == "one_shot":
                pipeline = OneShotPipeline(provider)
            elif pipeline_name == "chain_of_thought":
                pipeline = ChainOfThoughtPipeline(provider)
            elif pipeline_name == "multi_layer":
                pipeline = MultiLayerPipeline(provider)
            else:
                print(f"Unknown pipeline: {pipeline_name}")
                continue
            
            # Run on all CVs
            tasks = [run_pipeline(pipeline, cv) for cv in cv_data]
            pipeline_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            for result in pipeline_results:
                if isinstance(result, Exception):
                    print(f"Error: {result}")
                else:
                    results.append(result)
                    completed += 1
                    print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
    
    return results


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM CV analysis pipelines")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data", default="data/cvs_revised_v2.json", help="Path to CV data file")
    parser.add_argument("--models", nargs="+", help="Specific models to test (default: all in config)")
    parser.add_argument("--pipelines", nargs="+", choices=["one_shot", "chain_of_thought", "multi_layer"],
                       help="Specific pipelines to run (default: all enabled)")
    parser.add_argument("--cv-ids", nargs="+", help="Specific CV IDs to analyze (default: all)")
    parser.add_argument("--experiment-name", help="Name for this experiment")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test on first 3 CVs")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load CV data
    cv_data = load_cv_data(args.data)
    
    # Quick test mode
    if args.quick_test:
        cv_data = cv_data[:3]
        print("Running in quick test mode (first 3 CVs only)\n")
    
    # Run experiment
    results = await run_experiment(
        config=config,
        cv_data=cv_data,
        models=args.models,
        pipelines=args.pipelines,
        cv_ids=args.cv_ids
    )
    
    # Save and compare results
    framework = ComparisonFramework(results_dir=config.get('analysis', {}).get('results_dir', 'results'))
    experiment_dir = framework.save_results(results, experiment_name=args.experiment_name)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    comparison_df = framework.create_comparison_dataframe(results)
    print("\nComparison by Pipeline:")
    print(comparison_df.groupby('pipeline')['overall_rating'].value_counts().unstack(fill_value=0))
    
    print("\nToken Usage by Pipeline:")
    print(comparison_df.groupby('pipeline')['total_tokens'].agg(['mean', 'sum', 'count']))
    
    print(f"\nDetailed results saved to: {experiment_dir}")
    print(f"Comparison CSV: {experiment_dir / 'comparison.csv'}")


if __name__ == "__main__":
    asyncio.run(main())

