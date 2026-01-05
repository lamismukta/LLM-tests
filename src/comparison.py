"""Comparison and evaluation framework for pipeline results."""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from .pipelines.base import PipelineResult


class ComparisonFramework:
    """Framework for comparing pipeline results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: List[PipelineResult], experiment_name: str = None):
        """Save pipeline results to disk."""
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Save individual results
        for result in results:
            filename = f"{result.cv_id}_{result.pipeline_name}_{result.provider}_{result.model}.json"
            filepath = experiment_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result.model_dump(), f, indent=2)
        
        # Save summary
        summary = self._create_summary(results)
        summary_path = experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save comparison table
        comparison_df = self.create_comparison_dataframe(results)
        comparison_path = experiment_dir / "comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"Results saved to {experiment_dir}")
        return experiment_dir
    
    def _create_summary(self, results: List[PipelineResult]) -> Dict[str, Any]:
        """Create a summary of results."""
        summary = {
            "experiment_timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "pipelines": {},
            "providers": {},
            "models": {},
            "cv_ids": list(set(r.cv_id for r in results))
        }
        
        # Group by pipeline
        for result in results:
            pipeline_key = result.pipeline_name
            if pipeline_key not in summary["pipelines"]:
                summary["pipelines"][pipeline_key] = {
                    "count": 0,
                    "models": set(),
                    "total_tokens": 0
                }
            summary["pipelines"][pipeline_key]["count"] += 1
            summary["pipelines"][pipeline_key]["models"].add(result.model)
            if result.metadata.get("usage"):
                summary["pipelines"][pipeline_key]["total_tokens"] += result.metadata["usage"].get("total_tokens", 0)
        
        # Convert sets to lists for JSON serialization
        for pipeline in summary["pipelines"].values():
            pipeline["models"] = list(pipeline["models"])
        
        return summary
    
    def create_comparison_dataframe(self, results: List[PipelineResult]) -> pd.DataFrame:
        """Create a pandas DataFrame for easy comparison."""
        rows = []
        for result in results:
            row = {
                "cv_id": result.cv_id,
                "pipeline": result.pipeline_name,
                "provider": result.provider,
                "model": result.model,
                "overall_rating": self._extract_rating(result.analysis),
                "total_tokens": result.metadata.get("usage", {}).get("total_tokens", 0),
                "prompt_tokens": result.metadata.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": result.metadata.get("usage", {}).get("completion_tokens", 0),
            }
            
            # Add analysis fields if available
            if isinstance(result.analysis, dict):
                row["key_strengths_count"] = len(result.analysis.get("key_strengths", []))
                row["concerns_count"] = len(result.analysis.get("concerns_or_gaps", []))
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _extract_rating(self, analysis: Dict[str, Any]) -> str:
        """Extract overall rating from analysis."""
        if isinstance(analysis, dict):
            # Try different possible keys
            for key in ["overall_rating", "final_rating", "rating"]:
                if key in analysis:
                    return str(analysis[key])
            # For multi-layer, check layer_3_synthesis
            if "layer_3_synthesis" in analysis:
                return self._extract_rating(analysis["layer_3_synthesis"])
        return "Unknown"
    
    def compare_pipelines(self, results: List[PipelineResult], cv_id: str = None) -> pd.DataFrame:
        """Compare results across pipelines for a specific CV or all CVs."""
        if cv_id:
            filtered_results = [r for r in results if r.cv_id == cv_id]
        else:
            filtered_results = results
        
        comparison_data = []
        for result in filtered_results:
            comparison_data.append({
                "cv_id": result.cv_id,
                "pipeline": result.pipeline_name,
                "model": result.model,
                "rating": self._extract_rating(result.analysis),
                "tokens_used": result.metadata.get("usage", {}).get("total_tokens", 0),
                "analysis": json.dumps(result.analysis, indent=2)
            })
        
        return pd.DataFrame(comparison_data)
    
    def load_results(self, experiment_name: str) -> List[PipelineResult]:
        """Load results from a previous experiment."""
        experiment_dir = self.results_dir / experiment_name
        if not experiment_dir.exists():
            raise ValueError(f"Experiment {experiment_name} not found")
        
        results = []
        for filepath in experiment_dir.glob("*.json"):
            if filepath.name == "summary.json":
                continue
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(PipelineResult(**data))
        
        return results

