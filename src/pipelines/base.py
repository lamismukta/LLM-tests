"""Base class for analysis pipelines."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel
from ..providers.base import LLMProvider


class PipelineResult(BaseModel):
    """Result from a pipeline execution."""
    cv_id: str
    pipeline_name: str
    provider: str
    model: str
    analysis: Dict[str, Any]
    metadata: Dict[str, Any] = {}


class Pipeline(ABC):
    """Abstract base class for CV analysis pipelines."""
    
    def __init__(self, llm_provider: LLMProvider, name: str):
        self.llm_provider = llm_provider
        self.name = name
    
    @abstractmethod
    async def analyze(self, cv_data: Dict[str, Any]) -> PipelineResult:
        """Analyze a CV and return structured results.
        
        Args:
            cv_data: Dictionary containing CV information (id, name, content, etc.)
            
        Returns:
            PipelineResult with analysis
        """
        pass
    
    def get_base_prompt(self) -> str:
        """Get the base prompt for CV analysis."""
        return """Analyze the following CV and provide a comprehensive assessment. 
Focus on:
1. Overall fit and quality
2. Key strengths and achievements
3. Relevant experience and skills
4. Potential concerns or gaps
5. Recommendation (Excellent, Good, Borderline, Not a Fit)

Provide your analysis in a structured format."""

