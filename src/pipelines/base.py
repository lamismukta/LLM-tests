"""Base class for analysis pipelines."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel
from ..providers.base import LLMProvider


class RankingResult(BaseModel):
    """Result for a single CV ranking."""
    cv_id: str
    name: str
    ranking: int  # 1-4 (4=excellent, 3=good, 2=borderline, 1=not a fit)
    reasoning: str = ""


class PipelineResult(BaseModel):
    """Result from a pipeline execution."""
    pipeline_name: str
    provider: str
    model: str
    rankings: List[RankingResult]  # List of rankings for all CVs
    analysis: Dict[str, Any] = {}  # Additional analysis data
    metadata: Dict[str, Any] = {}


class Pipeline(ABC):
    """Abstract base class for CV analysis pipelines."""
    
    def __init__(self, llm_provider: LLMProvider, name: str):
        self.llm_provider = llm_provider
        self.name = name
    
    @abstractmethod
    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Analyze a list of CVs and return rankings.
        
        Args:
            cv_list: List of dictionaries containing CV information (id, content)
            job_ad: Job advertisement text
            detailed_criteria: Detailed hiring criteria
            
        Returns:
            PipelineResult with rankings for all CVs
        """
        pass

