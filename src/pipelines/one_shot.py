"""One-shot prompt pipeline - single comprehensive analysis."""
import json
from typing import Dict, Any
from .base import Pipeline, PipelineResult


class OneShotPipeline(Pipeline):
    """Single prompt analysis pipeline."""
    
    def __init__(self, llm_provider):
        super().__init__(llm_provider, "one_shot")
    
    async def analyze(self, cv_data: Dict[str, Any]) -> PipelineResult:
        """Perform one-shot analysis of the CV."""
        prompt = f"""{self.get_base_prompt()}

CV Content:
{cv_data['content']}

Please provide your analysis in JSON format with the following structure:
{{
    "overall_rating": "Excellent|Good|Borderline|Not a Fit",
    "key_strengths": ["strength1", "strength2", ...],
    "relevant_experience": "description",
    "skills_assessment": "description",
    "concerns_or_gaps": ["concern1", "concern2", ...],
    "reasoning": "detailed reasoning for the rating"
}}"""

        response = await self.llm_provider.generate(prompt)
        
        # Try to parse JSON from response
        try:
            # Extract JSON from markdown code blocks if present
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: return raw content if JSON parsing fails
            analysis = {"raw_response": response.content}
        
        return PipelineResult(
            cv_id=cv_data['id'],
            pipeline_name=self.name,
            provider=self.llm_provider.get_provider_name(),
            model=self.llm_provider.model,
            analysis=analysis,
            metadata={
                "usage": response.usage,
                "response_metadata": response.metadata
            }
        )

