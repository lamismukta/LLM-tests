"""Chain-of-thought pipeline - step-by-step reasoning."""
import json
from typing import Dict, Any
from .base import Pipeline, PipelineResult


class ChainOfThoughtPipeline(Pipeline):
    """Chain-of-thought reasoning pipeline with explicit steps."""
    
    def __init__(self, llm_provider):
        super().__init__(llm_provider, "chain_of_thought")
    
    async def analyze(self, cv_data: Dict[str, Any]) -> PipelineResult:
        """Perform chain-of-thought analysis with explicit reasoning steps."""
        prompt = f"""{self.get_base_prompt()}

CV Content:
{cv_data['content']}

Please analyze this CV using a step-by-step chain-of-thought approach:

Step 1: First, identify and list the key experiences and roles mentioned in the CV.
Step 2: Evaluate the quality and relevance of these experiences.
Step 3: Assess the technical and soft skills demonstrated.
Step 4: Identify any gaps or concerns.
Step 5: Synthesize all information to arrive at an overall rating.

After completing your step-by-step analysis, provide your final assessment in JSON format:
{{
    "step_by_step_reasoning": {{
        "step_1_experiences": "analysis",
        "step_2_quality": "analysis",
        "step_3_skills": "analysis",
        "step_4_gaps": "analysis",
        "step_5_synthesis": "analysis"
    }},
    "overall_rating": "Excellent|Good|Borderline|Not a Fit",
    "key_strengths": ["strength1", "strength2", ...],
    "relevant_experience": "description",
    "skills_assessment": "description",
    "concerns_or_gaps": ["concern1", "concern2", ...],
    "final_reasoning": "detailed reasoning for the rating"
}}"""

        response = await self.llm_provider.generate(prompt)
        
        # Try to parse JSON from response
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
        except json.JSONDecodeError:
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

