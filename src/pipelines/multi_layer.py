"""Multi-layer pipeline - iterative refinement approach."""
import json
from typing import Dict, Any
from .base import Pipeline, PipelineResult


class MultiLayerPipeline(Pipeline):
    """Multi-layer analysis pipeline with iterative refinement."""
    
    def __init__(self, llm_provider):
        super().__init__(llm_provider, "multi_layer")
    
    async def analyze(self, cv_data: Dict[str, Any]) -> PipelineResult:
        """Perform multi-layer analysis with iterative refinement."""
        
        # Layer 1: Initial extraction
        extraction_prompt = f"""Extract key information from this CV:

CV Content:
{cv_data['content']}

Extract and structure:
1. All professional roles and companies
2. Key achievements and metrics
3. Technical and soft skills
4. Education background

Provide this in a structured JSON format."""

        extraction_response = await self.llm_provider.generate(extraction_prompt)
        
        # Parse extraction
        try:
            content = extraction_response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            extracted_info = json.loads(content)
        except json.JSONDecodeError:
            extracted_info = {"raw": extraction_response.content}
        
        # Layer 2: Evaluation
        evaluation_prompt = f"""Based on the extracted information below, evaluate the candidate:

Extracted Information:
{json.dumps(extracted_info, indent=2)}

Evaluate:
1. Quality and depth of experience
2. Relevance of skills
3. Achievement impact
4. Overall profile strength

Provide evaluation in JSON format."""
        
        evaluation_response = await self.llm_provider.generate(evaluation_prompt)
        
        # Parse evaluation
        try:
            content = evaluation_response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            evaluation = json.loads(content)
        except json.JSONDecodeError:
            evaluation = {"raw": evaluation_response.content}
        
        # Layer 3: Final synthesis
        synthesis_prompt = f"""Synthesize the following information to provide a final assessment:

Original CV:
{cv_data['content']}

Extracted Information:
{json.dumps(extracted_info, indent=2)}

Evaluation:
{json.dumps(evaluation, indent=2)}

Provide final assessment in JSON format:
{{
    "overall_rating": "Excellent|Good|Borderline|Not a Fit",
    "key_strengths": ["strength1", "strength2", ...],
    "relevant_experience": "description",
    "skills_assessment": "description",
    "concerns_or_gaps": ["concern1", "concern2", ...],
    "reasoning": "detailed reasoning for the rating",
    "extraction_summary": "summary of extracted info",
    "evaluation_summary": "summary of evaluation"
}}"""
        
        synthesis_response = await self.llm_provider.generate(synthesis_prompt)
        
        # Parse final synthesis
        try:
            content = synthesis_response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            final_analysis = json.loads(content)
        except json.JSONDecodeError:
            final_analysis = {"raw_response": synthesis_response.content}
        
        # Combine all layers
        analysis = {
            "layer_1_extraction": extracted_info,
            "layer_2_evaluation": evaluation,
            "layer_3_synthesis": final_analysis,
            "final_rating": final_analysis.get("overall_rating", "Unknown")
        }
        
        # Calculate total usage
        def safe_get_usage(response, key):
            return response.usage.get(key, 0) if response.usage else 0
        
        total_usage = {
            "prompt_tokens": (
                safe_get_usage(extraction_response, "prompt_tokens") +
                safe_get_usage(evaluation_response, "prompt_tokens") +
                safe_get_usage(synthesis_response, "prompt_tokens")
            ),
            "completion_tokens": (
                safe_get_usage(extraction_response, "completion_tokens") +
                safe_get_usage(evaluation_response, "completion_tokens") +
                safe_get_usage(synthesis_response, "completion_tokens")
            ),
            "total_tokens": (
                safe_get_usage(extraction_response, "total_tokens") +
                safe_get_usage(evaluation_response, "total_tokens") +
                safe_get_usage(synthesis_response, "total_tokens")
            )
        }
        
        return PipelineResult(
            cv_id=cv_data['id'],
            pipeline_name=self.name,
            provider=self.llm_provider.get_provider_name(),
            model=self.llm_provider.model,
            analysis=analysis,
            metadata={
                "usage": total_usage,
                "layer_usage": {
                    "extraction": extraction_response.usage or {},
                    "evaluation": evaluation_response.usage or {},
                    "synthesis": synthesis_response.usage or {}
                }
            }
        )

