"""Multi-layer pipeline - iterative refinement approach."""
import json
import asyncio
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class MultiLayerPipeline(Pipeline):
    """Multi-layer analysis pipeline with iterative refinement."""
    
    def __init__(self, llm_provider):
        super().__init__(llm_provider, "multi_layer")
    
    async def _analyze_single_cv(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str) -> tuple:
        """Analyze a single CV independently with multi-layer approach."""
        
        # Layer 1: Evaluate each criteria separately for this CV
        criteria_evaluations = {}
        
        criteria_list = [
            ("Zero-to-One Operator", "zero_to_one"),
            ("Technical T-Shape", "technical_t_shape"),
            ("Recruitment Mastery", "recruitment_mastery")
        ]
        
        for criteria_name, criteria_key in criteria_list:
            # Extract relevant section from detailed_criteria
            criteria_section = self._extract_criteria_section(detailed_criteria, criteria_name)
            
            criteria_prompt = f"""Evaluate this candidate against the "{criteria_name}" criteria.

Job Description:
{job_ad}

Criteria Details:
{criteria_section}

Candidate CV:
{cv['content']}

Evaluate their fit to this specific criteria and rate as: Excellent, Good, Weak, or Not a Fit.

Provide your evaluation in JSON format:
{{
    "cv_id": "{cv['id']}",
    "rating": "Excellent/Good/Weak/Not a Fit"
}}"""

            criteria_response = await self.llm_provider.generate(criteria_prompt)
            
            try:
                content = criteria_response.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(content)
                criteria_evaluations[criteria_key] = {
                    "cv_id": cv['id'],
                    "rating": parsed.get("rating", "Unknown")
                }
            except (json.JSONDecodeError, KeyError) as e:
                criteria_evaluations[criteria_key] = {
                    "cv_id": cv['id'],
                    "error": str(e),
                    "raw": criteria_response.content
                }
        
        # Layer 2: Synthesize overall fit based on criteria evaluations
        synthesis_prompt = f"""Based on the individual criteria evaluations below, determine the overall fit rating (1-4) for this candidate.

Job Description:
{job_ad}

Individual Criteria Evaluations:
{json.dumps(criteria_evaluations, indent=2)}

Synthesize the three criteria evaluations into an overall fit rating:
- 4 = Excellent fit (meets all criteria at excellent level)
- 3 = Good fit (meets criteria at good level)
- 2 = Borderline fit (meets some criteria but has gaps)
- 1 = Not a fit (does not meet key criteria)

Provide your final ranking in JSON format:
{{
    "cv_id": "{cv['id']}",
    "criteria_evaluations": {{
        "zero_to_one": "{criteria_evaluations.get('zero_to_one', {}).get('rating', 'Unknown')}",
        "technical_t_shape": "{criteria_evaluations.get('technical_t_shape', {}).get('rating', 'Unknown')}",
        "recruitment_mastery": "{criteria_evaluations.get('recruitment_mastery', {}).get('rating', 'Unknown')}"
    }},
    "ranking": 4
}}"""

        synthesis_response = await self.llm_provider.generate(synthesis_prompt)
        
        # Extract name from CV content
        cv_content = cv.get("content", "")
        name = "Unknown"
        if cv_content:
            first_line = cv_content.split('\n')[0].strip()
            name = first_line.replace('#', '').replace('_', '').strip()
        
        # Parse final synthesis
        ranking = 0
        reasoning = ""
        criteria_eval_summary = {}
        try:
            content = synthesis_response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            
            # Handle different response structures
            if isinstance(parsed, dict):
                ranking_value = parsed.get("ranking", 0)
                if isinstance(ranking_value, int):
                    ranking = ranking_value
                elif isinstance(ranking_value, dict):
                    # If ranking is a dict (e.g., criteria-specific), try to extract numeric values
                    numeric_values = [v for v in ranking_value.values() if isinstance(v, (int, float))]
                    if numeric_values:
                        ranking = int(round(sum(numeric_values) / len(numeric_values)))
                    else:
                        ranking = 0
                elif isinstance(ranking_value, (float, str)):
                    try:
                        ranking = int(float(ranking_value))
                    except (ValueError, TypeError):
                        ranking = 0
                else:
                    ranking = 0
                
                reasoning_raw = parsed.get("reasoning", "")
                criteria_eval_summary = parsed.get("criteria_evaluations", {})
                
                # Handle reasoning if it's a dict (convert to string)
                if isinstance(reasoning_raw, dict):
                    reasoning = json.dumps(reasoning_raw, indent=2)
                elif isinstance(reasoning_raw, (list, tuple)):
                    reasoning = "\n".join(str(item) for item in reasoning_raw)
                else:
                    reasoning = str(reasoning_raw) if reasoning_raw else ""
            else:
                ranking = 0
                reasoning = synthesis_response.content
                criteria_eval_summary = {}
                
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract ranking from text
            import re
            ranking_match = re.search(r'"ranking"\s*:\s*(\d+)', synthesis_response.content)
            if ranking_match:
                ranking = int(ranking_match.group(1))
            else:
                ranking = 0
            reasoning = synthesis_response.content
            criteria_eval_summary = {}
        
        ranking_result = RankingResult(
            cv_id=cv['id'],
            name=name,
            ranking=ranking,
            reasoning=reasoning
        )
        
        analysis_data = {
            "layer_1_criteria_evaluations": criteria_evaluations,
            "layer_2_synthesis": {
                "criteria_evaluations": criteria_eval_summary,
                "ranking": ranking
            }
        }
        
        return ranking_result, analysis_data
    
    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform multi-layer analysis - each CV evaluated independently."""
        
        # Process each CV independently in parallel
        tasks = [self._analyze_single_cv(cv, job_ad, detailed_criteria) for cv in cv_list]
        results = await asyncio.gather(*tasks)
        
        # Separate rankings and analysis
        rankings = []
        all_analysis = {}
        for ranking_result, analysis_data in results:
            rankings.append(ranking_result)
            all_analysis[ranking_result.cv_id] = analysis_data
        
        analysis = {
            "note": "Each CV evaluated independently - 4 API calls per CV (3 criteria + 1 synthesis)",
            "total_cvs": len(cv_list),
            "per_cv_analyses": all_analysis
        }
        
        return PipelineResult(
            pipeline_name=self.name,
            provider=self.llm_provider.get_provider_name(),
            model=self.llm_provider.model,
            rankings=rankings,
            analysis=analysis,
            metadata={
                "usage": {"note": "Token usage not tracked per individual CV call"},
            }
        )
    
    def _extract_criteria_section(self, detailed_criteria: str, criteria_name: str) -> str:
        """Extract the relevant section from detailed criteria."""
        # Simple extraction - look for the criteria name
        lines = detailed_criteria.split('\n')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if criteria_name.lower() in line.lower() and '#' in line:
                start_idx = i
            elif start_idx is not None and line.strip().startswith('#') and criteria_name.lower() not in line.lower():
                end_idx = i
                break
        
        if start_idx is not None:
            if end_idx is None:
                end_idx = len(lines)
            return '\n'.join(lines[start_idx:end_idx])
        
        return detailed_criteria  # Fallback to full criteria
