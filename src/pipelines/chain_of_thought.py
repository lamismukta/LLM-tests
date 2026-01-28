"""Chain-of-thought pipeline - step-by-step reasoning."""
import json
import asyncio
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class ChainOfThoughtPipeline(Pipeline):
    """Chain-of-thought reasoning pipeline with explicit steps."""
    
    def __init__(self, llm_provider):
        super().__init__(llm_provider, "chain_of_thought")
    
    async def _analyze_single_cv(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str) -> RankingResult:
        """Analyze a single CV independently with chain-of-thought."""
        prompt = f"""You are a recruiter evaluating a candidate for a Founding Operator role.  

Job Description:
{job_ad}

Detailed Hiring Criteria:
{detailed_criteria}

Candidate CV:
{cv['content']}

Follow this step-by-step process:

Step 1: Evaluate Zero-to-One Operator fit
- Assess their experience building operational systems from scratch
- Look for evidence of "diagnose, build, then scale" mindset
- Rate: Excellent / Good / Weak / Not a Fit

Step 2: Evaluate Technical T-Shape fit
- Assess technical/analytical depth and ability to partner with engineers
- Look for evidence of AI tool usage and automation experience
- Rate: Excellent / Good / Weak / Not a Fit

Step 3: Evaluate Recruitment Mastery fit
- Assess end-to-end recruitment experience
- Look for evidence of building hiring pipelines from scratch
- Rate: Excellent / Good / Weak / Not a Fit

Step 4: Synthesize overall fit
- Consider all three criteria together
- Determine overall rating: 4 (Excellent), 3 (Good), 2 (Borderline), 1 (Not a Fit)

After completing your step-by-step analysis, provide your final ranking in JSON format:
{{
    "cv_id": "{cv['id']}",
    "ranking": 4
}}"""

        response = await self.llm_provider.generate(prompt)
        
        # Extract name from CV content
        cv_content = cv.get("content", "")
        name = "Unknown"
        if cv_content:
            first_line = cv_content.split('\n')[0].strip()
            name = first_line.replace('#', '').replace('_', '').strip()
        
        # Try to parse JSON from response
        ranking = 0
        reasoning = ""
        step_analysis = {}
        try:
            content = response.content.strip()
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
                
                reasoning = parsed.get("reasoning", "")
                step_analysis = parsed.get("step_by_step_analysis", {})
            else:
                ranking = 0
                reasoning = response.content
                step_analysis = {}
                
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract ranking from text
            import re
            ranking_match = re.search(r'"ranking"\s*:\s*(\d+)', response.content)
            if ranking_match:
                ranking = int(ranking_match.group(1))
            else:
                ranking = 0
            reasoning = response.content
            step_analysis = {}
        
        ranking_result = RankingResult(
            cv_id=cv['id'],
            name=name,
            ranking=ranking,
            reasoning=reasoning
        )
        return ranking_result, step_analysis
    
    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform chain-of-thought analysis - one API call per CV."""
        
        # Process each CV independently in parallel
        tasks = [self._analyze_single_cv(cv, job_ad, detailed_criteria) for cv in cv_list]
        results = await asyncio.gather(*tasks)
        
        # Separate rankings and analysis
        rankings = []
        all_analysis = {}
        for ranking_result, step_analysis in results:
            rankings.append(ranking_result)
            all_analysis[ranking_result.cv_id] = step_analysis
        
        analysis = {
            "note": "Each CV evaluated independently in separate API calls",
            "total_cvs": len(cv_list),
            "step_by_step_analyses": all_analysis
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
