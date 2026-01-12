"""One-shot prompt pipeline - single comprehensive analysis."""
import json
import asyncio
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class OneShotPipeline(Pipeline):
    """Single prompt analysis pipeline."""
    
    def __init__(self, llm_provider):
        super().__init__(llm_provider, "one_shot")
    
    async def _analyze_single_cv(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str) -> RankingResult:
        """Analyze a single CV independently."""
        prompt = f"""You are evaluating a candidate for a Founding Operator role. 

Job Description:
{job_ad}

Detailed Hiring Criteria:
{detailed_criteria}

You will be evaluating this candidate against three key criteria:
1. Zero-to-One Operator
2. Technical T-Shape  
3. Recruitment Mastery

Candidate CV:
{cv['content']}

Provide a fit rating from 1-4:
- 4 = Excellent fit
- 3 = Good fit
- 2 = Borderline fit
- 1 = Not a fit

Provide your response in JSON format:
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
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            
            # Handle different response structures
            if isinstance(parsed, dict):
                # Direct ranking
                if "ranking" in parsed:
                    ranking_value = parsed["ranking"]
                    if isinstance(ranking_value, int):
                        ranking = ranking_value
                    elif isinstance(ranking_value, dict):
                        # If ranking is a dict (e.g., criteria-specific), try to extract numeric values
                        numeric_values = [v for v in ranking_value.values() if isinstance(v, (int, float))]
                        if numeric_values:
                            # Use average if multiple numeric values, or first value
                            ranking = int(round(sum(numeric_values) / len(numeric_values)))
                        else:
                            ranking = 0
                    elif isinstance(ranking_value, (float, str)):
                        # Try to convert to int
                        try:
                            ranking = int(float(ranking_value))
                        except (ValueError, TypeError):
                            ranking = 0
                    else:
                        ranking = 0
                
                # Check if ranking might be nested
                if ranking == 0 and "result" in parsed:
                    result = parsed["result"]
                    if isinstance(result, dict) and "ranking" in result:
                        ranking = result["ranking"] if isinstance(result["ranking"], int) else 0
                
                reasoning_raw = parsed.get("reasoning", "")
                # Handle reasoning if it's a dict (convert to string)
                if isinstance(reasoning_raw, dict):
                    reasoning = json.dumps(reasoning_raw, indent=2)
                elif isinstance(reasoning_raw, (list, tuple)):
                    reasoning = "\n".join(str(item) for item in reasoning_raw)
                else:
                    reasoning = str(reasoning_raw) if reasoning_raw else ""
            else:
                # If parsed is not a dict, try to extract ranking from text
                ranking = 0
                reasoning = response.content
                
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract ranking from text
            import re
            # Look for ranking pattern in text
            ranking_match = re.search(r'"ranking"\s*:\s*(\d+)', response.content)
            if ranking_match:
                ranking = int(ranking_match.group(1))
            else:
                ranking = 0
            reasoning = response.content
        
        return RankingResult(
            cv_id=cv['id'],
            name=name,
            ranking=ranking,
            reasoning=reasoning
        )
    
    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform one-shot analysis of all CVs - one API call per CV."""
        
        # Process each CV independently in parallel
        tasks = [self._analyze_single_cv(cv, job_ad, detailed_criteria) for cv in cv_list]
        rankings = await asyncio.gather(*tasks)
        
        # Calculate total usage
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        # Note: We can't track individual usage per CV call easily without modifying the response structure
        # For now, we'll note that usage is per-CV
        analysis = {
            "note": "Each CV evaluated independently in separate API calls",
            "total_cvs": len(cv_list)
        }
        
        return PipelineResult(
            pipeline_name=self.name,
            provider=self.llm_provider.get_provider_name(),
            model=self.llm_provider.model,
            rankings=list(rankings),
            analysis=analysis,
            metadata={
                "usage": total_usage,
                "note": "Token usage not tracked per individual CV call"
            }
        )
