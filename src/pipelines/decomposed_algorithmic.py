"""Decomposed pipeline with algorithmic aggregation - shares criteria evaluation with multi_layer."""
import json
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class DecomposedAlgorithmicPipeline(Pipeline):
    """Evaluates criteria separately (like multi_layer) but uses algorithmic aggregation instead of LLM synthesis."""

    def __init__(self, llm_provider):
        super().__init__(llm_provider, "decomposed_algorithmic")

    def _map_rating_to_score(self, rating: str) -> int:
        """Map qualitative rating to numeric score."""
        rating_lower = rating.lower().strip()

        if 'excellent' in rating_lower:
            return 4
        elif 'good' in rating_lower:
            return 3
        elif 'weak' in rating_lower or 'borderline' in rating_lower:
            return 2
        elif 'not a fit' in rating_lower or 'not fit' in rating_lower:
            return 1
        else:
            # Default to borderline if unclear
            return 2

    def _aggregate_scores(self, criteria_evaluations: Dict[str, Any]) -> tuple[int, str]:
        """Aggregate criteria scores using simple average (no weights)."""

        scores = []
        reasoning_parts = []

        for criteria_key in ['zero_to_one', 'technical_t_shape', 'recruitment_mastery']:
            eval_data = criteria_evaluations.get(criteria_key, {})

            if isinstance(eval_data, dict):
                rating = eval_data.get('rating', 'Unknown')
                score = self._map_rating_to_score(rating)
                scores.append(score)
                reasoning_parts.append(f"{criteria_key}: {rating} (score: {score})")
            else:
                # Error case
                scores.append(2)  # Default to borderline
                reasoning_parts.append(f"{criteria_key}: Error in evaluation")

        if not scores:
            return 2, "No valid criteria evaluations"

        # Simple average (rounded to nearest integer)
        avg_score = sum(scores) / len(scores)
        final_ranking = round(avg_score)

        # Ensure ranking is in valid range
        final_ranking = max(1, min(4, final_ranking))

        reasoning = (
            f"Algorithmic aggregation: Average of criteria scores = {avg_score:.2f} → {final_ranking}\n"
            + "\n".join(reasoning_parts)
        )

        return final_ranking, reasoning

    async def _evaluate_criteria(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str) -> Dict[str, Any]:
        """Evaluate each criterion separately - same as multi_layer Layer 1."""

        cv_texts = [f"CV (ID: {cv['id']}):\n{cv['content']}\n"]
        cv_block = "\n---\n\n".join(cv_texts)

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

Candidate to Evaluate:
{cv_block}

Evaluate their fit to this specific criteria and rate as: Excellent, Good, Weak, or Not a Fit.

Provide your evaluation in JSON format:
{{
    "cv_id": "{cv['id']}",
    "rating": "Excellent/Good/Weak/Not a Fit",
    "evidence": "Specific evidence from CV supporting this rating"
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
                    "rating": parsed.get("rating", "Unknown"),
                    "evidence": parsed.get("evidence", "")
                }
            except (json.JSONDecodeError, KeyError) as e:
                criteria_evaluations[criteria_key] = {
                    "cv_id": cv['id'],
                    "error": str(e),
                    "raw": criteria_response.content,
                    "rating": "Unknown"
                }

        return criteria_evaluations

    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform decomposed analysis with algorithmic aggregation."""

        rankings = []
        all_criteria_evals = {}

        for cv in cv_list:
            # Layer 1: Evaluate each criteria (3 API calls)
            criteria_evaluations = await self._evaluate_criteria(cv, job_ad, detailed_criteria)

            # Layer 2: Algorithmic aggregation (no API call)
            final_ranking, reasoning = self._aggregate_scores(criteria_evaluations)

            # Extract name from CV content
            cv_content = cv.get("content", "")
            name = "Unknown"
            if cv_content:
                first_line = cv_content.split('\n')[0].strip()
                name = first_line.replace('#', '').replace('_', '').strip()

            ranking_result = RankingResult(
                cv_id=cv['id'],
                name=name,
                ranking=final_ranking,
                reasoning=reasoning
            )
            rankings.append(ranking_result)
            all_criteria_evals[cv['id']] = criteria_evaluations

        analysis = {
            "note": "Criteria evaluated via LLM (3 API calls per CV), aggregated algorithmically (simple average)",
            "total_cvs": len(cv_list),
            "criteria_evaluations": all_criteria_evals,
            "aggregation_method": "Simple average of criteria scores (no weights)"
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
