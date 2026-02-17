"""
AI/NLP Agent: Summarization, outline generation, insight extraction.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from src.agents.base_agent import BaseAgent, AgentResult
from src.core.llm_interface import llm

class AIAgent(BaseAgent):
    """AI-powered content analysis and summarization."""

    def __init__(self):
        super().__init__("AIAgent")

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Analyze content and generate insights."""
        start = datetime.now()

        try:
            content = task.get("content", "")
            analysis_type = task.get("analysis_type", "summarization")
            llm_task_overrides = task.get("llm_task_overrides", {}) or {}

            if not content:
                raise ValueError("No content provided for analysis")

            self.log_step(f"Performing {analysis_type} on {len(content)} characters")

            # Generate summary
            summary = self._generate_summary(
                content,
                route_override=llm_task_overrides.get("summarization"),
            )

            # Extract key insights
            insights = self._extract_insights(
                content,
                route_override=llm_task_overrides.get("insight_extraction"),
            )

            # Generate outline
            outline = self._generate_outline(
                summary,
                insights,
                route_override=llm_task_overrides.get("outline_generation"),
            )

            # Extract keywords
            keywords = self._extract_keywords(content, insights)

            result = {
                "summary": summary,
                "insights": insights,
                "outline": outline,
                "keywords": keywords,
                "content_length": len(content),
                "processing_status": "completed"
            }

            self.save_to_state("analysis_result", result)

            duration = (datetime.now() - start).total_seconds()

            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result,
                duration_seconds=duration
            )

        except Exception as e:
            self.log_error(str(e))
            return AgentResult(
                agent_name=self.name,
                success=False,
                data={},
                error=str(e)
            )

    def _generate_summary(
        self,
        content: str,
        max_length: int = 500,
        route_override: Dict[str, Any] = None,
    ) -> str:
        """Generate concise summary using LLM."""
        prompt = f"""Summarize the following financial content in {max_length} words or less.
Focus on key metrics, trends, and insights.

Content:
{content[:2000]}

Summary:"""

        try:
            summary = llm.generate(
                prompt,
                system_prompt="You are a financial analyst. Provide concise, factual summaries.",
                task_type="summarization",
                route_overrides=route_override,
            )
            return summary[:max_length]
        except Exception as e:
            self.log_error(f"Summary generation failed: {str(e)}")
            return self._fallback_summary(content, max_length=max_length)

    def _extract_insights(self, content: str, route_override: Dict[str, Any] = None) -> List[str]:
        """Extract key insights from content."""
        prompt = f"""Extract 3-5 key financial insights from the following content.
Return as a JSON array of strings.

Content:
{content[:2000]}

Insights (JSON array):"""

        try:
            response = llm.generate(
                prompt,
                system_prompt="You are a financial analyst. Extract actionable insights.",
                task_type="insight_extraction",
                route_overrides=route_override,
            )
            insights = llm.extract_json(response)
            if isinstance(insights, list):
                return [str(item).strip() for item in insights if str(item).strip()][:5]
            return self._fallback_insights(content)
        except Exception as e:
            self.log_error(f"Insight extraction failed: {str(e)}")
            return self._fallback_insights(content)

    def _generate_outline(
        self,
        summary: str,
        insights: List[str],
        route_override: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Generate report outline."""
        prompt = f"""Create a concise JSON outline for a financial report.
Return a valid JSON object with section titles as keys and short descriptions as values.

Summary:
{summary}

Insights:
{json.dumps(insights)}
"""

        try:
            response = llm.generate(
                prompt,
                system_prompt="You are a financial reporting strategist. Return valid JSON only.",
                task_type="outline_generation",
                route_overrides=route_override,
            )
            parsed = llm.extract_json(response)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception as e:
            self.log_error(f"Outline generation failed: {str(e)}")

        return {
            "Executive Summary": summary,
            "Key Insights": insights,
            "Financial Metrics": [],
            "Trends and Analysis": [],
            "Recommendations": [],
            "Appendix": [],
        }

    def _extract_keywords(self, content: str, insights: List[str]) -> List[str]:
        """Extract keywords for tagging."""
        # Simple approach: use common financial keywords
        financial_keywords = [
            "revenue", "profit", "loss", "assets", "liabilities", "equity",
            "cash flow", "gross margin", "operating income", "net income",
            "growth", "trend", "increase", "decrease", "forecast", "budget"
        ]

        content_lower = content.lower()
        found_keywords = [kw for kw in financial_keywords if kw in content_lower]

        # Add insights-based keywords (first 3 words from first 3 insights)
        for insight in insights[:3]:
            words = insight.split()[:2]
            found_keywords.extend(words)

        return list(set(found_keywords))[:10]

    def _fallback_summary(self, content: str, max_length: int = 500) -> str:
        """Local fallback summary when LLM is unavailable."""
        compact = " ".join(content.split())
        if len(compact) <= max_length:
            return compact
        return compact[: max_length - 3] + "..."

    def _fallback_insights(self, content: str) -> List[str]:
        """Heuristic fallback insights when LLM is unavailable."""
        content_lower = content.lower()
        insights = []
        patterns = [
            ("revenue", "Revenue signals are present in the uploaded statement."),
            ("profit", "Profitability indicators were detected."),
            ("expense", "Expense-related entries were identified."),
            ("cash", "Cash-related entries were identified."),
            ("growth", "Growth indicators were detected in the statement."),
        ]
        for keyword, insight in patterns:
            if keyword in content_lower:
                insights.append(insight)
        if not insights:
            insights.append("Financial line items were extracted, but advanced LLM insight generation is unavailable.")
        return insights[:5]
