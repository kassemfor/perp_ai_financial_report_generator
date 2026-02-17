"""
Visualization Agent: Chart generation, KPI graphics, color themes.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
import io
import base64
from src.agents.base_agent import BaseAgent, AgentResult
from src.config import CONFIG

class VisualizationAgent(BaseAgent):
    """Creates charts, KPIs, and visual analytics."""

    def __init__(self):
        super().__init__("VisualizationAgent")

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Generate visualizations from data."""
        start = datetime.now()

        try:
            data = task.get("data", {})
            chart_types = task.get("chart_types", ["line", "bar"])

            self.log_step(f"Generating {len(chart_types)} visualizations")

            visualizations = {}

            # Generate requested chart types
            for chart_type in chart_types:
                if chart_type == "line":
                    visualizations["revenue_trend"] = self._create_line_chart(data)
                elif chart_type == "bar":
                    visualizations["category_comparison"] = self._create_bar_chart(data)
                elif chart_type == "pie":
                    visualizations["composition"] = self._create_pie_chart(data)

            # Generate KPI cards
            kpis = self._create_kpi_cards(data)
            visualizations["kpis"] = kpis

            self.save_to_state("visualizations", visualizations)

            result = {
                "chart_count": len([v for v in visualizations if v != "kpis"]),
                "kpi_count": len(kpis),
                "chart_types_generated": list(visualizations.keys()),
                "branding_applied": CONFIG.branding.company_name
            }

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

    def _create_line_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create line chart for trends."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig, ax = plt.subplots(figsize=(10, 6))

            # Generate sample trend data
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            values = [100, 110, 105, 120, 130, 140]

            ax.plot(months, values, marker="o", color=CONFIG.branding.primary_color, linewidth=2)
            ax.fill_between(range(len(months)), values, alpha=0.3, color=CONFIG.branding.primary_color)
            ax.set_title("Revenue Trend", fontsize=14, fontweight="bold")
            ax.set_xlabel("Month")
            ax.set_ylabel("Amount ($K)")
            ax.grid(True, alpha=0.3)

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close(fig)

            return {
                "type": "line",
                "title": "Revenue Trend",
                "image": f"data:image/png;base64,{img_base64}"
            }
        except Exception as e:
            self.log_error(f"Line chart creation failed: {str(e)}")
            return {"type": "line", "error": str(e)}

    def _create_bar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create bar chart for comparisons."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig, ax = plt.subplots(figsize=(10, 6))

            categories = ["Product A", "Product B", "Product C", "Product D"]
            values = [450, 380, 520, 390]
            colors = [CONFIG.branding.primary_color, CONFIG.branding.secondary_color] * 2

            ax.bar(categories, values, color=colors, alpha=0.8)
            ax.set_title("Category Comparison", fontsize=14, fontweight="bold")
            ax.set_ylabel("Amount ($K)")
            ax.grid(True, alpha=0.3, axis="y")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close(fig)

            return {
                "type": "bar",
                "title": "Category Comparison",
                "image": f"data:image/png;base64,{img_base64}"
            }
        except Exception as e:
            self.log_error(f"Bar chart creation failed: {str(e)}")
            return {"type": "bar", "error": str(e)}

    def _create_pie_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create pie chart for composition."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig, ax = plt.subplots(figsize=(8, 8))

            labels = ["Revenue", "COGS", "OpEx", "Other"]
            sizes = [40, 25, 20, 15]
            colors = [
                CONFIG.branding.primary_color,
                CONFIG.branding.secondary_color,
                CONFIG.branding.accent_color,
                "#cccccc"
            ]

            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title("Composition", fontsize=14, fontweight="bold")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close(fig)

            return {
                "type": "pie",
                "title": "Composition",
                "image": f"data:image/png;base64,{img_base64}"
            }
        except Exception as e:
            self.log_error(f"Pie chart creation failed: {str(e)}")
            return {"type": "pie", "error": str(e)}

    def _create_kpi_cards(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create KPI metric cards."""
        kpis = [
            {
                "label": "Total Revenue",
                "value": "$2.45M",
                "change": "+12.5%",
                "color": CONFIG.branding.primary_color
            },
            {
                "label": "Gross Margin",
                "value": "42.3%",
                "change": "+2.1%",
                "color": CONFIG.branding.secondary_color
            },
            {
                "label": "Operating Income",
                "value": "$980K",
                "change": "+8.3%",
                "color": CONFIG.branding.accent_color
            },
            {
                "label": "Net Profit",
                "value": "$640K",
                "change": "+5.7%",
                "color": CONFIG.branding.primary_color
            }
        ]
        return kpis
