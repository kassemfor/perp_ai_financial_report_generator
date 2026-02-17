"""
Backend Agent: File parsing, data processing, PDF generation.
"""

import json
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.agents.base_agent import BaseAgent, AgentResult
from src.backend.file_parser import FileParser
from src.config import CONFIG

class BackendAgent(BaseAgent):
    """Handles file parsing, validation, and PDF report assembly."""

    def __init__(self):
        super().__init__("BackendAgent")
        self.parser = FileParser()

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Process uploaded file and prepare data."""
        start = datetime.now()

        try:
            file_path = task.get("file_path")
            if not file_path or not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            self.log_step(f"Parsing file: {file_path}")
            parsed_data = self.parser.parse(
                file_path=file_path,
                enable_ocr=bool(task.get("enable_ocr", True)),
                ocr_language=task.get("ocr_language", "eng"),
            )

            # Validate and clean data
            self.log_step("Validating data")
            validated_data = self._validate_data(parsed_data)

            # Extract financial metrics if applicable
            self.log_step("Extracting financial data")
            financial_data = self._extract_financial_data(validated_data)

            self.save_to_state("parsed_data", validated_data)
            self.save_to_state("financial_data", financial_data)
            self.save_to_state("parsed_text", validated_data.get("text", validated_data.get("content", "")))

            result = {
                "file_format": parsed_data.get("format"),
                "file_name": parsed_data.get("file_name"),
                "data_shape": self._get_data_shape(validated_data),
                "financial_records": len(financial_data.get("records", [])),
                "validation_status": "passed",
                "extraction_source": parsed_data.get("extraction_source", "structured"),
                "extraction_confidence": parsed_data.get("extraction_confidence", 0.0),
                "ocr": parsed_data.get("ocr", {}),
                "rfs_status": parsed_data.get("rfs_statement", {}).get("status", "unknown"),
                "rfs_quality_score": parsed_data.get("rfs_statement", {}).get("quality_score", 0.0),
                "rfs_line_items": parsed_data.get("rfs_statement", {}).get("summary", {}).get("line_items_detected", 0),
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

    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parsed data."""
        # Check for required fields, handle missing values
        if "data" in data:
            if isinstance(data["data"], list):
                # Remove empty rows
                data["data"] = [row for row in data["data"] if any(row.values())]
        return data

    def _extract_financial_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial-specific data (accounts, amounts, periods)."""
        records = data.get("data", [])
        financial_records = []

        # Look for common financial fields
        financial_cols = ["account", "amount", "account_code", "period", "company"]

        for record in records:
            if any(col in str(record).lower() for col in financial_cols):
                financial_records.append(record)

        # Fallback for unstructured documents (TXT, PDF OCR, image OCR): leverage RFS lines.
        if not financial_records:
            rfs_lines = data.get("rfs_statement", {}).get("statement_lines", [])
            for line in rfs_lines:
                financial_records.append(
                    {
                        "account": line.get("line_item", ""),
                        "amount": line.get("value", ""),
                        "period": line.get("period", ""),
                        "source": line.get("source", ""),
                        "confidence": line.get("confidence", 0.0),
                    }
                )

        return {
            "records": financial_records,
            "record_count": len(financial_records),
            "estimated_periods": self._extract_periods(financial_records)
        }

    def _extract_periods(self, records: list) -> list:
        """Extract unique periods from financial records."""
        periods = set()
        period_fields = ["period", "date", "month", "year"]

        for record in records:
            for key, value in record.items():
                if any(pf in key.lower() for pf in period_fields):
                    if value:
                        periods.add(str(value))

        return sorted(list(periods))

    def _get_data_shape(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Describe data dimensions."""
        if "data" in data:
            if isinstance(data["data"], list):
                return {
                    "rows": len(data["data"]),
                    "columns": len(data.get("columns", []))
                }
            elif isinstance(data["data"], dict):
                return {
                    "keys": len(data["data"]),
                    "nested_sheets": len(data.get("sheets", {}))
                }
        return {"rows": 0, "columns": 0}

    def generate_pdf(self, report_content: Dict[str, Any], output_path: str) -> bool:
        """Generate PDF report with branding."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

            self.log_step(f"Generating PDF: {output_path}")

            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor(CONFIG.branding.primary_color),
                alignment=TA_CENTER,
                spaceAfter=30
            )

            # Title
            story.append(Paragraph(report_content.get("title", "Financial Report"), title_style))
            story.append(Spacer(1, 0.3*inch))

            # Company branding
            story.append(Paragraph(f"<b>{CONFIG.branding.company_name}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.2*inch))

            # Content sections
            for section in report_content.get("sections", []):
                story.append(Paragraph(section.get("title", ""), styles["Heading2"]))
                story.append(Paragraph(section.get("content", ""), styles["Normal"]))
                story.append(Spacer(1, 0.2*inch))

            # Footer
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"<i>{CONFIG.branding.footer_text}</i>", styles["Normal"]))

            doc.build(story)
            self.log_step("PDF generated successfully")
            return True

        except ImportError:
            self.log_error("reportlab not installed")
            return False
        except Exception as e:
            self.log_error(f"PDF generation failed: {str(e)}")
            return False
