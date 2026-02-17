"""
Reader-first Streamlit interface with dynamic OCR and RFS statement extraction.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project-root imports work when Streamlit executes from src/frontend.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.coordinator_agent import CoordinatorAgent
from src.backend.file_parser import FileParser
from src.config import LLMBackend
from src.core.llm_interface import llm
from src.core.state import MEMORY

UPLOAD_FORMATS = ["csv", "txt", "pdf", "docx", "xlsx", "png", "jpg", "jpeg", "tiff", "bmp", "webp"]
BACKEND_OPTIONS = [LLMBackend.OLLAMA.value, LLMBackend.OPENAI.value, LLMBackend.GEMINI.value, LLMBackend.LM_STUDIO.value, LLMBackend.ANTHROPIC.value]
DEFAULT_MODEL_BY_BACKEND = {
    LLMBackend.OLLAMA.value: "mistral:latest",
    LLMBackend.OPENAI.value: "gpt-4o-mini",
    LLMBackend.GEMINI.value: "gemini-1.5-flash",
    LLMBackend.LM_STUDIO.value: "local-model",
    LLMBackend.ANTHROPIC.value: "claude-3-5-sonnet-20241022",
}
DEFAULT_BASE_URL_BY_BACKEND = {
    LLMBackend.OLLAMA.value: "http://localhost:11434/v1",
    LLMBackend.OPENAI.value: "https://api.openai.com/v1",
    LLMBackend.GEMINI.value: "https://generativelanguage.googleapis.com",
    LLMBackend.LM_STUDIO.value: "http://localhost:1234/v1",
    LLMBackend.ANTHROPIC.value: "https://api.anthropic.com",
}
DOCUMENT_TEMPLATES = [
    "Cover Letter",
    "Email Signature",
    "Report",
    "Proposal",
    "Business Plan",
    "Handbook",
    "Ebook",
    "Case Study",
    "White Paper",
]
DEFAULT_OUTLINE = [
    {"title": "Cover", "tag": "Cover"},
    {"title": "Table of Content", "tag": "Table of Content"},
    {"title": "Introduction", "tag": "Introduction"},
    {"title": "Unique Angle", "tag": "Content"},
    {"title": "Business Application", "tag": "Content"},
    {"title": "Business Model", "tag": "Content"},
]
TEMPLATE_OUTLINES: Dict[str, List[Dict[str, str]]] = {
    "Cover Letter": [
        {"title": "Cover", "tag": "Cover"},
        {"title": "Recipient Context", "tag": "Introduction"},
        {"title": "Opening Statement", "tag": "Content"},
        {"title": "Core Qualifications", "tag": "Content"},
        {"title": "Closing", "tag": "Summary"},
    ],
    "Email Signature": [
        {"title": "Profile Block", "tag": "Cover"},
        {"title": "Contact Details", "tag": "Content"},
        {"title": "Brand Line", "tag": "Content"},
        {"title": "Optional Disclaimer", "tag": "Summary"},
    ],
    "Report": [*DEFAULT_OUTLINE],
    "Proposal": [
        {"title": "Cover", "tag": "Cover"},
        {"title": "Executive Summary", "tag": "Introduction"},
        {"title": "Problem Statement", "tag": "Content"},
        {"title": "Proposed Solution", "tag": "Content"},
        {"title": "Scope and Deliverables", "tag": "Content"},
        {"title": "Timeline", "tag": "Content"},
        {"title": "Budget", "tag": "Content"},
        {"title": "Next Steps", "tag": "Summary"},
    ],
    "Business Plan": [
        {"title": "Cover", "tag": "Cover"},
        {"title": "Executive Summary", "tag": "Introduction"},
        {"title": "Market Analysis", "tag": "Content"},
        {"title": "Business Model", "tag": "Content"},
        {"title": "Go-To-Market Strategy", "tag": "Content"},
        {"title": "Financial Projections", "tag": "Content"},
        {"title": "Risks and Mitigation", "tag": "Content"},
        {"title": "Conclusion", "tag": "Summary"},
    ],
    "Handbook": [
        {"title": "Cover", "tag": "Cover"},
        {"title": "Purpose and Scope", "tag": "Introduction"},
        {"title": "Policies", "tag": "Content"},
        {"title": "Procedures", "tag": "Content"},
        {"title": "Roles and Responsibilities", "tag": "Content"},
        {"title": "FAQ", "tag": "Content"},
        {"title": "Revision History", "tag": "Summary"},
    ],
    "Ebook": [
        {"title": "Cover", "tag": "Cover"},
        {"title": "Table of Content", "tag": "Table of Content"},
        {"title": "Introduction", "tag": "Introduction"},
        {"title": "Chapter 1", "tag": "Content"},
        {"title": "Chapter 2", "tag": "Content"},
        {"title": "Chapter 3", "tag": "Content"},
        {"title": "Key Takeaways", "tag": "Summary"},
    ],
    "Case Study": [
        {"title": "Cover", "tag": "Cover"},
        {"title": "Client Context", "tag": "Introduction"},
        {"title": "Challenge", "tag": "Content"},
        {"title": "Approach", "tag": "Content"},
        {"title": "Results", "tag": "Content"},
        {"title": "Lessons Learned", "tag": "Content"},
        {"title": "Next Steps", "tag": "Summary"},
    ],
    "White Paper": [
        {"title": "Cover", "tag": "Cover"},
        {"title": "Abstract", "tag": "Introduction"},
        {"title": "Problem Definition", "tag": "Content"},
        {"title": "Methodology", "tag": "Content"},
        {"title": "Findings", "tag": "Content"},
        {"title": "Implications", "tag": "Content"},
        {"title": "References", "tag": "Summary"},
    ],
}
TEMPLATE_SUBTITLES = {
    "Cover Letter": "Personalized role-focused narrative with measurable value points.",
    "Email Signature": "Concise identity and contact block optimized for brand consistency.",
    "Report": "Data-backed analysis with statement-quality extraction and visual insights.",
    "Proposal": "Problem-solution narrative with scope, timeline, and commercial intent.",
    "Business Plan": "Strategy and financial model with execution pathways and risk control.",
    "Handbook": "Policy-driven operational guide with clear ownership and procedures.",
    "Ebook": "Long-form structured knowledge document with chapter-based storytelling.",
    "Case Study": "Evidence-led client story with challenge, execution, and outcomes.",
    "White Paper": "Research-driven technical argument with methodology and references.",
}
TEMPLATE_AI_HINTS = {
    "Cover Letter": "Keep tone professional and concise, 1-page equivalent, first person allowed.",
    "Email Signature": "Keep it compact, contact-first, and brand-clean with no long paragraphs.",
    "Report": "Use analytical tone, include quantified findings and clear executive takeaways.",
    "Proposal": "Emphasize client value, concrete scope, timeline realism, and persuasive clarity.",
    "Business Plan": "Balance strategy with numbers, include assumptions and risk-aware language.",
    "Handbook": "Use policy language, imperative procedures, and role-specific accountability.",
    "Ebook": "Use chapter flow, educational clarity, and coherent narrative transitions.",
    "Case Study": "Use factual chronology with measurable before/after outcomes.",
    "White Paper": "Use neutral research tone, define terms, cite assumptions and constraints.",
}
TEMPLATE_KEYWORDS = {
    "Cover Letter": ["job", "role", "position", "hiring", "application", "resume", "cv"],
    "Email Signature": ["email signature", "signature", "contact card", "footer"],
    "Report": ["report", "analysis", "dashboard", "summary", "market", "finance", "annual"],
    "Proposal": ["proposal", "pitch", "scope", "deliverables", "client", "bid"],
    "Business Plan": ["business plan", "startup", "go-to-market", "revenue model", "strategy"],
    "Handbook": ["handbook", "policy", "procedure", "guideline", "onboarding"],
    "Ebook": ["ebook", "chapter", "guidebook", "long-form", "book"],
    "Case Study": ["case study", "customer story", "implementation", "outcome"],
    "White Paper": ["white paper", "research", "methodology", "standard", "compliance", "rfs"],
}

st.set_page_config(
    page_title="RFS Reader Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 8% 0%, rgba(70, 150, 176, 0.18), transparent 45%),
    radial-gradient(circle at 94% 5%, rgba(255, 183, 82, 0.2), transparent 36%),
    linear-gradient(180deg, #edf2f5 0%, #f8fafc 100%);
}

.block-container {
  max-width: 1440px;
  padding-top: 1.2rem;
  padding-bottom: 1.2rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #103141 0%, #1e5368 100%);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
}

[data-testid="stSidebar"] * {
  color: #e6f3f8 !important;
}

.header-bar {
  background: linear-gradient(135deg, #1b4458 0%, #214f67 100%);
  border-radius: 14px;
  padding: 12px 16px;
  color: #ffffff;
  margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(26, 66, 87, 0.18);
}

.header-title {
  font-family: 'Manrope', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  margin: 0;
  line-height: 1.15;
}

.header-subtitle {
  margin-top: 0.3rem;
  color: #cae4ef;
  font-size: 0.95rem;
}

.hero-title {
  font-family: 'Manrope', sans-serif;
  font-size: 2.7rem;
  line-height: 1.05;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 0.2rem;
  color: #0f2f40;
}

.hero-note {
  color: #4c6f80;
  font-size: 1rem;
  margin-bottom: 1rem;
}

.input-shell {
  border: 1px solid #b8caef;
  border-radius: 14px;
  background: linear-gradient(180deg, #f4f8ff 0%, #eef2ff 100%);
  padding: 14px 16px;
  margin-bottom: 1.1rem;
}

.outline-card, .preview-card {
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(16, 54, 74, 0.13);
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 10px 26px rgba(14, 46, 63, 0.12);
}

.outline-title, .preview-title {
  font-family: 'Manrope', sans-serif;
  font-size: 1.7rem;
  font-weight: 800;
  color: #172f3d;
  margin-bottom: 0.2rem;
}

.preview-subtitle {
  color: #5f7786;
  margin-bottom: 0.7rem;
}

.outline-row {
  display: grid;
  grid-template-columns: 40px 1fr auto;
  align-items: center;
  gap: 10px;
  border: 1px solid #d7e2e8;
  border-radius: 12px;
  padding: 10px 12px;
  margin-bottom: 8px;
  background: #fbfdff;
}

.outline-index {
  text-align: center;
  color: #70828d;
  font-weight: 600;
}

.outline-text {
  font-weight: 600;
  color: #263c48;
}

.outline-tag {
  font-size: 0.75rem;
  border-radius: 999px;
  padding: 2px 10px;
  background: #d7e8f8;
  color: #2d5d83;
  font-weight: 700;
  white-space: nowrap;
}

.rfs-pass {
  color: #1f7a4f;
  font-weight: 700;
}

.rfs-review {
  color: #af6a00;
  font-weight: 700;
}

div[data-testid="stMetric"] {
  background: #f4f7fa;
  border: 1px solid #d9e3e8;
  border-radius: 14px;
  padding: 10px 12px;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 12px;
  font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  padding: 0.35rem 0.8rem;
  font-weight: 600;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _segmented(label: str, options: List[str], default: str, key: str) -> str:
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default, key=key)
    return st.radio(label, options=options, horizontal=True, index=options.index(default), key=key)


def _save_uploaded_file(uploaded_file: Any) -> Path:
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / uploaded_file.name
    with open(destination, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return destination


def _to_numeric_amount(value: str) -> float:
    raw = (value or "").strip()
    if not raw:
        return 0.0

    negative = False
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
    cleaned = re.sub(r"[^0-9.\-]", "", raw)
    if cleaned.count("-") > 1:
        cleaned = cleaned.replace("-", "")
    if cleaned in {"", "-", ".", "-."}:
        return 0.0

    try:
        parsed = float(cleaned)
    except ValueError:
        return 0.0
    return -abs(parsed) if negative else parsed


def _statement_rows(parsed_docs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for doc in parsed_docs:
        file_name = doc.get("file_name", "unknown")
        for line in doc.get("rfs_statement", {}).get("statement_lines", []):
            amount_numeric = _to_numeric_amount(line.get("value", "0"))
            rows.append(
                {
                    "file_name": file_name,
                    "line_item": line.get("line_item", "Unlabeled"),
                    "value": line.get("value", ""),
                    "amount_numeric": amount_numeric,
                    "period": line.get("period") or "Current",
                    "source": line.get("source", ""),
                    "confidence": float(line.get("confidence", 0.0)),
                }
            )

    if rows:
        return pd.DataFrame(rows)

    return pd.DataFrame(
        [
            {
                "file_name": "sample",
                "line_item": "Market Growth",
                "value": "32",
                "amount_numeric": 32,
                "period": "2024",
                "source": "sample",
                "confidence": 0.75,
            },
            {
                "file_name": "sample",
                "line_item": "Consumer Satisfaction",
                "value": "87",
                "amount_numeric": 87,
                "period": "2024",
                "source": "sample",
                "confidence": 0.75,
            },
            {
                "file_name": "sample",
                "line_item": "Revenue Increase",
                "value": "24",
                "amount_numeric": 24,
                "period": "2024",
                "source": "sample",
                "confidence": 0.75,
            },
            {
                "file_name": "sample",
                "line_item": "Conversion Rate",
                "value": "4.2",
                "amount_numeric": 4.2,
                "period": "2024",
                "source": "sample",
                "confidence": 0.75,
            },
        ]
    )


def _template_base_outline(template_choice: str) -> List[Dict[str, str]]:
    return [dict(item) for item in TEMPLATE_OUTLINES.get(template_choice, DEFAULT_OUTLINE)]


def _template_subtitle(template_choice: str) -> str:
    return TEMPLATE_SUBTITLES.get(template_choice, TEMPLATE_SUBTITLES["Report"])


def _template_ai_hint(template_choice: str) -> str:
    return TEMPLATE_AI_HINTS.get(template_choice, TEMPLATE_AI_HINTS["Report"])


def _recommend_template(
    topic_text: str,
    manual_text: str,
    parsed_docs: List[Dict[str, Any]],
    selected_template: str,
) -> Dict[str, str]:
    combined = " ".join(
        [
            (topic_text or "").strip().lower(),
            (manual_text or "").strip().lower(),
            " ".join(str(doc.get("file_name", "")).lower() for doc in parsed_docs),
            " ".join(str(doc.get("format", "")).lower() for doc in parsed_docs),
        ]
    )
    scores = {template: 0.0 for template in DOCUMENT_TEMPLATES}

    for template, keywords in TEMPLATE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                scores[template] += 1.8 if " " in keyword else 1.0

    doc_count = len(parsed_docs)
    line_items = sum(
        int(doc.get("rfs_statement", {}).get("summary", {}).get("line_items_detected", 0))
        for doc in parsed_docs
    )
    if doc_count >= 1:
        scores["Report"] += 1.2
    if doc_count >= 2:
        scores["White Paper"] += 0.8
    if line_items >= 8:
        scores["Report"] += 1.0
        scores["White Paper"] += 0.9
    if line_items >= 15:
        scores["Business Plan"] += 0.7

    best_template = max(scores, key=lambda name: scores[name]) if scores else selected_template
    if scores.get(best_template, 0.0) <= 0.0:
        best_template = selected_template or "Report"

    if best_template == selected_template:
        reason = "Current selection already matches the detected intent."
    else:
        reason = f"Detected intent and document signals align best with {best_template}."
    return {"template": best_template, "reason": reason}


def _derive_outline(
    topic: str,
    parsed_docs: List[Dict[str, Any]],
    template_choice: str,
) -> List[Dict[str, str]]:
    outline = _template_base_outline(template_choice)
    seen = {item["title"].lower() for item in outline}

    for doc in parsed_docs:
        for line in doc.get("rfs_statement", {}).get("statement_lines", []):
            title = (line.get("line_item", "") or "").strip()
            if len(title) < 4:
                continue
            key = title.lower()
            if key in seen:
                continue
            outline.append({"title": title[:50], "tag": "Content"})
            seen.add(key)
            if len(outline) >= 14:
                return outline

    if topic.strip():
        topic_title = " ".join(topic.strip().split()[:4]).title()
        key = topic_title.lower()
        if topic_title and key not in seen:
            outline.append({"title": f"{topic_title} Insights", "tag": "Content"})

    return outline


def _manual_text_doc(parser: FileParser, text: str) -> Dict[str, Any]:
    confidence = parser._estimate_extraction_confidence(text=text, ocr_used=False)
    return {
        "format": "txt",
        "file_name": "manual_input.txt",
        "text": text,
        "tables": parser._extract_tables_from_text(text),
        "ocr": {"enabled": False, "used": False, "status": "not_required", "language": "eng"},
        "extraction_source": "plain_text",
        "extraction_confidence": confidence,
        "rfs_statement": parser._build_rfs_statement(text=text, source="plain_text", confidence=confidence),
    }


def _compute_metrics(parsed_docs: List[Dict[str, Any]]) -> Dict[str, float]:
    if not parsed_docs:
        return {
            "docs": 0,
            "avg_confidence": 0.0,
            "avg_rfs_quality": 0.0,
            "line_items": 0,
            "ocr_coverage": 0.0,
        }

    confidences = [float(doc.get("extraction_confidence", 0.0)) for doc in parsed_docs]
    qualities = [float(doc.get("rfs_statement", {}).get("quality_score", 0.0)) for doc in parsed_docs]
    line_items = sum(int(doc.get("rfs_statement", {}).get("summary", {}).get("line_items_detected", 0)) for doc in parsed_docs)
    ocr_docs = sum(1 for doc in parsed_docs if doc.get("ocr", {}).get("used"))

    return {
        "docs": len(parsed_docs),
        "avg_confidence": sum(confidences) / len(confidences),
        "avg_rfs_quality": sum(qualities) / len(qualities),
        "line_items": line_items,
        "ocr_coverage": (ocr_docs / len(parsed_docs)) * 100.0,
    }


def _build_manual_routing_payload(
    route_summary_backend: str,
    route_summary_model: str,
    route_insight_backend: str,
    route_insight_model: str,
    route_outline_backend: str,
    route_outline_model: str,
    use_shared_credentials_for_routes: bool,
    llm_api_key: str,
    llm_base_url: str,
) -> Dict[str, Dict[str, Any]]:
    """Build explicit per-function route map from sidebar selections."""
    route_payload = {
        "summarization": {
            "backend": route_summary_backend,
            "model_name": route_summary_model,
        },
        "insight_extraction": {
            "backend": route_insight_backend,
            "model_name": route_insight_model,
        },
        "outline_generation": {
            "backend": route_outline_backend,
            "model_name": route_outline_model,
        },
    }
    if use_shared_credentials_for_routes:
        for route in route_payload.values():
            route["api_key"] = llm_api_key.strip()
            route["base_url"] = llm_base_url.strip()
    return route_payload


def _estimate_demand_profile(parsed_docs: List[Dict[str, Any]], topic_text: str, manual_text: str) -> Dict[str, Any]:
    """Estimate request demand/complexity for auto model routing."""
    docs = len(parsed_docs)
    extracted_chars = 0
    extracted_line_items = 0
    ocr_docs = 0
    for doc in parsed_docs:
        extracted_chars += len((doc.get("text") or doc.get("content") or "").strip())
        extracted_line_items += int(doc.get("rfs_statement", {}).get("summary", {}).get("line_items_detected", 0))
        if doc.get("ocr", {}).get("used"):
            ocr_docs += 1

    manual_chars = len((manual_text or "").strip()) + len((topic_text or "").strip())
    total_chars = extracted_chars + manual_chars

    score = 0.0
    score += min(30.0, docs * 8.0)
    score += min(35.0, total_chars / 260.0)
    score += min(25.0, extracted_line_items * 1.2)
    score += min(10.0, ocr_docs * 4.0)
    score = round(min(100.0, score), 1)

    if score >= 70:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"

    return {
        "score": score,
        "level": level,
        "documents": docs,
        "characters": total_chars,
        "line_items": extracted_line_items,
        "ocr_docs": ocr_docs,
    }


def _resolve_auto_routing(
    manual_routing: Dict[str, Dict[str, Any]],
    demand_profile: Dict[str, Any],
    routing_policy: str,
) -> Dict[str, Dict[str, Any]]:
    """Resolve final routing by demand level and policy."""
    level = demand_profile.get("level", "low")

    summary_route = dict(manual_routing["summarization"])
    insight_route = dict(manual_routing["insight_extraction"])
    outline_route = dict(manual_routing["outline_generation"])

    if routing_policy == "quality_first":
        return {
            "summarization": summary_route,
            "insight_extraction": insight_route,
            "outline_generation": outline_route,
        }

    if routing_policy == "cost_saver":
        if level == "high":
            return {
                "summarization": summary_route,
                "insight_extraction": insight_route,
                "outline_generation": summary_route,
            }
        return {
            "summarization": summary_route,
            "insight_extraction": summary_route,
            "outline_generation": summary_route,
        }

    # balanced policy
    if level == "low":
        return {
            "summarization": summary_route,
            "insight_extraction": summary_route,
            "outline_generation": summary_route,
        }
    if level == "medium":
        return {
            "summarization": summary_route,
            "insight_extraction": insight_route,
            "outline_generation": summary_route,
        }
    return {
        "summarization": summary_route,
        "insight_extraction": insight_route,
        "outline_generation": outline_route,
    }


def _routing_payload_to_df(routing_payload: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Render routing payload as a compact table."""
    rows = []
    for fn in ["summarization", "insight_extraction", "outline_generation"]:
        route = routing_payload.get(fn, {})
        rows.append(
            {
                "Function": fn,
                "Backend": route.get("backend", ""),
                "Model": route.get("model_name", ""),
            }
        )
    return pd.DataFrame(rows)


def _default_section_style() -> Dict[str, Any]:
    """Default visual style for section preview."""
    return {
        "font_size": 16,
        "text_color": "#1e3442",
        "bg_color": "#ffffff",
        "accent_color": "#4E8DA3",
        "alignment": "left",
        "line_spacing": 1.5,
    }


def _seed_section_content(title: str, parsed_docs: List[Dict[str, Any]]) -> str:
    """Seed section content from extracted lines."""
    if not parsed_docs:
        return f"Add content for **{title}**.\n\n- Key point 1\n- Key point 2"

    title_terms = {term for term in re.findall(r"[A-Za-z]+", title.lower()) if len(term) > 2}
    matches = []
    for doc in parsed_docs:
        for line in doc.get("rfs_statement", {}).get("statement_lines", []):
            line_item = str(line.get("line_item", "")).strip()
            if not line_item:
                continue
            lower_item = line_item.lower()
            overlap = sum(1 for term in title_terms if term in lower_item)
            if overlap > 0:
                matches.append((overlap, line_item, line.get("value", ""), line.get("period", "")))

    if not matches:
        fallback = []
        for doc in parsed_docs:
            for line in doc.get("rfs_statement", {}).get("statement_lines", [])[:4]:
                fallback.append(
                    f"- {line.get('line_item', 'Line Item')}: {line.get('value', '')} {line.get('period', '')}".strip()
                )
            if fallback:
                break
        if fallback:
            return "\n".join(fallback)
        return f"Extracted content available. Start drafting {title} here."

    matches.sort(key=lambda item: item[0], reverse=True)
    bullets = [f"- {item}: {value} {period}".strip() for _, item, value, period in matches[:6]]
    return "\n".join(bullets)


def _sections_from_outline(
    outline: List[Dict[str, str]],
    parsed_docs: List[Dict[str, Any]],
    existing_sections: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build editable section objects from outline."""
    existing_by_id: Dict[str, Dict[str, Any]] = {}
    existing_by_title: Dict[str, Dict[str, Any]] = {}
    for section in existing_sections or []:
        section_id = str(section.get("id", "")).strip()
        if section_id:
            existing_by_id[section_id] = section
        title_key = str(section.get("title", "")).strip().lower()
        if title_key:
            existing_by_title[title_key] = section

    sections = []
    for index, item in enumerate(outline, start=1):
        title = str(item.get("title", f"Section {index}")).strip() or f"Section {index}"
        tag = str(item.get("tag", "Content")).strip() or "Content"
        section_id = str(item.get("id", "")).strip()
        reused = existing_by_id.get(section_id) if section_id else None
        if reused is None:
            reused = existing_by_title.get(title.lower())
        content = reused.get("content", "") if reused else _seed_section_content(title, parsed_docs)
        existing_style = reused.get("style", {}) if reused else {}
        if not isinstance(existing_style, dict):
            existing_style = {}
        style = {**_default_section_style(), **existing_style}
        sections.append(
            {
                "id": str(reused.get("id", "")) if reused else f"sec_{index}_{uuid4().hex[:8]}",
                "order": index,
                "title": title[:80],
                "tag": tag[:40],
                "content": content,
                "content_format": str(reused.get("content_format", "markdown")) if reused else "markdown",
                "style": style,
            }
        )
    return sections


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_inline_markdown(line: str) -> str:
    escaped = _escape_html(line)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    return escaped


def _markdown_to_html(text: str) -> str:
    lines = (text or "").splitlines()
    if not lines:
        return "<p>No content yet for this topic.</p>"

    html_parts: List[str] = []
    in_ul = False
    in_ol = False

    def _close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            html_parts.append("</ul>")
            in_ul = False
        if in_ol:
            html_parts.append("</ol>")
            in_ol = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            _close_lists()
            html_parts.append("<br>")
            continue

        if stripped == "---":
            _close_lists()
            html_parts.append("<hr>")
            continue

        if stripped.startswith("### "):
            _close_lists()
            html_parts.append(f"<h3>{_format_inline_markdown(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            _close_lists()
            html_parts.append(f"<h2>{_format_inline_markdown(stripped[3:])}</h2>")
            continue
        if stripped.startswith("# "):
            _close_lists()
            html_parts.append(f"<h1>{_format_inline_markdown(stripped[2:])}</h1>")
            continue
        if stripped.startswith("> "):
            _close_lists()
            html_parts.append(f"<blockquote>{_format_inline_markdown(stripped[2:])}</blockquote>")
            continue

        if re.match(r"^[-*]\s+", stripped):
            if in_ol:
                html_parts.append("</ol>")
                in_ol = False
            if not in_ul:
                html_parts.append("<ul>")
                in_ul = True
            html_parts.append(f"<li>{_format_inline_markdown(re.sub(r'^[-*]\\s+', '', stripped, count=1))}</li>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            if in_ul:
                html_parts.append("</ul>")
                in_ul = False
            if not in_ol:
                html_parts.append("<ol>")
                in_ol = True
            html_parts.append(f"<li>{_format_inline_markdown(re.sub(r'^\\d+\\.\\s+', '', stripped, count=1))}</li>")
            continue

        _close_lists()
        html_parts.append(f"<p>{_format_inline_markdown(stripped)}</p>")

    _close_lists()
    return "\n".join(html_parts)


def _render_section_preview(section: Dict[str, Any]) -> None:
    """Render section content with user-selected style."""
    style_data = section.get("style", {})
    if not isinstance(style_data, dict):
        style_data = {}
    style = {**_default_section_style(), **style_data}
    content = str(section.get("content", "")).strip()
    if not content:
        content = "No content yet for this topic."
    content_format = str(section.get("content_format", "markdown")).lower()
    safe_content = _markdown_to_html(content) if content_format == "markdown" else _escape_html(content).replace("\n", "<br>")

    st.markdown(
        f"""
<div style="
  background:{style.get('bg_color', '#ffffff')};
  color:{style.get('text_color', '#1e3442')};
  font-size:{int(style.get('font_size', 16))}px;
  line-height:{float(style.get('line_spacing', 1.5))};
  text-align:{style.get('alignment', 'left')};
  border-left:5px solid {style.get('accent_color', '#4E8DA3')};
  border-radius:14px;
  padding:16px 18px;
  box-shadow:0 6px 18px rgba(20,44,58,0.08);
">
  {safe_content}
</div>
""",
        unsafe_allow_html=True,
    )


def _sync_outline_from_sections() -> None:
    """Normalize section state and keep reader outline in sync."""
    raw_sections = st.session_state.get("reader_sections", [])
    normalized_sections: List[Dict[str, Any]] = []
    for index, section in enumerate(raw_sections, start=1):
        section_id = str(section.get("id", "")).strip() or f"sec_{index}_{uuid4().hex[:8]}"
        title = str(section.get("title", f"Section {index}")).strip() or f"Section {index}"
        tag = str(section.get("tag", "Content")).strip() or "Content"
        style = section.get("style", {})
        if not isinstance(style, dict):
            style = {}
        normalized_sections.append(
            {
                "id": section_id,
                "order": index,
                "title": title[:80],
                "tag": tag[:40],
                "content": str(section.get("content", "")),
                "content_format": str(section.get("content_format", "markdown") or "markdown"),
                "style": {**_default_section_style(), **style},
            }
        )

    if not normalized_sections:
        normalized_sections = _sections_from_outline(DEFAULT_OUTLINE, parsed_docs=[])

    st.session_state.reader_sections = normalized_sections
    st.session_state.reader_outline = [{"title": section["title"], "tag": section["tag"]} for section in normalized_sections]
    selected = int(st.session_state.get("reader_selected_section", 0))
    st.session_state.reader_selected_section = max(0, min(selected, len(normalized_sections) - 1))


def _update_section(section_id: str, updates: Dict[str, Any]) -> None:
    """Update a section by ID and sync derived outline state."""
    sections = st.session_state.get("reader_sections", [])
    for section in sections:
        if section.get("id") == section_id:
            section.update(updates)
            break
    st.session_state.reader_sections = sections
    _sync_outline_from_sections()


def _fallback_section_ai_edit(action: str, content: str, title: str) -> str:
    """Local fallback transforms when LLM is unavailable."""
    text = (content or "").strip()
    if not text:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]

    if action == "Shorten for executives":
        if sentences:
            return " ".join(sentences[: min(3, len(sentences))])
        return "\n".join(lines[: min(4, len(lines))])

    if action == "Convert to bullet points":
        bullets = lines if lines else sentences
        if not bullets:
            bullets = [text]
        return "\n".join(f"- {item.lstrip('- ').strip()}" for item in bullets[:8] if item.strip())

    if action == "Expand with analysis":
        return (
            f"{text}\n\n"
            f"Additional context for {title}:\n"
            "- Add one quantitative trend.\n"
            "- Add one risk/constraint.\n"
            "- Add one execution recommendation."
        )

    if action == "Apply RFS compliance tone":
        return (
            f"{title} (RFS-ready)\n"
            f"{text}\n\n"
            "Compliance check: values should reference period, source, and confidence where possible."
        )

    return text


def _render_outline_panel() -> None:
    _sync_outline_from_sections()
    sections = st.session_state.reader_sections

    st.markdown(
        '<div class="outline-card"><div class="outline-title">Refine your outline</div><div class="preview-subtitle">Select a topic, reorder it, and edit details on the right.</div></div>',
        unsafe_allow_html=True,
    )

    if not sections:
        st.info("No topics available yet.")
        return

    selected_index = int(st.session_state.get("reader_selected_section", 0))
    selected_index = max(0, min(selected_index, len(sections) - 1))
    selected_id = sections[selected_index]["id"]
    option_ids = [section["id"] for section in sections]
    option_labels = {section["id"]: f"{section['order']}. {section['title']}  ({section['tag']})" for section in sections}

    picked_id = st.radio(
        "Topics",
        options=option_ids,
        index=option_ids.index(selected_id),
        format_func=lambda section_id: option_labels.get(section_id, section_id),
        key="reader_topic_picker",
        label_visibility="collapsed",
    )
    if picked_id != selected_id:
        st.session_state.reader_selected_section = option_ids.index(picked_id)
        st.rerun()

    selected_index = int(st.session_state.get("reader_selected_section", 0))
    selected_index = max(0, min(selected_index, len(sections) - 1))

    st.markdown("#### Reorder Topics")
    sortables_available = False
    sort_items = None
    try:
        from streamlit_sortables import sort_items as _sort_items

        sortables_available = True
        sort_items = _sort_items
    except Exception:
        sortables_available = False
        sort_items = None

    reorder_mode = _segmented(
        "Reorder mode",
        ["Drag & Drop", "Order Grid"],
        "Drag & Drop" if sortables_available else "Order Grid",
        key="outline_reorder_mode",
    )

    if reorder_mode == "Drag & Drop" and sortables_available and sort_items:
        display_items = [f"{s['order']:02d} | {s['title']} [{s['tag']}] | {s['id']}" for s in sections]
        reordered_items = sort_items(display_items, direction="vertical", key="outline_drag_sort")
        if reordered_items and reordered_items != display_items:
            id_by_item = {item: item.rsplit("|", 1)[-1].strip() for item in display_items}
            section_by_id = {section["id"]: section for section in sections}
            selected_section_id = sections[selected_index]["id"]
            reordered_sections: List[Dict[str, Any]] = []
            for item in reordered_items:
                section_id = id_by_item.get(item)
                if section_id and section_id in section_by_id:
                    reordered_sections.append(section_by_id[section_id])
            if len(reordered_sections) == len(sections):
                st.session_state.reader_sections = reordered_sections
                st.session_state.reader_selected_section = next(
                    (idx for idx, sec in enumerate(reordered_sections) if sec["id"] == selected_section_id),
                    0,
                )
                _sync_outline_from_sections()
                st.rerun()
    else:
        if reorder_mode == "Drag & Drop" and not sortables_available:
            st.info("Install `streamlit-sortables` for native drag-and-drop. Order Grid is active for now.")
        reorder_df = pd.DataFrame(
            [
                {"id": section["id"], "order": section["order"], "title": section["title"], "tag": section["tag"]}
                for section in sections
            ]
        )
        edited_reorder_df = st.data_editor(
            reorder_df,
            use_container_width=True,
            hide_index=True,
            key="outline_order_grid",
            column_config={
                "id": None,
                "order": st.column_config.NumberColumn("Order", min_value=1, step=1),
                "title": st.column_config.TextColumn("Topic", disabled=True),
                "tag": st.column_config.TextColumn("Tag", disabled=True),
            },
            disabled=["title", "tag"],
        )
        if st.button("Apply Reorder", use_container_width=True):
            order_map = {
                str(row.get("id", "")): int(row.get("order", idx + 1))
                for idx, row in enumerate(edited_reorder_df.to_dict("records"))
            }
            selected_section_id = sections[selected_index]["id"]
            reordered_sections = sorted(
                sections,
                key=lambda sec: (order_map.get(str(sec.get("id", "")), int(sec.get("order", 9999))), int(sec.get("order", 9999))),
            )
            st.session_state.reader_sections = reordered_sections
            st.session_state.reader_selected_section = next(
                (idx for idx, sec in enumerate(reordered_sections) if sec["id"] == selected_section_id),
                0,
            )
            _sync_outline_from_sections()
            st.success("Topic order updated.")
            st.rerun()

    move_up_col, move_down_col, remove_col = st.columns(3)
    with move_up_col:
        if st.button("Move Up", use_container_width=True, disabled=selected_index == 0):
            sections[selected_index - 1], sections[selected_index] = sections[selected_index], sections[selected_index - 1]
            st.session_state.reader_sections = sections
            st.session_state.reader_selected_section = selected_index - 1
            _sync_outline_from_sections()
            st.rerun()
    with move_down_col:
        if st.button("Move Down", use_container_width=True, disabled=selected_index >= len(sections) - 1):
            sections[selected_index + 1], sections[selected_index] = sections[selected_index], sections[selected_index + 1]
            st.session_state.reader_sections = sections
            st.session_state.reader_selected_section = selected_index + 1
            _sync_outline_from_sections()
            st.rerun()
    with remove_col:
        if st.button("Delete Topic", use_container_width=True, disabled=len(sections) <= 1):
            sections.pop(selected_index)
            st.session_state.reader_sections = sections
            st.session_state.reader_selected_section = max(0, selected_index - 1)
            _sync_outline_from_sections()
            st.rerun()

    st.markdown("#### Add Topic")
    add_title_col, add_tag_col = st.columns([2.2, 1.4])
    with add_title_col:
        new_page = st.text_input(
            "Topic title",
            key="new_outline_page",
            label_visibility="collapsed",
            placeholder="Add new topic",
        )
    with add_tag_col:
        new_tag = st.selectbox(
            "Tag",
            ["Content", "Introduction", "Table of Content", "Cover", "Appendix", "Summary"],
            key="new_outline_tag",
            label_visibility="collapsed",
        )

    if st.button("Add New Topic", use_container_width=True):
        title = new_page.strip()
        if not title:
            st.warning("Enter a topic title before adding.")
        else:
            sections.append(
                {
                    "id": f"sec_{len(sections) + 1}_{uuid4().hex[:8]}",
                    "order": len(sections) + 1,
                    "title": title[:80],
                    "tag": new_tag,
                    "content": _seed_section_content(title, st.session_state.reader_docs),
                    "content_format": "markdown",
                    "style": _default_section_style(),
                }
            )
            st.session_state.reader_sections = sections
            st.session_state.reader_selected_section = len(sections) - 1
            st.session_state.new_outline_page = ""
            _sync_outline_from_sections()
            st.rerun()


def _render_section_editor(
    reader_title: str,
    parsed_docs: List[Dict[str, Any]],
    template_choice: str,
    llm_runtime_settings: Dict[str, Any],
) -> None:
    _sync_outline_from_sections()
    sections = st.session_state.reader_sections
    if not sections:
        st.info("No topic selected.")
        return

    selected_index = int(st.session_state.get("reader_selected_section", 0))
    selected_index = max(0, min(selected_index, len(sections) - 1))
    st.session_state.reader_selected_section = selected_index
    section = sections[selected_index]
    section_id = section["id"]
    content_key = f"section_content_{section_id}"
    format_key = f"section_format_{section_id}"

    metrics = _compute_metrics(parsed_docs)
    st.markdown(
        f"""
<div class="preview-card">
  <div class="preview-title">{reader_title}</div>
  <div class="preview-subtitle">Template: {template_choice} | Editing topic: {section["title"]} ({section["tag"]})</div>
</div>
""",
        unsafe_allow_html=True,
    )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    with metric_col_1:
        st.metric("RFS Quality", f"{metrics['avg_rfs_quality']:.1f}%")
    with metric_col_2:
        st.metric("Confidence", f"{metrics['avg_confidence'] * 100:.1f}%")
    with metric_col_3:
        st.metric("Line Items", int(metrics["line_items"]))
    with metric_col_4:
        st.metric("OCR Coverage", f"{metrics['ocr_coverage']:.1f}%")

    content_tab, style_tab, ai_tab, preview_tab = st.tabs(
        ["Topic Content", "Format & Design", "AI Assist", "Live Preview"]
    )

    with content_tab:
        refresh_key = f"refresh_content_{section_id}"
        if st.session_state.pop(refresh_key, False):
            st.session_state[content_key] = section.get("content", "")
            st.session_state[format_key] = section.get("content_format", "markdown")
        if content_key not in st.session_state:
            st.session_state[content_key] = section.get("content", "")
        if format_key not in st.session_state:
            st.session_state[format_key] = section.get("content_format", "markdown")

        title_value = st.text_input(
            "Topic title",
            value=section["title"],
            key=f"section_title_{section_id}",
        )
        tag_options = ["Cover", "Table of Content", "Introduction", "Content", "Appendix", "Summary"]
        tag_value = st.selectbox(
            "Topic tag",
            options=tag_options,
            index=tag_options.index(section["tag"]) if section["tag"] in tag_options else 3,
            key=f"section_tag_{section_id}",
        )
        format_choice = st.selectbox(
            "Content format",
            options=["markdown", "plain"],
            index=0 if st.session_state.get(format_key, "markdown") == "markdown" else 1,
            key=format_key,
            help="Markdown enables headings, bullets, emphasis, and richer preview rendering.",
        )

        toolbar_col_1, toolbar_col_2, toolbar_col_3, toolbar_col_4 = st.columns(4)
        with toolbar_col_1:
            if st.button("H1", key=f"fmt_h1_{section_id}", use_container_width=True):
                st.session_state[content_key] = f"{st.session_state.get(content_key, '').rstrip()}\n# Heading".strip()
                st.rerun()
        with toolbar_col_2:
            if st.button("H2", key=f"fmt_h2_{section_id}", use_container_width=True):
                st.session_state[content_key] = f"{st.session_state.get(content_key, '').rstrip()}\n## Subheading".strip()
                st.rerun()
        with toolbar_col_3:
            if st.button("Bullet List", key=f"fmt_bullets_{section_id}", use_container_width=True):
                st.session_state[content_key] = (
                    f"{st.session_state.get(content_key, '').rstrip()}\n- Point 1\n- Point 2\n- Point 3"
                ).strip()
                st.rerun()
        with toolbar_col_4:
            if st.button("Table", key=f"fmt_table_{section_id}", use_container_width=True):
                st.session_state[content_key] = (
                    f"{st.session_state.get(content_key, '').rstrip()}\n| Metric | Value |\n|---|---|\n| Example | 0 |"
                ).strip()
                st.rerun()

        toolbar_col_5, toolbar_col_6, toolbar_col_7, toolbar_col_8 = st.columns(4)
        with toolbar_col_5:
            if st.button("Bold", key=f"fmt_bold_{section_id}", use_container_width=True):
                st.session_state[content_key] = f"{st.session_state.get(content_key, '').rstrip()} **bold text**".strip()
                st.rerun()
        with toolbar_col_6:
            if st.button("Italic", key=f"fmt_italic_{section_id}", use_container_width=True):
                st.session_state[content_key] = f"{st.session_state.get(content_key, '').rstrip()} *italic text*".strip()
                st.rerun()
        with toolbar_col_7:
            if st.button("Quote", key=f"fmt_quote_{section_id}", use_container_width=True):
                st.session_state[content_key] = f"{st.session_state.get(content_key, '').rstrip()}\n> Supporting quote".strip()
                st.rerun()
        with toolbar_col_8:
            if st.button("Divider", key=f"fmt_divider_{section_id}", use_container_width=True):
                st.session_state[content_key] = f"{st.session_state.get(content_key, '').rstrip()}\n---".strip()
                st.rerun()

        st.text_area(
            "Topic content",
            height=320,
            key=content_key,
            help="Edit the full content for this topic. The preview tab updates after saving.",
        )
        content_value = st.session_state.get(content_key, "")
        with st.expander("Formatting quick guide", expanded=False):
            st.caption(
                "# Heading | ## Subheading | **bold** | *italic* | - bullets | 1. numbered | > quote | --- divider"
            )

        if st.button("Save Topic Changes", use_container_width=True, key=f"save_topic_{section_id}"):
            final_title = title_value.strip() or section["title"]
            _update_section(
                section_id,
                {
                    "title": final_title[:80],
                    "tag": tag_value,
                    "content": content_value,
                    "content_format": format_choice,
                },
            )
            st.success("Topic content saved.")
            st.rerun()

    with style_tab:
        current_style = {**_default_section_style(), **section.get("style", {})}
        font_size = st.slider(
            "Font size",
            min_value=12,
            max_value=30,
            value=int(current_style.get("font_size", 16)),
            key=f"style_font_size_{section_id}",
        )
        line_spacing = st.slider(
            "Line spacing",
            min_value=1.1,
            max_value=2.2,
            value=float(current_style.get("line_spacing", 1.5)),
            step=0.1,
            key=f"style_line_spacing_{section_id}",
        )
        alignment = st.selectbox(
            "Text alignment",
            options=["left", "center", "right"],
            index=["left", "center", "right"].index(current_style.get("alignment", "left")),
            key=f"style_alignment_{section_id}",
        )
        color_col_1, color_col_2, color_col_3 = st.columns(3)
        with color_col_1:
            text_color = st.color_picker("Text color", value=current_style.get("text_color", "#1e3442"), key=f"style_text_{section_id}")
        with color_col_2:
            bg_color = st.color_picker("Background", value=current_style.get("bg_color", "#ffffff"), key=f"style_bg_{section_id}")
        with color_col_3:
            accent_color = st.color_picker(
                "Accent color",
                value=current_style.get("accent_color", "#4E8DA3"),
                key=f"style_accent_{section_id}",
            )

        if st.button("Apply Design Style", use_container_width=True, key=f"save_style_{section_id}"):
            _update_section(
                section_id,
                {
                    "style": {
                        "font_size": font_size,
                        "line_spacing": line_spacing,
                        "alignment": alignment,
                        "text_color": text_color,
                        "bg_color": bg_color,
                        "accent_color": accent_color,
                    }
                },
            )
            st.success("Style updated.")
            st.rerun()

    with ai_tab:
        action = st.selectbox(
            "AI action",
            [
                "Rewrite professionally",
                "Expand with analysis",
                "Shorten for executives",
                "Convert to bullet points",
                "Apply RFS compliance tone",
            ],
            key=f"ai_action_{section_id}",
        )
        ai_note = st.text_input(
            "Additional instruction (optional)",
            key=f"ai_note_{section_id}",
            placeholder="Example: Keep only data-backed statements and remove fluff",
        )
        draft_content = st.session_state.get(content_key, section.get("content", ""))
        route_task = "insight_extraction" if action in {"Expand with analysis", "Apply RFS compliance tone"} else "summarization"
        route_map = llm_runtime_settings.get("routing", {})
        route_target = route_map.get(route_task, {})
        if route_target:
            st.caption(
                f"Routed model: {route_target.get('backend', '?')} / {route_target.get('model_name', '?')} ({route_task})"
            )
        if st.button("Apply AI Update", type="primary", use_container_width=True, key=f"run_ai_{section_id}"):
            if not str(draft_content).strip():
                st.warning("Add content in Topic Content before running AI assist.")
            else:
                prompt = (
                    f"Template: {template_choice}\n"
                    f"Template guidance: {_template_ai_hint(template_choice)}\n"
                    f"Section title: {section.get('title', 'Topic')}\n"
                    f"Action: {action}\n"
                    f"Additional instruction: {ai_note or 'None'}\n\n"
                    f"Original content:\n{draft_content}\n\n"
                    "Return only the revised section content."
                )
                system_prompt = (
                    "You are a senior report editor. Keep claims grounded, preserve factual numbers, and structure cleanly."
                )
                ai_text = ""
                used_fallback = False
                try:
                    llm.apply_runtime_settings(llm_runtime_settings)
                    ai_text = llm.generate(
                        prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        task_type=route_task,
                    )
                except Exception:
                    used_fallback = True
                    ai_text = _fallback_section_ai_edit(action, str(draft_content), section.get("title", "Topic"))

                if ai_text.strip():
                    _update_section(section_id, {"content": ai_text.strip()})
                    st.session_state[f"refresh_content_{section_id}"] = True
                    if used_fallback:
                        st.warning("LLM unavailable right now. Applied a local transformation instead.")
                    else:
                        st.success("AI update applied to this topic.")
                    st.rerun()
                else:
                    st.warning("No AI output returned for this topic.")

    with preview_tab:
        st.caption("Live styled preview for the selected topic.")
        _render_section_preview(section)


def _render_preview_panel(reader_title: str, parsed_docs: List[Dict[str, Any]]) -> None:
    metrics = _compute_metrics(parsed_docs)
    st.markdown(
        f"""
<div class="preview-card">
  <div class="preview-title">{reader_title}</div>
  <div class="preview-subtitle">Industry Growth Trends & Consumer Insights</div>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Market Growth", f"{metrics['avg_rfs_quality']:.1f}%")
    with m2:
        st.metric("Extraction Confidence", f"{metrics['avg_confidence'] * 100:.1f}%")
    with m3:
        st.metric("Statement Lines", f"{int(metrics['line_items'])}")
    with m4:
        st.metric("OCR Coverage", f"{metrics['ocr_coverage']:.1f}%")

    rows = _statement_rows(parsed_docs)
    top_rows = rows.sort_values("amount_numeric", ascending=False).head(6)

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        bar_chart = px.bar(
            top_rows,
            x="line_item",
            y="amount_numeric",
            title="Consumer Preference Trends",
            color_discrete_sequence=["#4E8DA3"],
        )
        bar_chart.update_layout(
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor="#f7fafc",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        st.plotly_chart(bar_chart, use_container_width=True)

    with chart_col_2:
        trend_df = rows.groupby("period", as_index=False)["amount_numeric"].sum().sort_values("period")
        if len(trend_df) < 2:
            trend_df = pd.DataFrame(
                {"period": ["2021", "2022", "2023", "2024"], "amount_numeric": [12, 18, 24, 32]}
            )
        line_chart = px.area(
            trend_df,
            x="period",
            y="amount_numeric",
            title="Annual Industry Growth",
            color_discrete_sequence=["#7EB7CB"],
        )
        line_chart.update_layout(
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor="#f7fafc",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        st.plotly_chart(line_chart, use_container_width=True)

    chart_col_3, chart_col_4 = st.columns(2)
    with chart_col_3:
        share_df = top_rows[["line_item", "amount_numeric"]].copy()
        if share_df.empty:
            share_df = pd.DataFrame(
                {"line_item": ["Company A", "Company B", "Company C", "Others"], "amount_numeric": [35, 28, 22, 15]}
            )
        share_chart = px.bar(
            share_df,
            x="amount_numeric",
            y="line_item",
            orientation="h",
            title="Market Share by Company",
            color_discrete_sequence=["#D75D47"],
        )
        share_chart.update_layout(
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor="#f7fafc",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        st.plotly_chart(share_chart, use_container_width=True)

    with chart_col_4:
        format_df = pd.DataFrame(
            [{"format": doc.get("format", "unknown").upper(), "count": 1} for doc in parsed_docs]
        )
        if format_df.empty:
            format_df = pd.DataFrame(
                [
                    {"format": "PDF", "count": 1},
                    {"format": "IMG", "count": 1},
                    {"format": "TXT", "count": 1},
                    {"format": "XLSX", "count": 1},
                ]
            )
        format_df = format_df.groupby("format", as_index=False)["count"].sum()
        donut_chart = go.Figure(
            data=[
                go.Pie(
                    labels=format_df["format"],
                    values=format_df["count"],
                    hole=0.62,
                    marker=dict(colors=["#4E8DA3", "#2D546A", "#D95D47", "#E6A83A", "#70A15B"]),
                )
            ]
        )
        donut_chart.update_layout(
            title="Consumer Behavior Insights",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=20),
        )
        st.plotly_chart(donut_chart, use_container_width=True)


def _render_rfs_table(parsed_docs: List[Dict[str, Any]]) -> None:
    if not parsed_docs:
        st.info("Run OCR/extraction to view RFS statement compliance results.")
        return

    rows = []
    for doc in parsed_docs:
        rfs = doc.get("rfs_statement", {})
        rows.append(
            {
                "File": doc.get("file_name", ""),
                "Format": doc.get("format", ""),
                "Source": doc.get("extraction_source", ""),
                "OCR Status": doc.get("ocr", {}).get("status", "n/a"),
                "Extraction Confidence": f"{float(doc.get('extraction_confidence', 0.0)) * 100:.1f}%",
                "RFS Status": rfs.get("status", "unknown"),
                "RFS Quality": f"{float(rfs.get('quality_score', 0.0)):.1f}%",
                "Line Items": int(rfs.get("summary", {}).get("line_items_detected", 0)),
                "Warnings": "; ".join(rfs.get("warnings", [])),
            }
        )

    compliance_df = pd.DataFrame(rows)
    st.dataframe(compliance_df, use_container_width=True, hide_index=True)


def _initialize_state() -> None:
    st.session_state.setdefault("reader_outline", [*DEFAULT_OUTLINE])
    if "reader_sections" not in st.session_state:
        st.session_state.reader_sections = _sections_from_outline(st.session_state.reader_outline, parsed_docs=[])
    st.session_state.setdefault("reader_selected_section", 0)
    st.session_state.setdefault("reader_docs", [])
    st.session_state.setdefault("reader_title", "Marketing Analysis Overview")
    st.session_state.setdefault("reader_subtitle", "Industry Growth Trends & Consumer Insights")
    st.session_state.setdefault("template_choice", "Report")
    st.session_state.setdefault("manual_text", "")
    st.session_state.setdefault("topic_text", "")
    st.session_state.setdefault("route_summary_backend", LLMBackend.OLLAMA.value)
    st.session_state.setdefault("route_summary_model", DEFAULT_MODEL_BY_BACKEND[LLMBackend.OLLAMA.value])
    st.session_state.setdefault("route_insight_backend", LLMBackend.OLLAMA.value)
    st.session_state.setdefault("route_insight_model", DEFAULT_MODEL_BY_BACKEND[LLMBackend.OLLAMA.value])
    st.session_state.setdefault("route_outline_backend", LLMBackend.OLLAMA.value)
    st.session_state.setdefault("route_outline_model", DEFAULT_MODEL_BY_BACKEND[LLMBackend.OLLAMA.value])
    st.session_state.setdefault("auto_route_by_demand", True)
    st.session_state.setdefault("routing_policy", "balanced")
    st.session_state.setdefault("routing_decision", {})
    st.session_state.setdefault("routing_decision_preview", {})


_inject_styles()
_initialize_state()

with st.sidebar:
    st.markdown("## Reader Configuration")
    llm_backend = st.selectbox(
        "LLM Backend",
        BACKEND_OPTIONS,
        index=0,
    )
    llm_model = st.text_input("Model", value=DEFAULT_MODEL_BY_BACKEND.get(llm_backend, "mistral:latest"))
    llm_base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL_BY_BACKEND.get(llm_backend, ""))
    llm_api_key = st.text_input("API key", type="password", help="Needed for OpenAI, Gemini, or Anthropic")

    with st.expander("Function-Based Model Routing", expanded=False):
        st.caption("Use different models/backends by function.")
        route_summary_backend = st.selectbox("Summary backend", BACKEND_OPTIONS, key="route_summary_backend")
        route_summary_model = st.text_input(
            "Summary model",
            value=DEFAULT_MODEL_BY_BACKEND.get(route_summary_backend, "mistral:latest"),
            key="route_summary_model",
        )
        route_insight_backend = st.selectbox("Insight backend", BACKEND_OPTIONS, key="route_insight_backend")
        route_insight_model = st.text_input(
            "Insight model",
            value=DEFAULT_MODEL_BY_BACKEND.get(route_insight_backend, "mistral:latest"),
            key="route_insight_model",
        )
        route_outline_backend = st.selectbox("Outline backend", BACKEND_OPTIONS, key="route_outline_backend")
        route_outline_model = st.text_input(
            "Outline model",
            value=DEFAULT_MODEL_BY_BACKEND.get(route_outline_backend, "mistral:latest"),
            key="route_outline_model",
        )
        use_shared_credentials_for_routes = st.checkbox("Use shared API key/base URL for routed models", value=True)
        auto_route_by_demand = st.toggle(
            "Auto-route by demand (cost/latency aware)",
            value=st.session_state.auto_route_by_demand,
            key="auto_route_by_demand",
        )
        routing_policy = st.selectbox(
            "Routing policy",
            ["cost_saver", "balanced", "quality_first"],
            index=["cost_saver", "balanced", "quality_first"].index(st.session_state.routing_policy),
            key="routing_policy",
            help=(
                "cost_saver: keep most steps on the cheaper model, escalate insights when demand is high.\n"
                "balanced: escalate insights at medium/high demand, outline at high demand.\n"
                "quality_first: always use dedicated insight/outline models."
            ),
        )

    if st.button("Check LLM Connection", use_container_width=True):
        manual_routing = _build_manual_routing_payload(
            route_summary_backend=route_summary_backend,
            route_summary_model=route_summary_model,
            route_insight_backend=route_insight_backend,
            route_insight_model=route_insight_model,
            route_outline_backend=route_outline_backend,
            route_outline_model=route_outline_model,
            use_shared_credentials_for_routes=use_shared_credentials_for_routes,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
        demand_profile = _estimate_demand_profile(
            parsed_docs=st.session_state.reader_docs,
            topic_text=st.session_state.topic_text,
            manual_text=st.session_state.manual_text,
        )
        route_payload = (
            _resolve_auto_routing(manual_routing, demand_profile, routing_policy)
            if auto_route_by_demand
            else manual_routing
        )

        llm.apply_runtime_settings(
            {
                "backend": llm_backend,
                "model_name": llm_model,
                "api_key": llm_api_key.strip(),
                "base_url": llm_base_url.strip(),
                "routing": route_payload,
                "timeout": 20,
            }
        )
        if llm.health_check(task_type="summarization"):
            selected = route_payload.get("summarization", {})
            st.success(
                f"LLM reachable ({selected.get('backend','?')}/{selected.get('model_name','?')})"
            )
            st.caption(
                f"Demand: {demand_profile['level']} ({demand_profile['score']}) | Policy: {routing_policy if auto_route_by_demand else 'manual'}"
            )
        else:
            st.error(f"LLM check failed: {llm.last_error or 'No response'}")

    ocr_language = st.text_input("OCR language", value="eng")
    enable_ocr = st.toggle("Dynamic OCR for PDF/Image", value=True)
    run_full_pipeline = st.toggle("Run full multi-agent pipeline", value=False)
    include_summary = st.checkbox("Include summary", value=True)
    include_charts = st.checkbox("Include charts", value=True)
    include_kpis = st.checkbox("Include KPI cards", value=True)

    preview_manual_routing = _build_manual_routing_payload(
        route_summary_backend=route_summary_backend,
        route_summary_model=route_summary_model,
        route_insight_backend=route_insight_backend,
        route_insight_model=route_insight_model,
        route_outline_backend=route_outline_backend,
        route_outline_model=route_outline_model,
        use_shared_credentials_for_routes=use_shared_credentials_for_routes,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )
    preview_demand_profile = _estimate_demand_profile(
        parsed_docs=st.session_state.reader_docs,
        topic_text=st.session_state.topic_text,
        manual_text=st.session_state.manual_text,
    )
    preview_resolved_routing = (
        _resolve_auto_routing(preview_manual_routing, preview_demand_profile, routing_policy)
        if auto_route_by_demand
        else preview_manual_routing
    )
    st.session_state.routing_decision_preview = {
        "auto_route_by_demand": auto_route_by_demand,
        "routing_policy": routing_policy if auto_route_by_demand else "manual",
        "demand_profile": preview_demand_profile,
        "routing_payload": preview_resolved_routing,
    }

    st.markdown("### Routing Preview")
    st.caption(
        f"Demand: {preview_demand_profile['level']} ({preview_demand_profile['score']}) | "
        f"Policy: {routing_policy if auto_route_by_demand else 'manual'}"
    )
    st.dataframe(
        _routing_payload_to_df(preview_resolved_routing),
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    """
<div class="header-bar">
  <div class="header-title">Marketing Analysis Overview</div>
  <div class="header-subtitle">RFS-ready reader workspace with OCR-aware extraction for PDF and image statements.</div>
</div>
""",
    unsafe_allow_html=True,
)

nav_col_1, nav_col_2, nav_col_3 = st.columns([1, 1, 6])
with nav_col_1:
    st.button("← Back", use_container_width=True)
with nav_col_2:
    st.button("👁 Preview", use_container_width=True)
with nav_col_3:
    export_payload = json.dumps(st.session_state.reader_docs, indent=2, default=str)
    st.download_button(
        "⬇ Export Extraction JSON",
        data=export_payload,
        file_name="reader_extraction.json",
        mime="application/json",
        use_container_width=False,
    )

st.markdown('<div class="hero-title">AI-Powered Visual Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-note">Choose your template format and generate statement-ready content in seconds.</div>', unsafe_allow_html=True)

if "pending_template_choice" in st.session_state:
    pending_template = st.session_state.pop("pending_template_choice")
    if pending_template in DOCUMENT_TEMPLATES:
        st.session_state.template_choice = pending_template
        st.session_state.template_choice_control = pending_template

template_choice = _segmented(
    "Template",
    DOCUMENT_TEMPLATES,
    st.session_state.template_choice,
    key="template_choice_control",
)
if template_choice:
    st.session_state.template_choice = template_choice

template_recommendation = _recommend_template(
    topic_text=st.session_state.topic_text,
    manual_text=st.session_state.manual_text,
    parsed_docs=st.session_state.reader_docs,
    selected_template=st.session_state.template_choice,
)
recommended_template = template_recommendation["template"]
st.caption(
    f"Recommended template: {recommended_template} | {_template_subtitle(recommended_template)}"
)
st.caption(template_recommendation["reason"])

template_col_1, template_col_2, template_col_3 = st.columns([1.6, 1.6, 3.8])
with template_col_1:
    if st.button(
        f"Use Recommended ({recommended_template})",
        use_container_width=True,
        disabled=recommended_template == st.session_state.template_choice,
    ):
        st.session_state.pending_template_choice = recommended_template
        st.rerun()
with template_col_2:
    if st.button("Apply Template to Workspace", use_container_width=True):
        baseline_text = st.session_state.topic_text or st.session_state.manual_text
        st.session_state.reader_outline = _derive_outline(
            baseline_text,
            st.session_state.reader_docs,
            st.session_state.template_choice,
        )
        st.session_state.reader_sections = _sections_from_outline(
            st.session_state.reader_outline,
            st.session_state.reader_docs,
            existing_sections=st.session_state.get("reader_sections", []),
        )
        st.session_state.reader_selected_section = 0
        st.session_state.reader_subtitle = _template_subtitle(st.session_state.template_choice)
        st.success(f"Applied {st.session_state.template_choice} template structure.")
        st.rerun()
with template_col_3:
    st.info(
        f"Template output profile: {_template_ai_hint(st.session_state.template_choice)}"
    )

st.markdown('<div class="input-shell">', unsafe_allow_html=True)
input_mode = _segmented(
    "Create From",
    ["Type Your Topic", "Paste Text / Upload File"],
    "Paste Text / Upload File",
    key="input_mode",
)

uploaded_files = []
topic_text = ""
manual_text = ""

if input_mode == "Type Your Topic":
    topic_text = st.text_input(
        "Topic",
        value=st.session_state.topic_text,
        placeholder="Example: 2024 annual summary for financial AI market",
    )
    st.session_state.topic_text = topic_text
else:
    manual_text = st.text_area(
        "Content",
        value=st.session_state.manual_text,
        height=140,
        placeholder="Paste your statement content or upload files below...",
    )
    st.session_state.manual_text = manual_text
    uploaded_files = st.file_uploader(
        "Upload document/image files",
        type=UPLOAD_FORMATS,
        accept_multiple_files=True,
        help="Supports CSV, TXT, PDF, DOCX, XLSX, and common image formats.",
    )

action_col_1, action_col_2, action_col_3 = st.columns([1.4, 1.2, 4])
with action_col_1:
    run_reader = st.button("Sign Up & Generate", type="primary", use_container_width=True)
with action_col_2:
    reset_reader = st.button("Reset Reader", use_container_width=True)
with action_col_3:
    st.caption("RFS compliance checks run automatically after extraction.")

st.markdown("</div>", unsafe_allow_html=True)

if reset_reader:
    st.session_state.reader_docs = []
    st.session_state.reader_outline = _template_base_outline(st.session_state.template_choice)
    st.session_state.reader_sections = _sections_from_outline(st.session_state.reader_outline, parsed_docs=[])
    st.session_state.reader_selected_section = 0
    st.session_state.reader_title = "Marketing Analysis Overview"
    st.session_state.reader_subtitle = _template_subtitle(st.session_state.template_choice)
    MEMORY.clear()

if run_reader:
    parser = FileParser()
    parsed_docs: List[Dict[str, Any]] = []

    for file in uploaded_files or []:
        saved_path = _save_uploaded_file(file)
        parsed = parser.parse(str(saved_path), enable_ocr=enable_ocr, ocr_language=ocr_language)
        parsed_docs.append(parsed)

    if manual_text.strip():
        parsed_docs.append(_manual_text_doc(parser, manual_text.strip()))

    if topic_text.strip() and not parsed_docs:
        topic_stub = _manual_text_doc(parser, topic_text.strip())
        topic_stub["file_name"] = "topic_prompt.txt"
        parsed_docs.append(topic_stub)

    if not parsed_docs:
        st.error("Provide a topic, text, or upload at least one file before generating.")
    else:
        previous_sections = st.session_state.get("reader_sections", [])
        st.session_state.reader_docs = parsed_docs
        st.session_state.reader_outline = _derive_outline(
            topic_text or manual_text,
            parsed_docs,
            st.session_state.template_choice,
        )
        st.session_state.reader_sections = _sections_from_outline(
            st.session_state.reader_outline,
            parsed_docs,
            existing_sections=previous_sections,
        )
        st.session_state.reader_selected_section = 0
        if topic_text.strip():
            st.session_state.reader_title = " ".join(topic_text.strip().split()[:6]).title()
        else:
            primary = parsed_docs[0].get("file_name", "Statement Analysis")
            st.session_state.reader_title = f"{primary} Overview"
        st.session_state.reader_subtitle = (
            f"{_template_subtitle(st.session_state.template_choice)} | "
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if run_full_pipeline and parsed_docs:
            first_path = Path("uploads") / parsed_docs[0]["file_name"]
            if first_path.exists():
                coordinator = CoordinatorAgent()
                manual_routing = _build_manual_routing_payload(
                    route_summary_backend=route_summary_backend,
                    route_summary_model=route_summary_model,
                    route_insight_backend=route_insight_backend,
                    route_insight_model=route_insight_model,
                    route_outline_backend=route_outline_backend,
                    route_outline_model=route_outline_model,
                    use_shared_credentials_for_routes=use_shared_credentials_for_routes,
                    llm_api_key=llm_api_key,
                    llm_base_url=llm_base_url,
                )
                demand_profile = _estimate_demand_profile(
                    parsed_docs=parsed_docs,
                    topic_text=topic_text,
                    manual_text=manual_text,
                )
                routing_payload = (
                    _resolve_auto_routing(manual_routing, demand_profile, routing_policy)
                    if auto_route_by_demand
                    else manual_routing
                )
                st.session_state.routing_decision = {
                    "auto_route_by_demand": auto_route_by_demand,
                    "routing_policy": routing_policy,
                    "demand_profile": demand_profile,
                    "routing_payload": routing_payload,
                }

                pipeline_result = coordinator.execute(
                    {
                        "task_id": f"reader_{datetime.now().timestamp()}",
                        "file_path": str(first_path),
                        "report_type": "financial",
                        "include_summary": include_summary,
                        "include_charts": include_charts,
                        "include_kpis": include_kpis,
                        "enable_ocr": enable_ocr,
                        "ocr_language": ocr_language,
                        "llm_backend": llm_backend,
                        "llm_model": llm_model,
                        "llm_settings": {
                            "backend": llm_backend,
                            "model_name": llm_model,
                            "api_key": llm_api_key.strip(),
                            "base_url": llm_base_url.strip(),
                            "routing": routing_payload,
                        },
                    }
                )
                if pipeline_result.success:
                    st.success("Reader extraction and multi-agent pipeline completed.")
                    decision = st.session_state.routing_decision
                    st.caption(
                        f"Auto routing: {'on' if decision.get('auto_route_by_demand') else 'off'} | "
                        f"policy: {decision.get('routing_policy','manual')} | "
                        f"demand: {decision.get('demand_profile',{}).get('level','n/a')} "
                        f"({decision.get('demand_profile',{}).get('score','n/a')})"
                    )
                else:
                    st.warning(f"Reader extraction completed, but pipeline returned: {pipeline_result.error}")
            else:
                st.warning("Extraction succeeded but source file for full pipeline was not found on disk.")
        else:
            st.success("Reader extraction completed.")

effective_routing = (
    st.session_state.get("routing_decision", {}).get("routing_payload")
    or st.session_state.get("routing_decision_preview", {}).get("routing_payload")
    or {}
)
llm_runtime_settings = {
    "backend": llm_backend,
    "model_name": llm_model,
    "api_key": llm_api_key.strip(),
    "base_url": llm_base_url.strip(),
    "routing": effective_routing,
}

workspace_tab, inspector_tab, advanced_tab = st.tabs(["Reader Workspace", "Extraction Inspector", "Advanced"])

with workspace_tab:
    left_col, right_col = st.columns([1.05, 1.65], gap="large")
    with left_col:
        _render_outline_panel()
    with right_col:
        _render_section_editor(
            st.session_state.reader_title,
            st.session_state.reader_docs,
            st.session_state.template_choice,
            llm_runtime_settings=llm_runtime_settings,
        )

with inspector_tab:
    st.markdown("### RFS Statement Compliance")
    _render_rfs_table(st.session_state.reader_docs)

    statement_df = _statement_rows(st.session_state.reader_docs)
    statement_view_columns = ["file_name", "line_item", "value", "period", "source", "confidence"]
    statement_df = statement_df.reindex(columns=statement_df.columns.union(statement_view_columns), fill_value="")
    st.markdown("### Extracted Statement Lines")
    st.dataframe(
        statement_df[statement_view_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Raw Extraction JSON")
    st.code(json.dumps(st.session_state.reader_docs, indent=2, default=str), language="json")

with advanced_tab:
    st.markdown("### OCR + Reader Notes")
    st.write(
        "- PDF flow: embedded text extraction first, then OCR fallback for scanned pages.\n"
        "- Image flow: direct OCR with confidence scoring.\n"
        "- RFS flow: normalized statement lines + quality checks + compliance status."
    )
    st.write(
        "If OCR status reports `failed` due missing engine, install both:\n"
        "`pip install pytesseract pypdfium2` and the `tesseract` system binary."
    )

    st.markdown("### LLM Routing Decision")
    routing_decision = st.session_state.get("routing_decision", {})
    if routing_decision:
        st.code(json.dumps(routing_decision, indent=2, default=str), language="json")
    else:
        preview_decision = st.session_state.get("routing_decision_preview", {})
        if preview_decision:
            st.code(json.dumps(preview_decision, indent=2, default=str), language="json")
            st.info("Preview shown above. Run full pipeline to persist the final routing decision.")
        else:
            st.info("Run the full multi-agent pipeline to capture auto-routing decisions.")

    st.markdown("### Recent Execution History")
    history = MEMORY.get_history()
    if history:
        history_df = pd.DataFrame(history[:10])
        for column in history_df.columns:
            history_df[column] = history_df[column].apply(
                lambda value: json.dumps(value, default=str)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No full pipeline executions yet.")

st.caption("RFS Reader Studio | Dynamic OCR + Statement Compliance")
