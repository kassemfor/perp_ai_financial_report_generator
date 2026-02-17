"""
Structured financial statement standards knowledge for IFRS-aware analysis.
"""

from typing import Any, Dict, List, Tuple

FIVE_PRIMARY_STATEMENTS: List[str] = [
    "Statement of financial position",
    "Statement of profit or loss and other comprehensive income",
    "Statement of cash flows",
    "Statement of changes in equity",
    "Notes to the financial statements",
]

IFRS_STANDARDS_MAP: List[Dict[str, str]] = [
    {
        "standard": "IAS 1 / IFRS 18",
        "purpose": "Presentation and classification of the primary financial statements and subtotals.",
        "line_items": "Assets, liabilities, equity, revenue, expenses, operating profit, OCI, notes.",
    },
    {
        "standard": "IAS 7",
        "purpose": "Classification of cash flows into operating, investing, and financing.",
        "line_items": "Cash flow from operations, investing cash flow, financing cash flow.",
    },
    {
        "standard": "IFRS 15",
        "purpose": "Revenue recognition using the five-step model.",
        "line_items": "Revenue, contract assets, deferred revenue, performance obligations.",
    },
    {
        "standard": "IFRS 16",
        "purpose": "Lease accounting with right-of-use assets and lease liabilities.",
        "line_items": "Right-of-use assets, lease liabilities, depreciation, interest expense.",
    },
    {
        "standard": "IFRS 9",
        "purpose": "Classification and measurement of financial instruments and expected credit losses.",
        "line_items": "Financial assets/liabilities, FVOCI/FVTPL movements, ECL provisions.",
    },
    {
        "standard": "IAS 12",
        "purpose": "Recognition and presentation of current and deferred taxes.",
        "line_items": "Income tax expense, current tax, deferred tax assets/liabilities.",
    },
    {
        "standard": "IFRS 19",
        "purpose": "Reduced disclosures for eligible subsidiaries without public accountability.",
        "line_items": "Disclosure package for eligible subsidiaries (effective 2027).",
    },
]

VALIDATION_CHECKLIST: List[Dict[str, str]] = [
    {
        "check": "Structural completeness",
        "detail": "Confirm all five primary statements are represented in extracted data and narrative.",
    },
    {
        "check": "Comparative period",
        "detail": "At least one comparative period should be present for key metrics.",
    },
    {
        "check": "IFRS 15 revenue quality",
        "detail": "Revenue timing and deferred revenue treatment should be explicit when contract language appears.",
    },
    {
        "check": "IFRS 16 lease recognition",
        "detail": "Lease contracts should map to right-of-use assets and lease liabilities where relevant.",
    },
    {
        "check": "IFRS 9 financial instruments",
        "detail": "Financial instruments and expected credit losses should be tagged and separated.",
    },
    {
        "check": "IAS 12 tax split",
        "detail": "Tax lines should distinguish current vs deferred tax where possible.",
    },
]

JURISDICTION_GAAP_PROFILES: Dict[str, Dict[str, Any]] = {
    "IFRS_GLOBAL": {
        "label": "IFRS (Global Baseline)",
        "framework": "ifrs",
        "notes": [
            "Principles-based framework with broad use across jurisdictions.",
            "Allows impairment reversals in many areas except goodwill.",
            "Supports fair-value remeasurement in more cases than US GAAP.",
        ],
        "priority_standards": ["IAS 1 / IFRS 18", "IAS 7", "IFRS 15", "IFRS 16", "IFRS 9", "IAS 12"],
    },
    "US_GAAP": {
        "label": "US GAAP",
        "framework": "local_gaap",
        "notes": [
            "Rules-based with different lease and impairment treatment vs IFRS.",
            "LIFO inventory method is allowed under US GAAP but not IFRS.",
            "Impairment reversals are generally prohibited.",
        ],
        "priority_standards": ["ASC equivalents with IFRS reconciliation focus"],
    },
    "CANADA_ASPE": {
        "label": "Canada ASPE",
        "framework": "local_gaap",
        "notes": [
            "Simpler disclosure package for private entities vs IFRS.",
            "Revenue recognition and measurement may differ from IFRS 15 rigor.",
            "Historical cost orientation is stronger than IFRS fair-value usage.",
        ],
        "priority_standards": ["ASPE with IFRS conversion adjustments"],
    },
    "EU_IFRS_ADOPTED": {
        "label": "EU (IFRS as adopted)",
        "framework": "ifrs_with_overlays",
        "notes": [
            "IFRS adopted with specific carve-outs and options in limited areas.",
            "Portfolio hedge accounting carve-out under IAS 39 may apply.",
            "Insurer-related options can affect IFRS 9 / IFRS 17 adoption timing.",
        ],
        "priority_standards": ["IAS 39 carve-out checks", "IFRS 9 transition options", "IFRS 17 cohort options"],
    },
    "GCC_IFRS_LOCAL": {
        "label": "GCC (IFRS + local GAAP options)",
        "framework": "mixed",
        "notes": [
            "IFRS commonly used, but local regulator-specific requirements still apply.",
            "Some entities can use local GAAP or IFRS for SMEs by regime.",
            "Entity-level framework selection should be explicit and auditable.",
        ],
        "priority_standards": ["IFRS baseline with local regulator overlays"],
    },
}

RFS_CODE_LABELS: Dict[str, str] = {
    "REV": "Revenue",
    "COGS": "Cost of Goods Sold",
    "GROSS_PROFIT": "Gross Profit",
    "OPERATING_INCOME": "Operating Income",
    "NET_INCOME": "Net Income",
    "EXPENSE": "Expenses",
    "ASSETS": "Assets",
    "LIABILITIES": "Liabilities",
    "EQUITY": "Equity",
    "CASH_FLOW": "Cash Flow",
    "EBITDA": "EBITDA",
    "TAX": "Income Tax",
    "CURRENT_TAX": "Current Tax",
    "DEFERRED_TAX": "Deferred Tax",
    "ROU_ASSET": "Right-of-Use Asset",
    "LEASE_LIABILITY": "Lease Liability",
    "DEFERRED_REVENUE": "Deferred Revenue",
    "OCI": "Other Comprehensive Income",
    "FINANCIAL_INSTRUMENT": "Financial Instrument",
    "ECL_PROVISION": "Expected Credit Loss Provision",
    "OTHER": "Other",
}

RFS_CODE_PATTERNS: List[Tuple[str, str]] = [
    (r"deferred revenue|contract liability|unearned revenue", "DEFERRED_REVENUE"),
    (r"revenue|sales|turnover|contract revenue", "REV"),
    (r"cost of goods|cogs|cost of sales", "COGS"),
    (r"gross profit|gross margin", "GROSS_PROFIT"),
    (r"operating income|operating profit|ebit(?!da)", "OPERATING_INCOME"),
    (r"net income|net profit|profit after tax|earnings", "NET_INCOME"),
    (r"ebitda", "EBITDA"),
    (r"right[- ]of[- ]use|rou asset", "ROU_ASSET"),
    (r"lease liabilit|lease obligation", "LEASE_LIABILITY"),
    (r"asset", "ASSETS"),
    (r"liabilit", "LIABILITIES"),
    (r"equity|shareholder", "EQUITY"),
    (r"cash flow|cashflow|operating cash|investing cash|financing cash", "CASH_FLOW"),
    (r"expected credit loss|ecl|impairment allowance", "ECL_PROVISION"),
    (r"financial instrument|fvoci|fvtpl|amortized cost|amortised cost", "FINANCIAL_INSTRUMENT"),
    (r"other comprehensive income|oci", "OCI"),
    (r"deferred tax", "DEFERRED_TAX"),
    (r"current tax", "CURRENT_TAX"),
    (r"tax", "TAX"),
    (r"expense|opex|operating expense", "EXPENSE"),
]

RFS_CODE_TO_STANDARD: Dict[str, str] = {
    "REV": "IFRS 15",
    "DEFERRED_REVENUE": "IFRS 15",
    "ROU_ASSET": "IFRS 16",
    "LEASE_LIABILITY": "IFRS 16",
    "FINANCIAL_INSTRUMENT": "IFRS 9",
    "ECL_PROVISION": "IFRS 9",
    "OCI": "IAS 1 / IFRS 18",
    "CASH_FLOW": "IAS 7",
    "CURRENT_TAX": "IAS 12",
    "DEFERRED_TAX": "IAS 12",
    "TAX": "IAS 12",
    "ASSETS": "IAS 1 / IFRS 18",
    "LIABILITIES": "IAS 1 / IFRS 18",
    "EQUITY": "IAS 1 / IFRS 18",
    "COGS": "IAS 1 / IFRS 18",
    "GROSS_PROFIT": "IAS 1 / IFRS 18",
    "OPERATING_INCOME": "IAS 1 / IFRS 18",
    "NET_INCOME": "IAS 1 / IFRS 18",
    "EXPENSE": "IAS 1 / IFRS 18",
    "EBITDA": "IAS 1 / IFRS 18",
}

STANDARD_FINANCE_FORMULAS: List[Dict[str, str]] = [
    {
        "formula": "Present Value (PV)",
        "expression": "PV = FV / (1 + r)^n",
        "meaning": "Current worth of future cash flows discounted at required return r.",
        "need": "Valuation, investment appraisal, and discounting future obligations.",
    },
    {
        "formula": "Future Value (FV)",
        "expression": "FV = PV * (1 + r)^n",
        "meaning": "Value of a current amount after compounding.",
        "need": "Savings growth, forecasting, and capital planning.",
    },
    {
        "formula": "NPV",
        "expression": "NPV = Sum(CF_t / (1 + r)^t) - Initial Investment",
        "meaning": "Value created by a project after discounting all cash flows.",
        "need": "Capital budgeting accept/reject decisions.",
    },
    {
        "formula": "IRR",
        "expression": "Find r where NPV = 0",
        "meaning": "Internal rate of return implied by a project's cash flows.",
        "need": "Compare project return against hurdle rate or WACC.",
    },
    {
        "formula": "WACC",
        "expression": "WACC = (E/V)*Re + (D/V)*Rd*(1 - Tax)",
        "meaning": "Blended cost of financing from debt and equity.",
        "need": "Discount rate baseline for DCF and strategic investments.",
    },
    {
        "formula": "Gross Margin",
        "expression": "(Revenue - COGS) / Revenue",
        "meaning": "Share of revenue retained after direct cost of sales.",
        "need": "Unit economics and pricing performance tracking.",
    },
    {
        "formula": "Net Margin",
        "expression": "Net Income / Revenue",
        "meaning": "Bottom-line profitability relative to revenue.",
        "need": "Profit resilience benchmarking and operating discipline.",
    },
    {
        "formula": "Debt-to-Equity",
        "expression": "Liabilities / Equity",
        "meaning": "Leverage relative to shareholder capital.",
        "need": "Solvency risk, capital structure, covenant monitoring.",
    },
    {
        "formula": "ROE",
        "expression": "Net Income / Equity",
        "meaning": "Return generated from shareholders' invested capital.",
        "need": "Value creation assessment for owners.",
    },
    {
        "formula": "ROA",
        "expression": "Net Income / Assets",
        "meaning": "Efficiency of total assets in producing earnings.",
        "need": "Asset productivity and investment allocation decisions.",
    },
    {
        "formula": "Current Ratio",
        "expression": "Current Assets / Current Liabilities",
        "meaning": "Liquidity coverage of short-term obligations.",
        "need": "Working-capital and short-term solvency management.",
    },
    {
        "formula": "Quick Ratio",
        "expression": "(Current Assets - Inventory) / Current Liabilities",
        "meaning": "Acid-test liquidity excluding inventory dependence.",
        "need": "Stress-tested liquidity monitoring.",
    },
    {
        "formula": "Inventory Turnover",
        "expression": "COGS / Average Inventory",
        "meaning": "Inventory velocity through the income cycle.",
        "need": "Supply-chain efficiency and cash cycle management.",
    },
    {
        "formula": "Days Sales Outstanding (DSO)",
        "expression": "(Average Accounts Receivable / Revenue) * 365",
        "meaning": "Average number of days to collect receivables.",
        "need": "Collection performance and cash conversion optimization.",
    },
]

INDUSTRY_KPI_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "Cross-Industry": [
        {"kpi": "Gross Profit Margin", "formula": "(Revenue - COGS) / Revenue"},
        {"kpi": "EBITDA Margin", "formula": "EBITDA / Revenue"},
        {"kpi": "Net Profit Margin", "formula": "Net Income / Revenue"},
        {"kpi": "Return on Equity", "formula": "Net Income / Equity"},
        {"kpi": "Current Ratio", "formula": "Current Assets / Current Liabilities"},
        {"kpi": "Quick Ratio", "formula": "(Current Assets - Inventory) / Current Liabilities"},
        {"kpi": "Cash Flow from Operations", "formula": "Operating cash inflows - outflows"},
        {"kpi": "Days Sales Outstanding", "formula": "(Average AR / Revenue) * 365"},
        {"kpi": "Days Payables Outstanding", "formula": "(Average AP / COGS) * 365"},
        {"kpi": "Inventory Turnover", "formula": "COGS / Average Inventory"},
    ],
    "Manufacturing": [
        {"kpi": "Capacity Utilization", "formula": "Actual Output / Maximum Output"},
        {"kpi": "Yield", "formula": "Good Units / Total Units"},
        {"kpi": "Scrap Rate", "formula": "Scrap Units / Total Units"},
        {"kpi": "Cycle Time", "formula": "Total Production Time / Units Produced"},
        {"kpi": "Overall Equipment Effectiveness", "formula": "Availability * Performance * Quality"},
    ],
    "Retail and E-Commerce": [
        {"kpi": "Same-Store Sales Growth", "formula": "(Current Comparable Sales - Prior) / Prior"},
        {"kpi": "Average Basket Size", "formula": "Revenue / Number of Transactions"},
        {"kpi": "Gross Merchandise Volume", "formula": "Total Value of Goods Sold"},
        {"kpi": "Customer Acquisition Cost", "formula": "Sales and Marketing Spend / New Customers"},
        {"kpi": "Churn Rate", "formula": "Lost Customers / Starting Customers"},
    ],
    "SaaS and Technology": [
        {"kpi": "Annual Recurring Revenue", "formula": "Monthly Recurring Revenue * 12"},
        {"kpi": "Monthly Recurring Revenue", "formula": "Sum of active monthly subscription revenue"},
        {"kpi": "Customer Lifetime Value", "formula": "ARPU * Gross Margin % / Churn Rate"},
        {"kpi": "Net Revenue Retention", "formula": "(Starting ARR + Expansion - Contraction - Churn) / Starting ARR"},
        {"kpi": "Average Revenue Per User", "formula": "Revenue / Active Users"},
    ],
    "Banking and Financial Services": [
        {"kpi": "Net Interest Margin", "formula": "Net Interest Income / Average Earning Assets"},
        {"kpi": "Cost-to-Income Ratio", "formula": "Operating Costs / Operating Income"},
        {"kpi": "Loan-to-Deposit Ratio", "formula": "Total Loans / Total Deposits"},
        {"kpi": "Non-Performing Loan Ratio", "formula": "Non-Performing Loans / Total Loans"},
        {"kpi": "Capital Adequacy Ratio", "formula": "Regulatory Capital / Risk-Weighted Assets"},
    ],
    "Healthcare": [
        {"kpi": "Patient Revenue per Day", "formula": "Patient Revenue / Patient Days"},
        {"kpi": "Bed Occupancy Rate", "formula": "Occupied Beds / Available Beds"},
        {"kpi": "Cost per Procedure", "formula": "Total Procedure Cost / Number of Procedures"},
    ],
    "Energy": [
        {"kpi": "Production Volume", "formula": "Total Produced Units (barrels, MWh, etc.)"},
        {"kpi": "Reserve Replacement Ratio", "formula": "New Reserves Added / Production"},
        {"kpi": "Lifting Cost per Barrel", "formula": "Operating Cost / Barrels Produced"},
    ],
}


def jurisdiction_options() -> List[str]:
    """List supported jurisdiction profile keys."""
    return list(JURISDICTION_GAAP_PROFILES.keys())


def jurisdiction_label(jurisdiction_key: str) -> str:
    """Return UI-friendly jurisdiction label."""
    profile = JURISDICTION_GAAP_PROFILES.get(jurisdiction_key, {})
    return profile.get("label", jurisdiction_key)


def jurisdiction_notes(jurisdiction_key: str) -> List[str]:
    """Return notes for selected jurisdiction profile."""
    profile = JURISDICTION_GAAP_PROFILES.get(jurisdiction_key, {})
    notes = profile.get("notes", [])
    return [str(note) for note in notes][:8]


def flatten_kpi_library(max_per_industry: int = 6) -> List[Dict[str, str]]:
    """Flatten KPI libraries into table-friendly rows."""
    rows: List[Dict[str, str]] = []
    for industry, entries in INDUSTRY_KPI_LIBRARY.items():
        for entry in entries[:max_per_industry]:
            rows.append(
                {
                    "Industry": industry,
                    "KPI": str(entry.get("kpi", "")),
                    "Formula": str(entry.get("formula", "")),
                }
            )
    return rows


def standards_context_text(max_chars: int = 12000) -> str:
    """Build a compact text block used in AI prompts and contextual summaries."""
    lines: List[str] = []
    lines.append("IFRS structural baseline:")
    lines.append("Primary statements required: " + ", ".join(FIVE_PRIMARY_STATEMENTS))
    lines.append("")
    lines.append("IFRS standards map:")
    for entry in IFRS_STANDARDS_MAP:
        lines.append(
            f"- {entry['standard']}: {entry['purpose']} Affected: {entry['line_items']}"
        )
    lines.append("")
    lines.append("Validation checklist:")
    for item in VALIDATION_CHECKLIST:
        lines.append(f"- {item['check']}: {item['detail']}")
    lines.append("")
    lines.append("Jurisdiction overlays:")
    for key, profile in JURISDICTION_GAAP_PROFILES.items():
        lines.append(f"- {profile['label']} ({key}): {'; '.join(profile.get('notes', [])[:2])}")

    return "\n".join(lines)[:max_chars]


def standards_bundle(max_chars: int = 12000) -> Dict[str, Any]:
    """Return reusable knowledge bundle for parser, UI, and prompting."""
    return {
        "five_primary_statements": FIVE_PRIMARY_STATEMENTS,
        "ifrs_standards_map": IFRS_STANDARDS_MAP,
        "validation_checklist": VALIDATION_CHECKLIST,
        "jurisdictions": JURISDICTION_GAAP_PROFILES,
        "rfs_code_labels": RFS_CODE_LABELS,
        "rfs_code_patterns": RFS_CODE_PATTERNS,
        "rfs_code_to_standard": RFS_CODE_TO_STANDARD,
        "standard_formulas": STANDARD_FINANCE_FORMULAS,
        "industry_kpi_rows": flatten_kpi_library(),
        "context_text": standards_context_text(max_chars=max_chars),
    }
