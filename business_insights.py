"""Business insights and HR recommendations for the burnout system.

Derives top risk drivers and protective factors from the model coefficients
and exposes a curated list of actionable HR recommendations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# At least 15 actionable, evidence-aligned HR recommendations.
HR_RECOMMENDATIONS: List[str] = [
    "Cap excessive weekly meetings and enforce meeting-free focus blocks.",
    "Launch subsidised exercise and fitness programmes to boost the Health Index.",
    "Promote work-life balance with clear after-hours communication norms.",
    "Introduce flexible and hybrid work policies to reduce commuting strain.",
    "Run sleep-awareness initiatives and discourage late-night email culture.",
    "Offer stress-management workshops and mindfulness sessions.",
    "Monitor weekly work hours and flag sustained overtime automatically.",
    "Rebalance project loads so no employee handles too many concurrent projects.",
    "Provide manager training to spot early burnout signals in their teams.",
    "Set healthy email-volume guidelines and encourage async communication.",
    "Guarantee a minimum number of remote days per month for recovery.",
    "Establish confidential mental-health support and EAP access.",
    "Use the burnout model quarterly to proactively identify at-risk employees.",
    "Reward sustainable productivity, not just raw hours worked.",
    "Encourage regular, fully-disconnected paid time off.",
    "Audit roles with low productivity efficiency for process bottlenecks.",
    "Create peer-support and team-building programmes to improve wellbeing.",
]


def split_factors(
    coefficients: Dict[str, float]
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Split coefficients into risk-increasing and risk-reducing factors.

    Args:
        coefficients: Mapping of feature -> standardized coefficient.

    Returns:
        (positive_factors, negative_factors), each sorted by magnitude.
    """
    positive = sorted(
        [(k, v) for k, v in coefficients.items() if v > 0],
        key=lambda kv: kv[1], reverse=True,
    )
    negative = sorted(
        [(k, v) for k, v in coefficients.items() if v < 0],
        key=lambda kv: kv[1],
    )
    return positive, negative


def build_insights(coefficients: Dict[str, float], top_n: int = 5) -> str:
    """Render a business insights summary string.

    Args:
        coefficients: Model coefficients.
        top_n: Number of top factors to surface per category.

    Returns:
        A formatted insights report.
    """
    positive, negative = split_factors(coefficients)
    lines: List[str] = ["BUSINESS INSIGHTS", "=" * 60, "Top burnout RISK drivers:"]
    for name, val in positive[:top_n]:
        lines.append(f"  + {name} (+{val:.2f})")
    lines.append("Top PROTECTIVE factors (reduce burnout):")
    for name, val in negative[:top_n]:
        lines.append(f"  - {name} ({val:.2f})")
    lines.append("")
    lines.append("Actionable HR recommendations:")
    for i, rec in enumerate(HR_RECOMMENDATIONS, 1):
        lines.append(f"  {i:>2}. {rec}")
    return "\n".join(lines)
