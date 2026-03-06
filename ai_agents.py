"""
ai_agents.py  --  Claude-powered agentic team for price intelligence.

Three specialised agents collaborate on a shared pricing dataset:

  Agent 1 (Validator)   - Scores match quality between our product and
                          each competitor listing (sanity-check layer).
  Agent 2 (Analyst)     - Generates market intelligence & key findings.
  Agent 3 (Strategist)  - Produces actionable pricing recommendations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from Claude's response."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


class PriceIntelligenceAgents:
    """Multi-agent AI system using Claude for competitive price intelligence."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250514"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    # ------------------------------------------------------------------
    # Agent 1: Match Validator
    # ------------------------------------------------------------------

    def validate_matches(
        self,
        product_name: str,
        brand: str,
        our_price: float,
        competitor_results: List[dict],
    ) -> dict:
        """Score how well each competitor listing matches our actual product.

        Returns::
            {
              "scores": [0.95, 0.6, ...],
              "flags":  ["exact match", "different capacity", ...],
              "best_match_index": 0,
              "summary": "..."
            }
        """
        if not competitor_results:
            return {"scores": [], "flags": [], "summary": "No results to validate."}

        listings = "\n".join(
            f"  {i + 1}. [{r['merchant']}] {r['title']}  --  ${r['price']:.2f}"
            for i, r in enumerate(competitor_results)
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=(
                "You are a product-matching expert for retail price comparison. "
                "Your job is to evaluate whether competitor listings are the SAME "
                "product as ours.  Return **only** valid JSON."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f'Our product: "{brand} {product_name}" priced at ${our_price:.2f}\n\n'
                        f"Competitor listings:\n{listings}\n\n"
                        "For each listing assign a match score (0.0-1.0):\n"
                        "  1.0 = exact same product\n"
                        "  0.7-0.9 = very likely same (minor spec diff)\n"
                        "  0.4-0.6 = similar but may differ in size/color/capacity\n"
                        "  0.0-0.3 = different product\n\n"
                        "Return ONLY this JSON:\n"
                        '{"scores": [...], "flags": ["...", ...], '
                        '"best_match_index": 0, "summary": "..."}'
                    ),
                }
            ],
        )
        result = _extract_json(resp.content[0].text)
        if not result.get("scores"):
            result = {
                "scores": [0.5] * len(competitor_results),
                "flags": ["auto"] * len(competitor_results),
                "summary": "Could not validate matches.",
            }
        return result

    # ------------------------------------------------------------------
    # Agent 2: Market Analyst
    # ------------------------------------------------------------------

    def analyze_market(self, pricing_data: List[dict]) -> dict:
        """Analyze the full pricing dataset and return strategic insights.

        Returns::
            {
              "executive_summary": "...",
              "key_findings": ["...", ...],
              "category_insights": { ... },
              "risk_alerts": ["...", ...],
              "opportunities": ["...", ...]
            }
        """
        # -- pre-aggregate so we don't blow the token budget ----------------
        by_cat: Dict[str, Dict[str, Any]] = {}
        by_brand: Dict[str, Dict[str, Any]] = {}
        competitive_n = overpriced_n = 0

        for p in pricing_data:
            cat = p.get("category", "Other")
            brand = p.get("brand", "Unknown")
            status = p.get("status", "unknown")
            gap = p.get("savings_opportunity", 0) or 0
            diff = p.get("pct_diff") or 0

            by_cat.setdefault(cat, {"n": 0, "comp": 0, "high": 0, "gap": 0})
            by_cat[cat]["n"] += 1
            if status == "competitive":
                by_cat[cat]["comp"] += 1
                competitive_n += 1
            elif status == "high":
                by_cat[cat]["high"] += 1
                overpriced_n += 1
            by_cat[cat]["gap"] += gap

            by_brand.setdefault(brand, {"n": 0, "sum_diff": 0})
            by_brand[brand]["n"] += 1
            by_brand[brand]["sum_diff"] += diff

        for b in by_brand.values():
            b["avg_diff"] = b["sum_diff"] / b["n"] if b["n"] else 0

        top_over = sorted(
            (p for p in pricing_data if (p.get("pct_diff") or 0) > 0),
            key=lambda x: x["pct_diff"],
            reverse=True,
        )[:10]

        cat_lines = "\n".join(
            f"  {c}: {d['n']} products | {d['comp']} competitive | "
            f"{d['high']} overpriced | gap ${d['gap']:,.0f}"
            for c, d in sorted(by_cat.items())
        )
        brand_lines = "\n".join(
            f"  {b}: {d['n']} products | avg diff {d['avg_diff']:+.1f}%"
            for b, d in sorted(
                by_brand.items(), key=lambda x: x[1]["avg_diff"], reverse=True
            )[:15]
        )
        over_lines = "\n".join(
            f"  {p['description']} (SKU {p['sku']}): ours ${p['our_price']:.2f} "
            f"vs mkt ${p.get('min_competitor', 0):.2f}  ({p['pct_diff']:+.1f}%)"
            for p in top_over
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=(
                "You are a senior pricing analyst. Provide crisp, data-driven "
                "insights in a professional tone. Return **only** valid JSON."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"COMPETITIVE PRICING ANALYSIS\n"
                        f"Total products: {len(pricing_data)}  |  "
                        f"Competitive: {competitive_n}  |  Overpriced: {overpriced_n}\n\n"
                        f"BY CATEGORY:\n{cat_lines}\n\n"
                        f"BY BRAND (top 15 by premium):\n{brand_lines}\n\n"
                        f"TOP 10 OVERPRICED:\n{over_lines}\n\n"
                        "Return ONLY this JSON:\n"
                        "{\n"
                        '  "executive_summary": "2-3 sentences",\n'
                        '  "key_findings": ["...", "...", "...", "...", "..."],\n'
                        '  "category_insights": {\n'
                        '    "strongest": ["category: rationale", ...],\n'
                        '    "weakest": ["category: rationale", ...]\n'
                        "  },\n"
                        '  "risk_alerts": ["...", "..."],\n'
                        '  "opportunities": ["...", "...", "..."]\n'
                        "}"
                    ),
                }
            ],
        )
        out = _extract_json(resp.content[0].text)
        if not out.get("executive_summary"):
            out["executive_summary"] = "Analysis unavailable."
        return out

    # ------------------------------------------------------------------
    # Agent 3: Pricing Strategist
    # ------------------------------------------------------------------

    def generate_strategy(
        self, pricing_data: List[dict], market_analysis: dict
    ) -> dict:
        """Produce specific, actionable pricing recommendations.

        Returns::
            {
              "immediate_actions": [ {product, sku, action, current, recommended, rationale}, ... ],
              "category_strategies": [ {category, strategy, impact}, ... ],
              "margin_opportunities": [ {product, current, market, recommended, upside_pct}, ... ],
              "overall_recommendation": "..."
            }
        """
        exec_sum = market_analysis.get("executive_summary", "N/A")
        findings = "\n".join(
            f"  - {f}" for f in market_analysis.get("key_findings", [])
        )

        overpriced = sorted(
            (p for p in pricing_data if (p.get("savings_opportunity") or 0) > 0),
            key=lambda x: x["savings_opportunity"],
            reverse=True,
        )[:12]
        underpriced = sorted(
            (p for p in pricing_data if (p.get("pct_diff") or 0) < -5),
            key=lambda x: x["pct_diff"],
        )[:10]

        over_txt = "\n".join(
            f"  {p['description']} ({p['sku']}): ours ${p['our_price']:.2f} "
            f"vs low ${p.get('min_competitor', 0):.2f} (gap ${p['savings_opportunity']:.2f})"
            for p in overpriced
        )
        under_txt = (
            "\n".join(
                f"  {p['description']} ({p['sku']}): ours ${p['our_price']:.2f} "
                f"vs mkt ${p.get('min_competitor', 0):.2f} ({p['pct_diff']:+.1f}%)"
                for p in underpriced
            )
            or "  None found"
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            system=(
                "You are a pricing strategy consultant advising a retail company. "
                "Provide concrete, actionable recommendations with specific dollar "
                "amounts. Return **only** valid JSON."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"EXECUTIVE SUMMARY:\n  {exec_sum}\n\n"
                        f"KEY FINDINGS:\n{findings}\n\n"
                        f"TOP OVERPRICED (reduce?):\n{over_txt}\n\n"
                        f"UNDERPRICED (raise?):\n{under_txt}\n\n"
                        "Return ONLY this JSON:\n"
                        "{\n"
                        '  "immediate_actions": [\n'
                        '    {"product": "...", "sku": "...", "action": "reduce|increase|maintain",\n'
                        '     "current_price": 0.00, "recommended_price": 0.00, "rationale": "..."}\n'
                        "  ],\n"
                        '  "category_strategies": [\n'
                        '    {"category": "...", "strategy": "...", "expected_impact": "..."}\n'
                        "  ],\n"
                        '  "margin_opportunities": [\n'
                        '    {"product": "...", "current_price": 0.00, "market_price": 0.00,\n'
                        '     "recommended_price": 0.00, "upside_pct": 0.0}\n'
                        "  ],\n"
                        '  "overall_recommendation": "2-3 sentence executive recommendation"\n'
                        "}"
                    ),
                }
            ],
        )
        out = _extract_json(resp.content[0].text)
        if not out.get("overall_recommendation"):
            out["overall_recommendation"] = "Strategy generation incomplete."
        return out
