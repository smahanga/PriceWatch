"""
scraper.py  --  Multi-source competitor price scraping engine.

Primary strategy: DuckDuckGo text search with intelligent price extraction
from result snippets, filtering by retailer domains and price reasonableness.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known retailer domain -> display name
# ---------------------------------------------------------------------------
_MERCHANTS = {
    "amazon.com": "Amazon",
    "walmart.com": "Walmart",
    "bestbuy.com": "Best Buy",
    "target.com": "Target",
    "ebay.com": "eBay",
    "newegg.com": "Newegg",
    "bhphotovideo.com": "B&H Photo",
    "adorama.com": "Adorama",
    "costco.com": "Costco",
    "homedepot.com": "Home Depot",
    "lowes.com": "Lowe's",
    "macys.com": "Macy's",
    "nordstrom.com": "Nordstrom",
    "sephora.com": "Sephora",
    "ulta.com": "Ulta",
    "chewy.com": "Chewy",
    "petco.com": "Petco",
    "dickssportinggoods.com": "Dick's Sporting Goods",
    "rei.com": "REI",
    "nike.com": "Nike",
    "adidas.com": "Adidas",
    "samsung.com": "Samsung",
    "apple.com": "Apple",
    "dell.com": "Dell",
    "hp.com": "HP",
    "lenovo.com": "Lenovo",
    "lego.com": "LEGO",
    "kohls.com": "Kohl's",
    "wayfair.com": "Wayfair",
    "overstock.com": "Overstock",
    "zappos.com": "Zappos",
    "gamestop.com": "GameStop",
    "bjs.com": "BJ's",
    "samsclub.com": "Sam's Club",
    "backmarket.com": "Back Market",
    "att.com": "AT&T",
    "verizon.com": "Verizon",
    "cspire.com": "C Spire",
    "tmobile.com": "T-Mobile",
    "microcenter.com": "Micro Center",
    "staples.com": "Staples",
    "officedepot.com": "Office Depot",
    "williams-sonoma.com": "Williams Sonoma",
    "crateandbarrel.com": "Crate & Barrel",
    "bedbathandbeyond.com": "Bed Bath & Beyond",
    "petsmart.com": "PetSmart",
    "lululemon.com": "Lululemon",
    "gap.com": "Gap",
    "oldnavy.com": "Old Navy",
    "jcrew.com": "J.Crew",
    "anthropologie.com": "Anthropologie",
    "underarmour.com": "Under Armour",
}

# Sites we skip (aggregators / review / news — not actual retailers)
_SKIP_DOMAINS = {
    # review / news / tech blogs
    "gsmarena.com", "tomsguide.com", "techradar.com", "pcmag.com",
    "cnet.com", "theverge.com", "androidauthority.com", "9to5mac.com",
    "phonearena.com", "wired.com", "engadget.com", "zdnet.com",
    "howtogeek.com", "mensjournal.com", "laptopmag.com", "gotechtor.com",
    "macrumors.com", "tomshardware.com", "rtings.com", "soundguys.com",
    # deal aggregator blogs / coupon sites
    "bradsdeals.com", "dannydealguru.com", "frugalbuzz.com",
    "dealnews.com", "slickdeals.net", "hip2save.com", "wirecutter.com",
    "freestufffinder.com", "buyvia.com", "gizmochina.com",
    "pocketnow.com", "notebookcheck.net", "xda-developers.com",
    "retailmenot.com", "coupon.com", "honey.com", "rakuten.com",
    # social / search / general
    "reddit.com", "youtube.com", "wikipedia.org", "quora.com",
    "twitter.com", "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com", "klarna.com", "google.com", "bing.com",
    "duckduckgo.com",
}


def _merchant_from_url(url: str) -> Optional[str]:
    """Map URL to a friendly merchant name.  Return None for non-retailers."""
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return None

    # Skip aggregator / review sites
    for skip in _SKIP_DOMAINS:
        if skip in host:
            return None

    for domain, name in _MERCHANTS.items():
        if domain in host:
            return name

    # For unknown domains, derive a name from the hostname
    parts = host.split(".")
    name = parts[0].capitalize() if parts else None
    return name


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CompetitorPrice:
    merchant: str
    price: float
    title: str
    url: str
    currency: str = "USD"
    match_score: Optional[float] = None  # filled by AI agent


@dataclass
class ProductComparison:
    sku: str
    description: str
    brand: str
    category: str
    our_price: float
    competitor_prices: List[CompetitorPrice] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def min_price(self) -> Optional[float]:
        return min((c.price for c in self.competitor_prices), default=None)

    @property
    def max_price(self) -> Optional[float]:
        return max((c.price for c in self.competitor_prices), default=None)

    @property
    def avg_price(self) -> Optional[float]:
        if not self.competitor_prices:
            return None
        return sum(c.price for c in self.competitor_prices) / len(
            self.competitor_prices
        )

    @property
    def price_status(self) -> str:
        if not self.competitor_prices:
            return "unknown"
        ratio = (self.our_price - self.min_price) / self.min_price * 100
        if ratio <= 0:
            return "competitive"
        if ratio <= 15:
            return "slightly_high"
        return "high"

    @property
    def savings_opportunity(self) -> float:
        if self.min_price is None:
            return 0.0
        return max(0.0, self.our_price - self.min_price)

    @property
    def pct_diff(self) -> Optional[float]:
        if self.min_price is None or self.min_price == 0:
            return None
        return (self.our_price - self.min_price) / self.min_price * 100


# ---------------------------------------------------------------------------
# Price Scraper
# ---------------------------------------------------------------------------

class PriceScraper:
    """Fetch competitor prices via DuckDuckGo text search + price extraction."""

    def __init__(self, delay: float = 2.5, max_results: int = 6):
        self.delay = delay
        self.max_results = max_results

    # -- price helpers ------------------------------------------------------

    @staticmethod
    def parse_price(raw: str) -> Optional[float]:
        """Robustly convert a price string like '$1,299.99' to a float."""
        if not raw:
            return None
        text = str(raw).strip()
        m = re.search(
            r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:\.\d{1,2})?)", text
        )
        if not m:
            return None
        s = m.group(1)
        if "," in s and "." in s:
            s = s.replace(",", "")
        elif "," in s:
            after = s.split(",")[1]
            s = s.replace(",", ".") if len(after) <= 2 else s.replace(",", "")
        try:
            v = float(s)
            return v if v > 0 else None
        except ValueError:
            return None

    @staticmethod
    def _extract_dollar_prices(text: str) -> List[float]:
        """Find all $X,XXX.XX patterns in text.  Returns deduplicated list."""
        matches = re.findall(r"\$[\d,]+(?:\.\d{1,2})?", text)
        prices: List[float] = []
        for m in matches:
            cleaned = m.replace("$", "").replace(",", "")
            try:
                v = float(cleaned)
                if v >= 2.0:  # filter out $0, $1 noise
                    prices.append(v)
            except ValueError:
                pass
        return prices

    # -- DDG text search with price extraction ------------------------------

    def _ddg_search(self, query: str, our_price: float) -> List[CompetitorPrice]:
        """Search DuckDuckGo, extract prices from result snippets.

        We only keep results from recognisable retailer domains and filter
        prices to a reasonable range around our own price.
        """
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                logger.warning("Neither ddgs nor duckduckgo_search installed.")
                return []

        try:
            out: List[CompetitorPrice] = []
            seen: set[str] = set()

            with DDGS() as ddgs:
                items = list(ddgs.text(f"{query} buy price", max_results=15))

            lo = our_price * 0.30
            hi = our_price * 2.5

            for item in items:
                href = item.get("href", "")
                merchant = _merchant_from_url(href)
                if merchant is None:
                    continue
                m_lower = merchant.lower()
                if m_lower in seen:
                    continue

                blob = f"{item.get('title', '')} {item.get('body', '')}"
                all_prices = self._extract_dollar_prices(blob)

                # Keep only prices in a reasonable range
                reasonable = [p for p in all_prices if lo <= p <= hi]
                if not reasonable:
                    continue

                # Pick the price closest to ours (most likely the actual product)
                best = min(reasonable, key=lambda p: abs(p - our_price))
                out.append(
                    CompetitorPrice(
                        merchant=merchant,
                        price=best,
                        title=item.get("title", "")[:120],
                        url=href,
                    )
                )
                seen.add(m_lower)

                if len(out) >= self.max_results:
                    break

            return out
        except Exception as exc:
            logger.warning("DDG text search failed: %s", exc)
            return []

    # -- public API ---------------------------------------------------------

    def get_competitor_prices(
        self,
        sku: str,
        description: str,
        brand: str,
        category: str,
        our_price: float,
    ) -> ProductComparison:
        comp = ProductComparison(
            sku=sku,
            description=description,
            brand=brand,
            category=category,
            our_price=our_price,
        )
        try:
            # Strategy 1: brand + product name
            prices = self._ddg_search(f"{brand} {description}", our_price)

            # Strategy 2: include SKU
            if len(prices) < 2:
                more = self._ddg_search(f"{sku} {description}", our_price)
                seen = {p.merchant.lower() for p in prices}
                for p in more:
                    if p.merchant.lower() not in seen:
                        prices.append(p)
                        seen.add(p.merchant.lower())

            comp.competitor_prices = prices[: self.max_results]
            if not prices:
                comp.error = "No competitor listings found."
        except Exception as exc:
            comp.error = str(exc)
        finally:
            time.sleep(self.delay + random.uniform(0.2, 0.7))
        return comp
