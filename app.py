"""
PriceWatch Pro  --  Competitor Price Intelligence Platform
==========================================================
Upload your product catalogue, scrape competitor prices from the web,
and let Claude AI agents analyse market position & recommend pricing strategy.

    streamlit run app.py
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scraper import CompetitorPrice, PriceScraper, ProductComparison

# ======================================================================
# PAGE CONFIG
# ======================================================================

st.set_page_config(
    page_title="PriceWatch Pro",
    page_icon="\U0001F4B0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CSS
# ======================================================================

st.markdown(
    """
<style>
/* --- base ----------------------------------------------------------- */
[data-testid="stAppViewContainer"]{background:linear-gradient(160deg,#f0f4ff 0%,#faf5ff 100%)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e293b,#0f172a)!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
[data-testid="stSidebar"] label{color:#94a3b8!important}
section[data-testid="stSidebar"] .stDownloadButton button{background:#2563eb;color:#fff!important;border:none}

/* --- hero ----------------------------------------------------------- */
.hero{background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 50%,#7c3aed 100%);
      padding:2.6rem 2.5rem 2rem;border-radius:20px;color:#fff;margin-bottom:1.8rem;
      box-shadow:0 12px 40px rgba(37,99,235,.35);position:relative;overflow:hidden}
.hero::before{content:"";position:absolute;top:-60px;right:-60px;width:260px;height:260px;
              border-radius:50%;background:rgba(255,255,255,.07)}
.hero h1{font-size:2.4rem;font-weight:800;margin:0;letter-spacing:-.5px}
.hero p{margin:.5rem 0 0;font-size:1.1rem;opacity:.88}

/* --- section titles ------------------------------------------------- */
.sec{font-size:1.25rem;font-weight:700;color:#1e293b;border-left:4px solid #2563eb;
     padding-left:.75rem;margin:1.5rem 0 1rem}

/* --- kpi card ------------------------------------------------------- */
.kpi{background:#fff;border-radius:14px;padding:1.2rem 1.4rem;
     box-shadow:0 2px 14px rgba(0,0,0,.07);text-align:center;border-top:4px solid}
.kpi .lab{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b}
.kpi .val{font-size:1.8rem;font-weight:800;margin:.2rem 0 0}

/* --- badges --------------------------------------------------------- */
.bg{display:inline-block;padding:3px 12px;border-radius:30px;font-weight:700;font-size:.78rem}
.bg-g{background:#dcfce7;color:#166534}.bg-y{background:#fef9c3;color:#854d0e}
.bg-r{background:#fee2e2;color:#991b1b}.bg-x{background:#f1f5f9;color:#475569}

/* --- callout -------------------------------------------------------- */
.call{background:#eff6ff;border-left:4px solid #2563eb;padding:.8rem 1rem;
      border-radius:8px;font-size:.9rem;color:#1e40af;margin-bottom:1rem}
.call-w{background:#fefce8;border-left-color:#d97706;color:#854d0e}

/* --- ai card -------------------------------------------------------- */
.ai-card{background:#fff;border-radius:14px;padding:1.3rem 1.5rem;
         box-shadow:0 2px 12px rgba(0,0,0,.06);margin-bottom:1rem;
         border-left:5px solid #7c3aed}
.ai-card h4{margin:0 0 .5rem;color:#7c3aed}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# SESSION STATE
# ======================================================================

_DEFAULTS = {
    "df": None,
    "results": [],
    "results_df": None,
    "analysis_done": False,
    "ai_market": None,
    "ai_strategy": None,
    "run_ts": None,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ======================================================================
# HELPERS
# ======================================================================

COL_MAP = {
    "sku": ["sku", "product_id", "item_id", "item id", "id"],
    "product_name": [
        "product_name", "product name", "description", "item description",
        "item_description", "name", "title",
    ],
    "brand": ["brand", "manufacturer", "vendor"],
    "category": ["category", "department", "product category", "type"],
    "price": [
        "price", "current price", "current_price", "our price",
        "our_price", "unit price", "retail price", "msrp",
    ],
}


def _norm_cols(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    lc = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for canon, aliases in COL_MAP.items():
        for a in aliases:
            if a in lc:
                mapping[lc[a]] = canon
                break
    needed = {"sku", "product_name", "price"}
    if not needed.issubset(set(mapping.values())):
        return None
    df = df.rename(columns=mapping)
    keep = [c for c in ["sku", "product_name", "brand", "category", "price"] if c in df.columns]
    df = df[keep].copy()
    df["price"] = (
        df["price"].astype(str)
        .str.replace(r"[^\d.,]", "", regex=True)
        .str.replace(",", "")
        .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=["price"])
    if "brand" not in df.columns:
        df["brand"] = "Unknown"
    if "category" not in df.columns:
        df["category"] = "General"
    df["sku"] = df["sku"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()
    return df.reset_index(drop=True)


def _usd(v):
    return f"${v:,.2f}" if v is not None else "\u2014"


def _pct(v):
    if v is None:
        return "\u2014"
    return f"{'+' if v > 0 else ''}{v:.1f}%"


def _badge(status):
    m = {
        "competitive": ("Competitive", "bg-g"),
        "slightly_high": ("Slightly High", "bg-y"),
        "high": ("Overpriced", "bg-r"),
        "unknown": ("No Data", "bg-x"),
    }
    t, c = m.get(status, ("?", "bg-x"))
    return f'<span class="bg {c}">{t}</span>'


def _build_results_df(results: List[ProductComparison]) -> pd.DataFrame:
    rows = []
    for r in results:
        # Get top 2 cheapest competitors with company names
        sorted_comps = sorted(r.competitor_prices, key=lambda c: c.price) if r.competitor_prices else []
        c1 = sorted_comps[0] if len(sorted_comps) >= 1 else None
        c2 = sorted_comps[1] if len(sorted_comps) >= 2 else None

        rows.append(
            {
                "SKU": r.sku,
                "Product": r.description,
                "Brand": r.brand,
                "Category": r.category,
                "Our Price": r.our_price,
                "#1 Price": c1.price if c1 else None,
                "#1 Company": c1.merchant if c1 else None,
                "#2 Price": c2.price if c2 else None,
                "#2 Company": c2.merchant if c2 else None,
                "Min Competitor": r.min_price,
                "Avg Competitor": r.avg_price,
                "Max Competitor": r.max_price,
                "% Diff": r.pct_diff,
                "Savings Opp.": r.savings_opportunity,
                "Status": r.price_status,
                "# Competitors": len(r.competitor_prices),
            }
        )
    return pd.DataFrame(rows)


def _pricing_data_for_ai(results: List[ProductComparison]) -> List[dict]:
    """Flatten results into simple dicts for AI agents."""
    return [
        {
            "sku": r.sku,
            "description": r.description,
            "brand": r.brand,
            "category": r.category,
            "our_price": r.our_price,
            "min_competitor": r.min_price,
            "avg_competitor": r.avg_price,
            "pct_diff": r.pct_diff,
            "savings_opportunity": r.savings_opportunity,
            "status": r.price_status,
            "n_competitors": len(r.competitor_prices),
        }
        for r in results
    ]


def _to_excel(rdf: pd.DataFrame, results: List[ProductComparison]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        rdf.to_excel(w, sheet_name="Summary", index=False)
        detail = []
        for r in results:
            for c in r.competitor_prices:
                detail.append(
                    {
                        "SKU": r.sku,
                        "Our Product": r.description,
                        "Our Price": r.our_price,
                        "Merchant": c.merchant,
                        "Competitor Title": c.title,
                        "Competitor Price": c.price,
                        "URL": c.url,
                    }
                )
        if detail:
            pd.DataFrame(detail).to_excel(w, sheet_name="Competitor Detail", index=False)
    return buf.getvalue()


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Required for AI-powered insights (match validation, market analysis, pricing strategy). Scraping works without it.",
    )
    model_choice = st.selectbox(
        "Claude Model",
        ["claude-sonnet-4-5-20250514", "claude-haiku-4-5-20251001"],
        help="Sonnet is more capable; Haiku is faster & cheaper.",
    )

    st.markdown("---")
    max_results = st.slider("Max competitors per product", 3, 10, 5)
    delay = st.slider("Request delay (seconds)", 1.0, 5.0, 2.5, 0.5)

    st.markdown("---")
    st.markdown("### CSV Format")
    st.markdown(
        "Columns: **SKU**, **Product Name**, **Brand**, **Category**, **Price**"
    )
    st.download_button(
        "Download Sample CSV",
        data=(
            "sku,product_name,brand,category,price\n"
            "SKU-001,Galaxy S24 Ultra 256GB,Samsung,Electronics,1208.86\n"
            "SKU-002,AirPods Pro 2nd Gen,Apple,Electronics,228.46\n"
            "SKU-003,Artisan Stand Mixer 5qt,KitchenAid,Home & Kitchen,420.90\n"
        ),
        file_name="sample_products.csv",
        mime="text/csv",
    )

# ======================================================================
# HERO
# ======================================================================

st.markdown(
    '<div class="hero">'
    "<h1>PriceWatch Pro</h1>"
    "<p>Upload your product catalogue \u2014 we scan the web and let Claude AI agents "
    "tell you exactly how your prices stack up.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ======================================================================
# STEP 1 -- UPLOAD
# ======================================================================

st.markdown('<div class="sec">1 &nbsp; Upload Your Product Catalogue</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a CSV file (SKU, Product Name, Brand, Category, Price)",
    type=["csv"],
)

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        df = _norm_cols(raw)
        if df is None:
            st.error(
                "Could not detect required columns (SKU, Product Name, Price). "
                "Check sidebar for the expected format."
            )
        else:
            st.session_state["df"] = df
            st.success(f"{len(df)} products loaded from **{uploaded.name}**")
            with st.expander("Preview uploaded data", expanded=True):
                st.dataframe(
                    df.style.format({"price": "${:.2f}"}),
                    use_container_width=True,
                    height=min(320, 60 + len(df) * 35),
                )
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

# ======================================================================
# STEP 2 -- RUN ANALYSIS
# ======================================================================

st.markdown('<div class="sec">2 &nbsp; Run Competitor Price Scan</div>', unsafe_allow_html=True)

df: Optional[pd.DataFrame] = st.session_state.get("df")

if df is None:
    st.markdown('<div class="call">Upload a CSV above to get started.</div>', unsafe_allow_html=True)
    st.stop()

# --- Filters ---
search_query = st.text_input("Search by product name", placeholder="e.g. keyboard, airpods, mixer...")

filter_cols = st.columns([2, 2, 1])
with filter_cols[0]:
    cats = sorted(df["category"].unique())
    sel_cats = st.multiselect("Filter categories", cats, default=cats)
with filter_cols[1]:
    brands = sorted(df[df["category"].isin(sel_cats)]["brand"].unique())
    sel_brands = st.multiselect("Filter brands", brands, default=brands)
with filter_cols[2]:
    max_products = st.number_input("Max products", 1, len(df), min(len(df), 20), help="Limit for faster demo runs")

work_df = df[df["category"].isin(sel_cats) & df["brand"].isin(sel_brands)]
if search_query.strip():
    work_df = work_df[work_df["product_name"].str.contains(search_query.strip(), case=False, na=False)]
work_df = work_df.head(max_products)
n = len(work_df)
est = int(n * (delay + 1.2))

st.markdown(
    f'<div class="call">Ready to scan <strong>{n}</strong> products.  '
    f"Estimated time: <strong>~{est}s</strong></div>",
    unsafe_allow_html=True,
)

run_col, reset_col = st.columns([3, 1])
with run_col:
    run_btn = st.button("Start Competitor Price Scan", type="primary", use_container_width=True)
with reset_col:
    if st.button("Reset", use_container_width=True):
        for k in _DEFAULTS:
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()

if run_btn:
    scraper = PriceScraper(delay=delay, max_results=max_results)
    results: List[ProductComparison] = []

    bar = st.progress(0, text="Initialising scanner...")
    info = st.empty()

    for idx, (_, row) in enumerate(work_df.iterrows()):
        pct = idx / n
        bar.progress(pct, text=f"Scanning {idx + 1}/{n}: {row['product_name'][:55]}...")
        with info.container():
            st.info(f"Searching for **{row['product_name']}** ({row['brand']})")

        comp = scraper.get_competitor_prices(
            sku=str(row["sku"]),
            description=str(row["product_name"]),
            brand=str(row.get("brand", "")),
            category=str(row.get("category", "")),
            our_price=float(row["price"]),
        )
        results.append(comp)

    bar.progress(1.0, text="Scan complete!")
    info.empty()

    results = [r for r in results if r.competitor_prices]
    st.session_state["results"] = results
    st.session_state["results_df"] = _build_results_df(results)
    st.session_state["analysis_done"] = True
    st.session_state["run_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["ai_market"] = None
    st.session_state["ai_strategy"] = None
    st.rerun()

# ======================================================================
# STEP 3 -- RESULTS DASHBOARD
# ======================================================================

results: List[ProductComparison] = st.session_state.get("results", [])
if not results:
    st.stop()

rdf: pd.DataFrame = st.session_state["results_df"]

st.markdown(
    f'<div class="sec">3 &nbsp; Results Dashboard '
    f'<span style="font-size:.85rem;font-weight:400;color:#64748b">'
    f'&mdash; {st.session_state["run_ts"]}</span></div>',
    unsafe_allow_html=True,
)

# --- KPIs ---
total = len(results)
comp_n = sum(1 for r in results if r.price_status == "competitive")
high_n = sum(1 for r in results if r.price_status == "high")
slight_n = sum(1 for r in results if r.price_status == "slightly_high")
total_gap = sum(r.savings_opportunity for r in results)
score = comp_n / max(total, 1) * 100

kpis = [
    ("Products Scanned", str(total), "#2563eb"),
    ("Competitive", str(comp_n), "#16a34a"),
    ("Slightly High", str(slight_n), "#d97706"),
    ("Overpriced", str(high_n), "#dc2626"),
    ("Competitiveness Score", f"{score:.0f}%", "#7c3aed"),
    ("Total Savings Gap", _usd(total_gap), "#0891b2"),
]

cols = st.columns(len(kpis))
for col, (lab, val, clr) in zip(cols, kpis):
    col.markdown(
        f'<div class="kpi" style="border-top-color:{clr}">'
        f'<div class="lab">{lab}</div>'
        f'<div class="val" style="color:{clr}">{val}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tab_tbl, tab_detail, tab_charts, tab_ai, tab_export = st.tabs(
    ["Summary Table", "Product Deep Dive", "Charts & Insights", "AI Strategy Room", "Export"]
)

# ======================================================================
# TAB 1 -- SUMMARY TABLE
# ======================================================================
with tab_tbl:
    fc1, fc2 = st.columns(2)
    with fc1:
        st_filter = st.multiselect(
            "Status",
            ["competitive", "slightly_high", "high"],
            default=["competitive", "slightly_high", "high"],
            format_func=lambda s: {"competitive": "Competitive", "slightly_high": "Slightly High", "high": "Overpriced"}[s],
        )
    with fc2:
        sort_by = st.selectbox("Sort by", ["% Diff", "Savings Opp.", "Our Price", "Min Competitor"])

    view = rdf[rdf["Status"].isin(st_filter)].sort_values(
        sort_by, ascending=(sort_by not in ["% Diff", "Savings Opp."]), na_position="last"
    )

    disp = view.copy()
    for c in ["Our Price", "#1 Price", "#2 Price", "Min Competitor", "Avg Competitor", "Max Competitor", "Savings Opp."]:
        disp[c] = disp[c].apply(_usd)
    disp["% Diff"] = disp["% Diff"].apply(_pct)
    disp["Status"] = disp["Status"].map(
        {"competitive": "Competitive", "slightly_high": "Slightly High", "high": "Overpriced"}
    )
    # Reorder columns to show top competitors prominently
    col_order = ["SKU", "Product", "Brand", "Category", "Our Price",
                 "#1 Price", "#1 Company", "#2 Price", "#2 Company",
                 "Min Competitor", "Avg Competitor", "Max Competitor",
                 "% Diff", "Savings Opp.", "Status", "# Competitors"]
    disp = disp[[c for c in col_order if c in disp.columns]]
    st.dataframe(disp, use_container_width=True, height=min(600, 60 + len(disp) * 36))

# ======================================================================
# TAB 2 -- PRODUCT DEEP DIVE
# ======================================================================
with tab_detail:
    sel_sku = st.selectbox(
        "Select a product",
        [r.sku for r in results],
        format_func=lambda s: next(
            (f"{r.sku}  --  {r.description[:50]}" for r in results if r.sku == s), s
        ),
    )
    res = next((r for r in results if r.sku == sel_sku), None)
    if res:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Our Price", _usd(res.our_price))
        m2.metric(
            "Lowest Competitor",
            _usd(res.min_price),
            delta=_pct(res.pct_diff) if res.pct_diff is not None else None,
            delta_color="inverse",
        )
        m3.metric("Market Average", _usd(res.avg_price))
        m4.metric("Savings Gap", _usd(res.savings_opportunity))

        if res.competitor_prices:
            st.markdown("#### Competitor Listings")
            cdf = pd.DataFrame(
                [
                    {"Merchant": c.merchant, "Price": _usd(c.price), "Title": c.title[:80], "URL": c.url}
                    for c in sorted(res.competitor_prices, key=lambda x: x.price)
                ]
            )
            st.dataframe(cdf, use_container_width=True, hide_index=True)

            chart_data = [("Our Price", res.our_price)] + [
                (c.merchant[:18], c.price) for c in res.competitor_prices
            ]
            fig = go.Figure(
                go.Bar(
                    x=[d[0] for d in chart_data],
                    y=[d[1] for d in chart_data],
                    marker_color=["#2563eb" if d[0] == "Our Price" else "#94a3b8" for d in chart_data],
                    text=[_usd(d[1]) for d in chart_data],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title=f"Price Comparison: {res.description[:50]}",
                yaxis_title="Price (USD)",
                plot_bgcolor="white",
                height=360,
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- AI Match Validation ---
            if api_key:
                if st.button("Validate matches with AI", key="validate_btn"):
                    try:
                        from ai_agents import PriceIntelligenceAgents

                        agents = PriceIntelligenceAgents(api_key, model_choice)
                        comp_dicts = [
                            {"merchant": c.merchant, "title": c.title, "price": c.price}
                            for c in res.competitor_prices
                        ]
                        with st.spinner("Claude is validating product matches..."):
                            val = agents.validate_matches(
                                res.description, res.brand, res.our_price, comp_dicts
                            )
                        st.markdown("##### AI Match Validation")
                        st.markdown(f"**{val.get('summary', '')}**")
                        scores = val.get("scores", [])
                        flags = val.get("flags", [])
                        for i, c in enumerate(sorted(res.competitor_prices, key=lambda x: x.price)):
                            sc = scores[i] if i < len(scores) else 0.5
                            fl = flags[i] if i < len(flags) else ""
                            bar_color = "#16a34a" if sc >= 0.7 else "#d97706" if sc >= 0.4 else "#dc2626"
                            st.markdown(
                                f"**{c.merchant}** ({_usd(c.price)}) &mdash; "
                                f'Match: <span style="color:{bar_color};font-weight:700">{sc:.0%}</span> '
                                f"&mdash; {fl}",
                                unsafe_allow_html=True,
                            )
                    except Exception as exc:
                        st.error(f"Validation failed: {exc}")
        elif res.error:
            st.warning(res.error)
        else:
            st.info("No competitor data found for this product.")

# ======================================================================
# TAB 3 -- CHARTS
# ======================================================================
with tab_charts:
    has = rdf[rdf["Min Competitor"].notna()].copy()
    if has.empty:
        st.info("No competitor data to chart.")
    else:
        cl, cr = st.columns(2)

        # 1 -- grouped bar
        with cl:
            top = has.nlargest(15, "Savings Opp.").copy()
            top["Short"] = top["Product"].str[:28]
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(name="Our Price", x=top["Short"], y=top["Our Price"], marker_color="#2563eb"))
            fig1.add_trace(go.Bar(name="Lowest Competitor", x=top["Short"], y=top["Min Competitor"], marker_color="#f59e0b"))
            fig1.update_layout(
                title="Our Price vs Lowest Competitor (Top 15 Gaps)",
                barmode="group", plot_bgcolor="white", height=420,
                legend=dict(orientation="h", y=1.08), xaxis_tickangle=-35,
                margin=dict(b=100, t=60), yaxis_title="USD",
            )
            st.plotly_chart(fig1, use_container_width=True)

        # 2 -- donut
        with cr:
            sc = rdf["Status"].value_counts().reset_index()
            sc.columns = ["Status", "Count"]
            lbl = {"competitive": "Competitive", "slightly_high": "Slightly High", "high": "Overpriced", "unknown": "No Data"}
            clrs = {"competitive": "#16a34a", "slightly_high": "#f59e0b", "high": "#dc2626", "unknown": "#94a3b8"}
            sc["Label"] = sc["Status"].map(lbl)
            fig2 = px.pie(sc, names="Label", values="Count", hole=.55, color="Status", color_discrete_map=clrs)
            fig2.update_traces(textposition="outside", textinfo="percent+label")
            fig2.update_layout(title="Price Competitiveness Breakdown", height=420, showlegend=False, margin=dict(t=60, b=40))
            st.plotly_chart(fig2, use_container_width=True)

        # 3 -- % diff bar
        diff_df = has.dropna(subset=["% Diff"]).sort_values("% Diff", ascending=False).head(20)
        diff_df["Short"] = diff_df["Product"].str[:32]
        diff_df["Color"] = diff_df["% Diff"].apply(lambda x: "#dc2626" if x > 15 else "#f59e0b" if x > 0 else "#16a34a")
        fig3 = go.Figure(
            go.Bar(
                x=diff_df["Short"], y=diff_df["% Diff"],
                marker_color=diff_df["Color"], opacity=.85,
                text=diff_df["% Diff"].apply(lambda v: f"{v:+.1f}%"), textposition="outside",
            )
        )
        fig3.add_hline(y=0, line_color="#1e293b", line_width=1.5)
        fig3.add_hline(y=15, line_dash="dash", line_color="#dc2626", line_width=1,
                       annotation_text="15% threshold", annotation_position="top right")
        fig3.update_layout(
            title="Price Premium / Discount vs Cheapest Competitor",
            plot_bgcolor="white", height=440, yaxis_title="% Difference",
            xaxis_tickangle=-35, margin=dict(b=120, t=70),
        )
        st.plotly_chart(fig3, use_container_width=True)

        cl2, cr2 = st.columns(2)

        # 4 -- scatter
        with cl2:
            sct = has.dropna(subset=["Avg Competitor"]).copy()
            sct["Label"] = sct["Status"].map(lbl)
            fig4 = px.scatter(
                sct, x="Avg Competitor", y="Our Price", color="Label",
                color_discrete_map={v: clrs[k] for k, v in lbl.items()},
                hover_data={"Product": True, "SKU": True},
            )
            pmin = min(sct["Our Price"].min(), sct["Avg Competitor"].min()) * .9
            pmax = max(sct["Our Price"].max(), sct["Avg Competitor"].max()) * 1.1
            fig4.add_trace(go.Scatter(x=[pmin, pmax], y=[pmin, pmax], mode="lines",
                                      line=dict(color="#64748b", dash="dot", width=1.5), name="Parity"))
            fig4.update_layout(title="Our Price vs Market Average", plot_bgcolor="white",
                               height=420, xaxis_title="Market Avg (USD)", yaxis_title="Our Price (USD)")
            st.plotly_chart(fig4, use_container_width=True)

        # 5 -- category heatmap
        with cr2:
            cat_agg = has.groupby("Category").agg(
                avg_diff=("% Diff", "mean"),
                count=("SKU", "count"),
                gap=("Savings Opp.", "sum"),
            ).reset_index()
            fig5 = px.bar(
                cat_agg.sort_values("avg_diff", ascending=True),
                x="avg_diff", y="Category", orientation="h",
                color="avg_diff",
                color_continuous_scale=["#16a34a", "#fbbf24", "#dc2626"],
                text=cat_agg.sort_values("avg_diff", ascending=True)["avg_diff"].apply(lambda v: f"{v:+.1f}%"),
            )
            fig5.update_layout(
                title="Avg Price Premium by Category",
                plot_bgcolor="white", height=420,
                xaxis_title="Avg % Difference", yaxis_title="",
                coloraxis_showscale=False,
            )
            fig5.update_traces(textposition="outside")
            st.plotly_chart(fig5, use_container_width=True)

        # 6 -- brand analysis
        brand_agg = has.groupby("Brand").agg(
            avg_diff=("% Diff", "mean"), n=("SKU", "count"), gap=("Savings Opp.", "sum"),
        ).reset_index().sort_values("avg_diff", ascending=False).head(20)
        fig6 = go.Figure(
            go.Bar(
                x=brand_agg["Brand"], y=brand_agg["avg_diff"],
                marker_color=brand_agg["avg_diff"].apply(
                    lambda v: "#dc2626" if v > 15 else "#f59e0b" if v > 0 else "#16a34a"
                ),
                text=brand_agg["avg_diff"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
            )
        )
        fig6.add_hline(y=0, line_color="#1e293b", line_width=1.5)
        fig6.update_layout(
            title="Average Price Premium by Brand (Top 20)",
            plot_bgcolor="white", height=400, yaxis_title="Avg % Diff",
            xaxis_tickangle=-40, margin=dict(b=100, t=60),
        )
        st.plotly_chart(fig6, use_container_width=True)

# ======================================================================
# TAB 4 -- AI STRATEGY ROOM
# ======================================================================
with tab_ai:
    if not api_key:
        st.markdown(
            '<div class="call-w call">Enter your Anthropic API key in the '
            "sidebar to unlock AI-powered market analysis and pricing strategy.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown(
        '<div class="call">Three specialised Claude agents collaborate to analyse your '
        "pricing data: a **Market Analyst**, a **Pricing Strategist**, and a "
        "**Match Validator** (available in Product Deep Dive tab).</div>",
        unsafe_allow_html=True,
    )

    if st.button("Run AI Analysis", type="primary", use_container_width=True):
        try:
            from ai_agents import PriceIntelligenceAgents

            agents = PriceIntelligenceAgents(api_key, model_choice)
            pd_list = _pricing_data_for_ai(results)

            with st.spinner("Agent 1/2: Market Analyst is studying your data..."):
                mkt = agents.analyze_market(pd_list)
                st.session_state["ai_market"] = mkt

            with st.spinner("Agent 2/2: Pricing Strategist is building recommendations..."):
                strat = agents.generate_strategy(pd_list, mkt)
                st.session_state["ai_strategy"] = strat

            st.rerun()
        except Exception as exc:
            st.error(f"AI analysis failed: {exc}")

    # --- Display AI results ---
    mkt = st.session_state.get("ai_market")
    strat = st.session_state.get("ai_strategy")

    if mkt:
        st.markdown("---")
        st.markdown("### Market Intelligence Report")

        # Executive summary
        st.markdown(
            f'<div class="ai-card"><h4>Executive Summary</h4>{mkt.get("executive_summary", "")}</div>',
            unsafe_allow_html=True,
        )

        # Key findings
        findings = mkt.get("key_findings", [])
        if findings:
            st.markdown("#### Key Findings")
            for i, f in enumerate(findings, 1):
                st.markdown(f"**{i}.** {f}")

        fc1, fc2 = st.columns(2)
        ci = mkt.get("category_insights", {})
        with fc1:
            st.markdown("#### Strongest Categories")
            for s in ci.get("strongest", []):
                st.markdown(f"- {s}")
        with fc2:
            st.markdown("#### Weakest Categories")
            for w in ci.get("weakest", []):
                st.markdown(f"- {w}")

        # Risks
        risks = mkt.get("risk_alerts", [])
        if risks:
            st.markdown("#### Risk Alerts")
            for r in risks:
                st.warning(r)

        # Opportunities
        opps = mkt.get("opportunities", [])
        if opps:
            st.markdown("#### Opportunities")
            for o in opps:
                st.success(o)

    if strat:
        st.markdown("---")
        st.markdown("### Pricing Strategy Recommendations")

        st.markdown(
            f'<div class="ai-card"><h4>Overall Recommendation</h4>'
            f'{strat.get("overall_recommendation", "")}</div>',
            unsafe_allow_html=True,
        )

        # Immediate actions table
        actions = strat.get("immediate_actions", [])
        if actions:
            st.markdown("#### Immediate Price Actions")
            act_df = pd.DataFrame(actions)
            rename = {
                "product": "Product", "sku": "SKU", "action": "Action",
                "current_price": "Current", "recommended_price": "Recommended",
                "rationale": "Rationale",
            }
            act_df = act_df.rename(columns={k: v for k, v in rename.items() if k in act_df.columns})
            st.dataframe(act_df, use_container_width=True, hide_index=True)

        # Category strategies
        cat_strats = strat.get("category_strategies", [])
        if cat_strats:
            st.markdown("#### Category-Level Strategies")
            for cs in cat_strats:
                st.markdown(
                    f'<div class="ai-card"><h4>{cs.get("category", "")}</h4>'
                    f'<p><strong>Strategy:</strong> {cs.get("strategy", "")}</p>'
                    f'<p><strong>Expected Impact:</strong> {cs.get("expected_impact", "")}</p></div>',
                    unsafe_allow_html=True,
                )

        # Margin opportunities
        margins = strat.get("margin_opportunities", [])
        if margins:
            st.markdown("#### Margin Improvement Opportunities")
            st.markdown("*Products where we are priced below market and can safely increase.*")
            mar_df = pd.DataFrame(margins)
            rename2 = {
                "product": "Product", "current_price": "Current",
                "market_price": "Market", "recommended_price": "Recommended",
                "upside_pct": "Upside %",
            }
            mar_df = mar_df.rename(columns={k: v for k, v in rename2.items() if k in mar_df.columns})
            st.dataframe(mar_df, use_container_width=True, hide_index=True)

# ======================================================================
# TAB 5 -- EXPORT
# ======================================================================
with tab_export:
    st.markdown("#### Download Your Results")

    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "Download Summary CSV",
            data=rdf.to_csv(index=False).encode(),
            file_name=f"pricewatch_summary_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ec2:
        try:
            xl = _to_excel(rdf, results)
            st.download_button(
                "Download Full Excel Report",
                data=xl,
                file_name=f"pricewatch_report_{datetime.now():%Y%m%d_%H%M}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.info("Install `openpyxl` to enable Excel export.")

    # If AI results exist, offer those too
    mkt = st.session_state.get("ai_market")
    strat = st.session_state.get("ai_strategy")
    if mkt or strat:
        import json

        ai_report = {"market_analysis": mkt, "pricing_strategy": strat}
        st.download_button(
            "Download AI Strategy Report (JSON)",
            data=json.dumps(ai_report, indent=2),
            file_name=f"pricewatch_ai_report_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown(
        "| Export | Contents |\n"
        "|--------|----------|\n"
        "| **Summary CSV** | One row per product: our price, competitor min/avg/max, % diff, status |\n"
        "| **Excel Report** | Sheet 1 = Summary, Sheet 2 = Every competitor listing with merchant & URL |\n"
        "| **AI Report** | Full JSON of market analysis + pricing strategy from Claude agents |"
    )

# ======================================================================
# FOOTER
# ======================================================================
st.markdown(
    '<hr style="border:none;border-top:1px solid #e2e8f0;margin:2rem 0 1rem">'
    '<p style="text-align:center;color:#94a3b8;font-size:.8rem">'
    "PriceWatch Pro &middot; MSIS 521 Final Project &middot; "
    "Powered by DuckDuckGo Search &amp; Claude AI Agents"
    "</p>",
    unsafe_allow_html=True,
)
