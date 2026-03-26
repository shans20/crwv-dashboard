#!/usr/bin/env python3
"""
CRWV Thesis Tracker — Streamlit Web Dashboard
Launch: streamlit run crwv_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(page_title="CRWV Thesis Tracker", page_icon="⚡", layout="wide")

# --- Dark theme styling ---
st.markdown("""
<style>
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 13px; color: #8b949e; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("CRWV Contract Anatomy Tracker")
st.caption(f"CoreWeave per-GW economics model  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ============================================================
# MODEL ENGINE
# ============================================================
def compute_model(gpu_hr, gpus_per_gw_k, capex_per_gpu_k, prepay_pct, finance_rate, opex_pct,
                  yr1_ramp, ext_rev_pct, share_count_mn,
                  storage_pct, software_pct, n_gw=1.0, interest_only=False,
                  contract_yrs=5, total_yrs=8):
    """Compute per-GW cash flow model."""
    gpus_per_gw = gpus_per_gw_k * 1000
    capex_per_gpu = capex_per_gpu_k * 1000  # slider in $K, convert to $
    capex = gpus_per_gw * capex_per_gpu / 1e9 * n_gw  # $Bn

    base_annual_rev = gpu_hr * 8760 * gpus_per_gw / 1e9 * n_gw  # $Bn (GPU only)
    storage_rev = base_annual_rev * storage_pct
    software_rev = base_annual_rev * software_pct
    total_annual_rev = base_annual_rev + storage_rev + software_rev

    tcv = total_annual_rev * contract_yrs
    prepay = tcv * prepay_pct
    amt_financed = capex - prepay

    # Debt service (amortized over contract length)
    if interest_only:
        annual_interest = amt_financed * finance_rate
        annual_principal = amt_financed / contract_yrs
        annual_pmt = annual_interest + annual_principal
    else:
        if finance_rate > 0 and amt_financed > 0:
            n = contract_yrs
            annual_pmt = amt_financed * (finance_rate * (1 + finance_rate)**n) / ((1 + finance_rate)**n - 1)
        else:
            annual_pmt = amt_financed / contract_yrs

    ext_yrs = total_yrs - contract_yrs  # how many extension years

    years = []
    cumulative_cf = 0

    for yr in range(0, total_yrs + 1):  # yr 0 through total_yrs
        if yr == 0:
            rev = 0
            opex = 0
            debt = 0
            net_cf = -capex + prepay
            interest = 0
        elif yr <= contract_yrs:
            # Contract years
            rev = total_annual_rev * (yr1_ramp if yr == 1 else 1.0)
            opex = rev * opex_pct
            if interest_only:
                remaining = amt_financed * (1 - (yr - 1) / contract_yrs)
                interest = remaining * finance_rate
                debt = amt_financed / contract_yrs + interest
            else:
                debt = annual_pmt
                outstanding = amt_financed
                for i in range(1, yr):
                    outstanding -= (annual_pmt - outstanding * finance_rate)
                interest = max(0, outstanding * finance_rate)
            net_cf = rev - opex - debt
        else:
            # Extension / recontracting years — no debt, opex flat from contract period
            rev = total_annual_rev * ext_rev_pct
            opex = total_annual_rev * opex_pct  # flat: same $ as steady-state contract years
            debt = 0
            interest = 0
            net_cf = rev - opex

        cumulative_cf += net_cf
        cf_per_share = net_cf / share_count_mn * 1000

        if yr <= contract_yrs:
            label = f"Yr {yr}"
        elif yr == contract_yrs + 1:
            label = f"Yr {yr} ext"
        else:
            label = f"Yr {yr} re"

        years.append({
            "year": yr,
            "label": label,
            "revenue": rev,
            "opex": opex,
            "interest": interest,
            "debt_service": debt,
            "net_cf": net_cf,
            "cumulative_cf": cumulative_cf,
            "cf_per_share": cf_per_share,
        })

    # Payback period (interpolate)
    payback = None
    for i in range(1, len(years)):
        if years[i]["cumulative_cf"] >= 0 and years[i-1]["cumulative_cf"] < 0:
            prev = years[i-1]["cumulative_cf"]
            curr = years[i]["cumulative_cf"]
            payback = (i - 1) + (-prev) / (curr - prev)
            break

    # Profit metrics (contract period only)
    profit_contract = sum(y["net_cf"] for y in years[:contract_yrs + 1])  # yr0 through contract end
    return_contract = profit_contract / capex if capex > 0 else 0
    take_rate = return_contract / contract_yrs

    # Interest as % of TCV
    total_interest = sum(y["interest"] for y in years[:contract_yrs + 1])
    interest_pct_tcv = total_interest / tcv if tcv > 0 else 0

    # Revenue breakdown
    gpu_rev_annual = base_annual_rev
    storage_rev_annual = storage_rev
    software_rev_annual = software_rev

    # Contribution margin
    contribution_margin = 1 - opex_pct

    return {
        "years": years,
        "capex": capex,
        "tcv": tcv,
        "prepay": prepay,
        "amt_financed": amt_financed,
        "annual_rev": total_annual_rev,
        "gpu_rev": gpu_rev_annual,
        "storage_rev": storage_rev_annual,
        "software_rev": software_rev_annual,
        "payback": payback,
        "profit_contract": profit_contract,
        "return_contract": return_contract,
        "contract_yrs": contract_yrs,
        "total_yrs": total_yrs,
        "take_rate": take_rate,
        "interest_pct_tcv": interest_pct_tcv,
        "contribution_margin": contribution_margin,
        "annual_pmt": annual_pmt if amt_financed > 0 else 0,
    }


# ============================================================
# SIDEBAR: Scenario Builder
# ============================================================
st.sidebar.header("Scenario Builder")
st.sidebar.caption("Override contract assumptions to model scenarios")

st.sidebar.markdown("### Contract Economics")
gpu_hr = st.sidebar.slider("GPU/hr ($)", 1.50, 8.00, 3.30, 0.10)
gpus_per_gw_k = st.sidebar.slider("GPUs per GW (thousands)", 300, 1000, 600, 50)
capex_per_gpu_k = st.sidebar.slider("NVIDIA Capex per GPU ($K)", 30, 120, 60, 5)
contract_yrs = st.sidebar.slider("Contract Length (yrs)", 3, 8, 5, 1)
total_yrs = st.sidebar.slider("Total Asset Life (yrs)", contract_yrs, 10, max(contract_yrs, 8), 1)
prepay_pct = st.sidebar.slider("Prepay % of TCV", 0.10, 0.35, 0.20, 0.01)
finance_rate = st.sidebar.slider("Finance Rate (%)", 3.0, 12.0, 7.0, 0.25)
finance_rate_dec = finance_rate / 100
interest_only = st.sidebar.checkbox("Interest-only debt structure", value=False)

st.sidebar.markdown("### Operations")
opex_pct = st.sidebar.slider("Opex % of Revenue", 0.15, 0.40, 0.25, 0.01)
yr1_ramp = st.sidebar.slider("Yr 1 Ramp (J-curve)", 0.50, 1.00, 0.75, 0.05)
ext_rev_pct = st.sidebar.slider("Extension Year Revenue %", 0.50, 1.00, 0.75, 0.05)

st.sidebar.markdown("### Platform Revenue")
storage_pct = st.sidebar.slider("Storage Revenue (% of GPU)", 0.0, 0.25, 0.02, 0.01)
software_pct = st.sidebar.slider("Software Revenue (% of GPU)", 0.0, 0.20, 0.0, 0.01)

st.sidebar.markdown("### Corporate")
share_count = st.sidebar.slider("Share Count (mn)", 400, 700, 530, 10)
n_gw = st.sidebar.slider("GW Deployed", 0.5, 10.0, 1.0, 0.5)

# ============================================================
# RUN MODEL
# ============================================================
model = compute_model(
    gpu_hr=gpu_hr,
    gpus_per_gw_k=gpus_per_gw_k,
    capex_per_gpu_k=capex_per_gpu_k,
    prepay_pct=prepay_pct,
    finance_rate=finance_rate_dec,
    opex_pct=opex_pct,
    yr1_ramp=yr1_ramp,
    ext_rev_pct=ext_rev_pct,
    share_count_mn=share_count,
    storage_pct=storage_pct,
    software_pct=software_pct,
    n_gw=n_gw,
    interest_only=interest_only,
    contract_yrs=contract_yrs,
    total_yrs=total_yrs,
)

# ============================================================
# TOP METRICS
# ============================================================
st.markdown("---")
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Capex/GW", f"${model['capex']/n_gw:.1f}B", f"Revenue/GW: ${model['annual_rev']/n_gw:.1f}B")
c2.metric("Payback Period", f"{model['payback']:.1f} yrs" if model['payback'] else "N/A",
          f"IR guide: 2.5-3.0 yrs")
c3.metric(f"{contract_yrs}yr Return", f"{model['return_contract']:.0%}", f"Profit: ${model['profit_contract']:.1f}B")
c4.metric("Take Rate", f"{model['take_rate']:.1%}", f"per yr on capex")
c5.metric("Interest/TCV", f"{model['interest_pct_tcv']:.1%}", f"IR deck: ~8%")
c6.metric("Avg CF/Share", f"${sum(y['cf_per_share'] for y in model['years'])/len(model['years']):.2f}",
          f"{n_gw:.1f} GW deployed")

# ============================================================
# ROW 1: Cash Flow Waterfall + Cumulative CF
# ============================================================
st.markdown("---")
col_wf, col_cum = st.columns(2)

with col_wf:
    st.subheader("Annual Cash Flow by Year")
    labels = [y["label"] for y in model["years"]]
    values = [y["net_cf"] for y in model["years"]]
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in values]

    fig_cf = go.Figure(go.Bar(
        x=labels, y=values,
        text=[f"${v:.1f}B" for v in values],
        textposition="outside",
        marker_color=colors,
    ))
    fig_cf.update_layout(
        height=420, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        yaxis=dict(gridcolor="#21262d", title="$ Billions"),
        xaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_cf, use_container_width=True)

with col_cum:
    st.subheader("Cumulative Cash Flow / Share")
    labels = [y["label"] for y in model["years"]]
    cum_values = []
    running = 0
    for y in model["years"]:
        running += y["cf_per_share"]
        cum_values.append(running)

    colors_cum = ["#3fb950" if v >= 0 else "#f85149" for v in cum_values]

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=labels, y=cum_values,
        mode="lines+markers+text",
        text=[f"${v:.1f}" for v in cum_values],
        textposition="top center",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=10, color=colors_cum),
    ))
    # Add zero line
    fig_cum.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)

    # Add payback annotation
    if model["payback"]:
        fig_cum.add_vline(x=model["payback"], line_dash="dot", line_color="#d29922", opacity=0.7)
        fig_cum.add_annotation(x=model["payback"], y=0, text=f"Payback: {model['payback']:.1f}yr",
                               showarrow=True, arrowhead=2, font=dict(color="#d29922", size=12))

    fig_cum.update_layout(
        height=420, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        yaxis=dict(gridcolor="#21262d", title="$/share (cumulative)"),
        xaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# ============================================================
# ROW 2: Revenue Composition + Contract Waterfall
# ============================================================
st.markdown("---")
col_rev, col_waterfall = st.columns(2)

with col_rev:
    st.subheader("Revenue Composition (Annual/GW)")
    rev_labels = ["GPU Compute", "Storage", "Software/Platform"]
    rev_values = [model["gpu_rev"]/n_gw, model["storage_rev"]/n_gw, model["software_rev"]/n_gw]
    rev_colors = ["#58a6ff", "#3fb950", "#bc8cff"]

    # Only show non-zero
    filtered = [(l, v, c) for l, v, c in zip(rev_labels, rev_values, rev_colors) if v > 0.001]
    if filtered:
        fl, fv, fc = zip(*filtered)
    else:
        fl, fv, fc = rev_labels[:1], [model["gpu_rev"]/n_gw], rev_colors[:1]

    fig_rev = go.Figure(go.Pie(
        labels=list(fl), values=list(fv),
        hole=0.5, textinfo="label+percent",
        textposition="outside",
        marker=dict(colors=list(fc)),
    ))
    fig_rev.update_layout(
        height=400, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        showlegend=False,
        annotations=[dict(text=f"${model['annual_rev']/n_gw:.1f}B", x=0.5, y=0.5,
                         font_size=18, font_color="#58a6ff", showarrow=False)],
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Platform parity quote
    gpu_pct = model["gpu_rev"] / model["annual_rev"] * 100 if model["annual_rev"] > 0 else 100
    st.info(f'GPU: {gpu_pct:.0f}% of revenue | CEO: *"I am 12-15% of the GPU market, 0.1% of storage. My goal is to bring those to parity."*')

with col_waterfall:
    st.subheader("Contract Economics Waterfall (per GW, annual)")
    steady_yr = model["years"][2]  # Yr 2 = steady state

    wf_labels = ["Revenue", "Opex", "Debt Service", "Net CF"]
    wf_measures = ["relative", "relative", "relative", "total"]
    wf_values = [
        steady_yr["revenue"] / n_gw,
        -steady_yr["opex"] / n_gw,
        -steady_yr["debt_service"] / n_gw,
        0,
    ]
    wf_text = [f"${abs(v):.1f}B" for v in wf_values]
    wf_text[3] = f"${steady_yr['net_cf']/n_gw:.1f}B"

    fig_wf = go.Figure(go.Waterfall(
        x=wf_labels, y=wf_values, measure=wf_measures,
        text=wf_text, textposition="outside",
        connector=dict(line=dict(color="#30363d")),
        increasing=dict(marker_color="#3fb950"),
        decreasing=dict(marker_color="#f85149"),
        totals=dict(marker_color="#58a6ff"),
    ))
    fig_wf.update_layout(
        height=400, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        yaxis=dict(gridcolor="#21262d", title="$ Billions"),
        xaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    contribution = (1 - opex_pct) * 100
    st.info(f"Contribution margin: {contribution:.0f}% | IR guided: mid-20s")

# ============================================================
# ROW 3: Sensitivity Heatmaps
# ============================================================
st.markdown("---")
col_payback_sens, col_return_sens = st.columns(2)

with col_payback_sens:
    st.subheader("Payback Period Sensitivity")
    gpu_prices = [2.50, 3.00, 3.30, 3.75, 4.50, 5.00]
    rates = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    z_payback = []
    for rate in rates:
        row = []
        for price in gpu_prices:
            m = compute_model(price, gpus_per_gw_k, capex_per_gpu_k, prepay_pct, rate/100, opex_pct,
                              yr1_ramp, ext_rev_pct, share_count, storage_pct, software_pct,
                              contract_yrs=contract_yrs, total_yrs=total_yrs)
            row.append(round(m["payback"], 1) if m["payback"] else 10.0)
        z_payback.append(row)

    fig_pb = go.Figure(go.Heatmap(
        z=z_payback,
        x=[f"${p:.2f}/hr" for p in gpu_prices],
        y=[f"{r:.0f}%" for r in rates],
        text=[[f"{v:.1f}" for v in row] for row in z_payback],
        texttemplate="%{text}yr",
        colorscale="RdYlGn_r",
        zmin=1.5, zmax=5.0,
    ))
    fig_pb.update_layout(
        height=400, margin=dict(t=30, b=20, l=20, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        xaxis=dict(title="GPU/hr Price"),
        yaxis=dict(title="Finance Rate"),
    )
    st.plotly_chart(fig_pb, use_container_width=True)

with col_return_sens:
    st.subheader(f"{contract_yrs}yr Return Sensitivity")

    z_return = []
    for rate in rates:
        row = []
        for price in gpu_prices:
            m = compute_model(price, gpus_per_gw_k, capex_per_gpu_k, prepay_pct, rate/100, opex_pct,
                              yr1_ramp, ext_rev_pct, share_count, storage_pct, software_pct,
                              contract_yrs=contract_yrs, total_yrs=total_yrs)
            row.append(round(m["return_contract"] * 100, 0))
        z_return.append(row)

    fig_ret = go.Figure(go.Heatmap(
        z=z_return,
        x=[f"${p:.2f}/hr" for p in gpu_prices],
        y=[f"{r:.0f}%" for r in rates],
        text=[[f"{v:.0f}%" for v in row] for row in z_return],
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=10, zmax=100,
    ))
    fig_ret.update_layout(
        height=400, margin=dict(t=30, b=20, l=20, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        xaxis=dict(title="GPU/hr Price"),
        yaxis=dict(title="Finance Rate"),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

# ============================================================
# ROW 4: Capacity Trajectory
# ============================================================
st.markdown("---")
st.subheader("Thesis Trajectory: GW Ramp & Revenue")

col_gw, col_rev_proj = st.columns(2)

# Thesis assumptions for capacity ramp
years_proj = ["2025", "2026E", "2027E", "2028E", "2029E", "2030E"]
gw_proj = [0.47, 1.2, 2.5, 3.5, 4.5, 5.0]
rev_per_gw = model["annual_rev"] / n_gw  # use current scenario per-GW rev

with col_gw:
    fig_gw = go.Figure(go.Bar(
        x=years_proj, y=gw_proj,
        text=[f"{v:.1f} GW" for v in gw_proj],
        textposition="outside",
        marker_color=["#58a6ff"] + ["#3fb950"] * 5,
    ))
    fig_gw.update_layout(
        title="Active GW Capacity",
        height=380, margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        yaxis=dict(gridcolor="#21262d", title="GW"),
    )
    st.plotly_chart(fig_gw, use_container_width=True)

with col_rev_proj:
    rev_proj = [gw * rev_per_gw for gw in gw_proj]
    fig_rev_p = go.Figure(go.Bar(
        x=years_proj, y=rev_proj,
        text=[f"${v:.0f}B" for v in rev_proj],
        textposition="outside",
        marker_color=["#d29922"] + ["#bc8cff"] * 5,
    ))
    fig_rev_p.update_layout(
        title=f"Revenue Projection (at ${gpu_hr:.2f}/hr)",
        height=380, margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        yaxis=dict(gridcolor="#21262d", title="$ Billions"),
    )
    st.plotly_chart(fig_rev_p, use_container_width=True)

# ============================================================
# SIDEBAR: Scenario Results
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### Scenario Results")
st.sidebar.metric("Capex", f"${model['capex']:.1f}B")
st.sidebar.metric("TCV (5yr)", f"${model['tcv']:.1f}B")
st.sidebar.metric("Amt Financed", f"${model['amt_financed']:.1f}B")
st.sidebar.metric("Annual Revenue", f"${model['annual_rev']:.1f}B")
st.sidebar.metric("Payback", f"{model['payback']:.1f} yrs" if model['payback'] else "N/A")
st.sidebar.metric(f"{contract_yrs}yr Profit", f"${model['profit_contract']:.1f}B")
st.sidebar.metric("Take Rate", f"{model['take_rate']:.1%}")

# ============================================================
# FOOTER: Detailed P&L
# ============================================================
st.markdown("---")
with st.expander("Detailed Cash Flow Table"):
    header = "| Year | Revenue | Opex | Debt Service | Net CF | CF/Share | Cumulative |"
    sep = "|------|---------|------|-------------|--------|---------|------------|"
    rows = [header, sep]
    cum = 0
    for y in model["years"]:
        cum += y["cf_per_share"]
        rows.append(
            f"| {y['label']} | ${y['revenue']:.2f}B | ${y['opex']:.2f}B | "
            f"${y['debt_service']:.2f}B | ${y['net_cf']:.2f}B | "
            f"${y['cf_per_share']:.2f} | ${cum:.2f} |"
        )
    st.markdown("\n".join(rows))

st.caption("Model: Shan's CRWV contract anatomy model")
