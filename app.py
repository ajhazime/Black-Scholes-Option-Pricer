import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from math import log, sqrt, exp

#
# basic config
# 
st.set_page_config(
    page_title="Black-Scholes Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.header("Made by: Ali Hazime Jr. ")
st.sidebar.markdown("[![LinkedIn](https://www.linkedin.com/favicon.ico)](https://linkedin.com/in/ali-hazime-jr)")

st.sidebar.markdown("---")

if True: #LOL
    PLOTLY_TEMPLATE = "plotly_dark"
    APP_BG          = "#0e1117"
    SECONDARY_BG    = "#161b22"
    TEXT_COLOR      = "#fafafa"
    BORDER_COLOR    = "#30363d"
    TAB_BAR_BG      = "rgba(255,255,255,0.06)"
    TAB_HOVER_BG    = "rgba(255,255,255,0.12)"
    TAB_INACTIVE    = "#cbd5e0"


#CSS styling for streamlit hosting 
st.markdown(
    f"""
    <style>
        /* App background & text  */
        .stApp {{
            background-color: {APP_BG};
            color: {TEXT_COLOR};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {SECONDARY_BG};
            border-right: 1px solid {BORDER_COLOR};
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp p, .stApp label, .stApp .stMarkdown {{
            color: {TEXT_COLOR};
        }}

        /* Price cards */
        .metric-card {{
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        }}
        .call-card  {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .put-card   {{ background: linear-gradient(135deg, #c31432 0%, #ee0979 100%); }}
        .price-label{{ font-size: 0.85rem; opacity: 0.9; letter-spacing: 0.05em; }}
        .price-value{{ font-size: 2.25rem; font-weight: 700; margin-top: 0.25rem; }}

        /* TABS */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: {TAB_BAR_BG};
            padding: 8px;
            border-radius: 14px;
            margin-bottom: 1.25rem;
            border: 1px solid {BORDER_COLOR};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 54px;
            padding: 0 28px;
            font-size: 1.05rem;
            font-weight: 600;
            border-radius: 10px;
            background: transparent;
            color: {TAB_INACTIVE};
            transition: all 0.2s ease;
            border: none;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: {TAB_HOVER_BG};
            color: {TEXT_COLOR};
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            box-shadow: 0 4px 14px rgba(102, 126, 234, 0.45);
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
        .stTabs [data-baseweb="tab-border"]    {{ display: none; }}
        .stTabs [data-baseweb="tab-panel"]     {{ padding-top: 0.5rem; }}

        /*  Summary table */
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            background: {SECONDARY_BG};
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {BORDER_COLOR};
            margin: 0.5rem 0 1.5rem 0;
            color: {TEXT_COLOR};
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        }}
        .summary-table thead th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 14px 18px;
            text-align: left;
            font-weight: 700;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .summary-table thead th.num {{ text-align: right; }}
        .summary-table td {{
            padding: 11px 18px;
            border-bottom: 1px solid {BORDER_COLOR};
            color: {TEXT_COLOR};
            font-size: 0.95rem;
        }}
        .summary-table tbody tr:last-child td {{ border-bottom: none; }}
        .summary-table tr.section td {{
            background: rgba(102, 126, 234, 0.12);
            font-weight: 700;
            color: #818cf8 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.8rem;
            padding: 10px 18px;
        }}
        .summary-table .num    {{ text-align: right; font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, Consolas, monospace; }}
        .summary-table .center {{ text-align: center; font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, Consolas, monospace; }}
        .summary-table .strong {{ font-weight: 700; font-size: 1.05rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

#
# Black-Scholes functions
#

#_d1(Spot, Strike, Expiration, riskfree rate, sigma)
def _d1(S, K, T, r, sigma): 
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * sqrt(T)


#price for a European option.
def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        intrinsic = (S - K) if option_type == "call" else (K - S)
        return max(intrinsic, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

#Returns delta, gamma, theta (per day), vega (per 1%), rho (per 1%).
def greeks(S, K, T, r, sigma, option_type="call"):
    
    if T <= 0 or sigma <= 0:
        return dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * sqrt(T))
    vega = S * pdf_d1 * sqrt(T) / 100.0

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * sqrt(T))
                 - r * K * exp(-r * T) * norm.cdf(d2)) / 365.0
        rho = K * T * exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-S * pdf_d1 * sigma / (2 * sqrt(T))
                 + r * K * exp(-r * T) * norm.cdf(-d2)) / 365.0
        rho = -K * T * exp(-r * T) * norm.cdf(-d2) / 100.0

    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


#
# Sidebar inputs
#
st.sidebar.header("Model Parameters")

S = st.sidebar.number_input("Spot Price", min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Expiry (years)", min_value=0.001, value=1.0, step=0.05, format="%.3f")
r = st.sidebar.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.001, max_value=3.0, value=0.20, step=0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("P&L Reference Prices")
call_purchase = st.sidebar.number_input("Call Purchase Price", min_value=0.0, value=0.0, step=0.1,
                                        help="If you bought the call at this price, the heatmap shows P&L.")
put_purchase = st.sidebar.number_input("Put Purchase Price", min_value=0.0, value=0.0, step=0.1,
                                       help="If you bought the put at this price, the heatmap shows P&L.")

st.sidebar.markdown("---")
st.sidebar.subheader("Heatmap Range")
spot_min = st.sidebar.slider("Min Spot", min_value=S * 0.50, max_value=S * 0.99, value=S * 0.80)
spot_max = st.sidebar.slider("Max Spot", min_value=S * 1.01, max_value=S * 1.50, value=S * 1.20)
vol_min = st.sidebar.slider("Min Vol", min_value=0.01, max_value=float(sigma), value=max(0.05, sigma * 0.5))
vol_max = st.sidebar.slider("Max Vol", min_value=float(sigma), max_value=float(sigma * 2), value=min(1.0, sigma * 1.5))



# 
# Main area
# 

st.title("Black-Scholes Options Pricer")

call_price = bs_price(S, K, T, r, sigma, "call")
put_price = bs_price(S, K, T, r, sigma, "put")
call_g = greeks(S, K, T, r, sigma, "call")
put_g = greeks(S, K, T, r, sigma, "put")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"<div class='metric-card call-card'>"
        f"<div class='price-label'>CALL OPTION VALUE</div>"
        f"<div class='price-value'>${call_price:,.4f}</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-card put-card'>"
        f"<div class='price-label'>PUT OPTION VALUE</div>"
        f"<div class='price-value'>${put_price:,.4f}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("")

st.subheader("Model Parameters, Pricing, and Greeks")

summary_html = f"""
<table class="summary-table">
    <thead>
        <tr>
        </tr>
    </thead>
    <tbody>
        <tr class="section"><td colspan="3">Model Parameters</td></tr>
        <tr><td>Spot Price</td><td colspan="2" class="center">${S:,.2f}</td></tr>
        <tr><td>Strike Price</td><td colspan="2" class="center">${K:,.2f}</td></tr>
        <tr><td>Time to Expiry</td><td colspan="2" class="center">{T:.3f} years</td></tr>
        <tr><td>Risk-Free Rate</td><td colspan="2" class="center">{r*100:.2f}%</td></tr>
        <tr><td>Volatility (σ)</td><td colspan="2" class="center">{sigma*100:.2f}%</td></tr>
        <tr class="section"><td colspan="3">Pricing & Greeks</td></tr>
        <tr><th>Metric</th><th class="num">Call</th><th class="num">Put</th></tr>
        <tr><td>Option Price</td><td class="num strong">${call_price:.4f}</td><td class="num strong">${put_price:.4f}</td></tr>
        <tr><td>Delta (Δ)</td><td class="num">{call_g['delta']:.4f}</td><td class="num">{put_g['delta']:.4f}</td></tr>
        <tr><td>Gamma (Γ)</td><td class="num">{call_g['gamma']:.4f}</td><td class="num">{put_g['gamma']:.4f}</td></tr>
        <tr><td>Theta (Θ) — per day</td><td class="num">{call_g['theta']:.4f}</td><td class="num">{put_g['theta']:.4f}</td></tr>
        <tr><td>Vega (ν) — per 1% vol</td><td class="num">{call_g['vega']:.4f}</td><td class="num">{put_g['vega']:.4f}</td></tr>
        <tr><td>Rho (ρ) — per 1% rate</td><td class="num">{call_g['rho']:.4f}</td><td class="num">{put_g['rho']:.4f}</td></tr>
    </tbody>
</table>
"""
st.markdown(summary_html, unsafe_allow_html=True)
st.markdown("---")

# 
# Tabs for Heatmap, Payoffs, and Greeks
# 

tab1, tab2, tab3 = st.tabs(["P&L Heatmaps", "Payoff Diagrams", "Sensitivity Charts"])

#  Heatmaps 
with tab1:
    st.markdown("### P&L across spot price and volatility")
   
    grid_size = 10
    spot_range = np.linspace(spot_min, spot_max, grid_size)
    vol_range = np.linspace(vol_min, vol_max, grid_size)

    call_pnl = np.zeros((len(vol_range), len(spot_range)))
    put_pnl = np.zeros((len(vol_range), len(spot_range)))
    for i, v in enumerate(vol_range):
        for j, s in enumerate(spot_range):
            call_pnl[i, j] = bs_price(s, K, T, r, v, "call") - call_purchase
            put_pnl[i, j] = bs_price(s, K, T, r, v, "put") - put_purchase

    x_labels = [f"{s:.2f}" for s in spot_range]
    y_labels = [f"{v:.2f}" for v in vol_range]

    hc1, hc2 = st.columns(2)
    with hc1:
        fig = go.Figure(data=go.Heatmap(
            z=call_pnl, x=x_labels, y=y_labels, colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="P&L"),
            text=np.round(call_pnl, 2), texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(title="Call Option P&L", xaxis_title="Spot Price",
                          yaxis_title="Volatility", height=500, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    with hc2:
        fig = go.Figure(data=go.Heatmap(
            z=put_pnl, x=x_labels, y=y_labels, colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="P&L"),
            text=np.round(put_pnl, 2), texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(title="Put Option P&L", xaxis_title="Spot Price",
                          yaxis_title="Volatility", height=500, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

# Payoff diagrams
with tab2:
    st.markdown("### Payoff at expiry vs current option value")
    st.caption("*If purchase prices are set, an additional P&L line is shown.")

    spot_payoff = np.linspace(K * 0.5, K * 1.5, 200)
    call_payoff_exp = np.maximum(spot_payoff - K, 0)
    put_payoff_exp = np.maximum(K - spot_payoff, 0)
    call_value_now = np.array([bs_price(s, K, T, r, sigma, "call") for s in spot_payoff])
    put_value_now = np.array([bs_price(s, K, T, r, sigma, "put") for s in spot_payoff])

    pc1, pc2 = st.columns(2)
    with pc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_payoff, y=call_payoff_exp, name="Payoff at Expiry", line=dict(color="#11998e", width=3)))
        fig.add_trace(go.Scatter(x=spot_payoff, y=call_value_now, name="Current Value", line=dict(color="#667eea", dash="dash", width=2)))
        if call_purchase > 0:
            fig.add_trace(go.Scatter(x=spot_payoff, y=call_payoff_exp - call_purchase,
                                     name="P&L at Expiry", line=dict(color="#ee0979", width=2)))
        fig.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.update_layout(title="Call Payoff", xaxis_title="Spot Price at Expiry",
                          yaxis_title="Payoff / Value", height=460, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    with pc2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_payoff, y=put_payoff_exp,
                                 name="Payoff at Expiry", line=dict(color="#c31432", width=3)))
        fig.add_trace(go.Scatter(x=spot_payoff, y=put_value_now,
                                 name="Current Value", line=dict(color="#667eea", dash="dash", width=2)))
        if put_purchase > 0:
            fig.add_trace(go.Scatter(x=spot_payoff, y=put_payoff_exp - put_purchase,
                                     name="P&L at Expiry", line=dict(color="#11998e", width=2)))
        fig.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.update_layout(title="Put Payoff", xaxis_title="Spot Price at Expiry",
                          yaxis_title="Payoff / Value", height=460, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

# Sensitivity 
with tab3:
    st.markdown("### Greeks and price as a function of spot")

    sens_spot = np.linspace(K * 0.5, K * 1.5, 100)
    cg = [greeks(s, K, T, r, sigma, "call") for s in sens_spot]
    pg = [greeks(s, K, T, r, sigma, "put") for s in sens_spot]

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("Delta", "Gamma", "Theta", "Vega", "Rho", "Option Price"))

    fig.add_trace(go.Scatter(x=sens_spot, y=[g["delta"] for g in cg], name="Call", line=dict(color="#11998e")), 1, 1)
    fig.add_trace(go.Scatter(x=sens_spot, y=[g["delta"] for g in pg], name="Put",  line=dict(color="#c31432")), 1, 1)
    fig.add_trace(go.Scatter(x=sens_spot, y=[g["gamma"] for g in cg], line=dict(color="#667eea"), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=sens_spot, y=[g["theta"] for g in cg], line=dict(color="#11998e"), showlegend=False), 1, 3)
    fig.add_trace(go.Scatter(x=sens_spot, y=[g["theta"] for g in pg], line=dict(color="#c31432"), showlegend=False), 1, 3)
    fig.add_trace(go.Scatter(x=sens_spot, y=[g["vega"]  for g in cg], line=dict(color="#667eea"), showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=sens_spot, y=[g["rho"]   for g in cg], line=dict(color="#11998e"), showlegend=False), 2, 2)
    fig.add_trace(go.Scatter(x=sens_spot, y=[g["rho"]   for g in pg], line=dict(color="#c31432"), showlegend=False), 2, 2)
    fig.add_trace(go.Scatter(x=sens_spot, y=[bs_price(s, K, T, r, sigma, "call") for s in sens_spot],
                             line=dict(color="#11998e"), showlegend=False), 2, 3)
    fig.add_trace(go.Scatter(x=sens_spot, y=[bs_price(s, K, T, r, sigma, "put") for s in sens_spot],
                             line=dict(color="#c31432"), showlegend=False), 2, 3)

    for row in (1, 2):
        for col in (1, 2, 3):
            fig.update_xaxes(title_text="Spot", row=row, col=col)
    fig.update_layout(height=720, title_text="Greeks & Price vs Spot Price", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Option price vs time to expiry")
    time_range = np.linspace(0.01, max(T * 2, 1.0), 60)
    call_t = [bs_price(S, K, t, r, sigma, "call") for t in time_range]
    put_t = [bs_price(S, K, t, r, sigma, "put") for t in time_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_range, y=call_t, name="Call", line=dict(color="#11998e", width=3)))
    fig.add_trace(go.Scatter(x=time_range, y=put_t, name="Put",  line=dict(color="#c31432", width=3)))
    fig.add_vline(x=T, line_dash="dot", line_color="gray", annotation_text="Current T")
    fig.update_layout(title="Time-to-Expiry Sensitivity",
                      xaxis_title="Time to Expiry (years)",
                      yaxis_title="Option Price",
                      height=420, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

#
# Expanded Footer
# 
st.markdown("---")
with st.expander(" About the Black-Scholes model"):
    st.markdown(
        r"""
        The Black-Scholes formula prices a European option on a non-dividend-paying asset:

        $$C = S \, N(d_1) - K e^{-rT} N(d_2)$$
        $$P = K e^{-rT} N(-d_2) - S \, N(-d_1)$$

        where:

        $$d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

        and $N(\cdot)$ is the standard normal CDF.

        **Assumptions**
        - European-style exercise
        - No dividends
        - Constant volatility and risk-free rate
        - Log-normal asset returns
        - Frictionless markets
        """
    )
    st.caption("**Not financial advice** · For educational project purposes only.")