# Black-Scholes Options Pricer

An interactive Streamlit app for pricing options using the Black-Scholes model. Includes call/put valuation, all five Greeks, P&L heatmaps, payoff diagrams, and sensitivity charts.

## Features

- Real-time call/put pricing
- Greeks: Delta, Gamma, Theta, Vega, Rho
- P&L heatmaps across spot price and volatility
- Payoff diagrams (at expiry vs current value)
- Sensitivity charts for all Greeks
- Fully interactive — every input updates the whole dashboard

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Deploy to Streamlit Community Cloud (free)

1. Push the project to a public GitHub repo. Required files at the repo root:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml` (optional, for theming)

2. Go to https://share.streamlit.io and sign in with GitHub.

3. Click **New app**, pick your repo, set the main file to `app.py`, and deploy.

4. You'll get a URL like `https://your-app-name.streamlit.app/`. That's the link to put on your portfolio.

## Embed on your portfolio

**Option A — Just link to it (recommended).** Add a project card to your portfolio that links to the Streamlit URL. Simplest and most reliable.

**Option B — Embed as an iframe.** Streamlit apps can be embedded, but the host page must allow iframes from `streamlit.app`. Use the `?embed=true` query param to hide the Streamlit menu:

```html
<iframe
  src="https://your-app-name.streamlit.app/?embed=true"
  height="900"
  width="100%"
  style="border:none;"
></iframe>
```

Note: free Streamlit Cloud apps go to sleep after inactivity and take a few seconds to wake up. For zero cold-start, consider Hugging Face Spaces or a paid host like Render/Railway.

## Project structure

```
blackscholes/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme + server config
└── README.md
```

## Disclaimer

This is an educational tool. Black-Scholes assumes constant volatility, no dividends, European-style exercise, and frictionless markets — none of which fully hold in real markets. Do not use for live trading decisions.
