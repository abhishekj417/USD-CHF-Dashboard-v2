# USD/CHF 30-Year Macro Dashboard

This is a fully automated finance dashboard that shows USD/CHF exchange rate over the past 30 years 
and its relationship with macroeconomic and market variables.

## Features
- USD/CHF exchange rate (monthly) 
- US & Swiss interest rates
- CPI (US and Switzerland)
- Gold & Brent Oil prices
- S&P 500 & MSCI World ETF
- Swiss GDP & Exports (forward-filled monthly)
- Interactive charts, heatmaps, rolling correlations, and scatter plots
- Downloadable monthly data CSV

## Deployment Instructions

1. **Create a FRED API key** (recommended) here: https://fred.stlouisfed.org/docs/api/api_key.html

2. **Upload to GitHub**:
   - Create a new repo, e.g., `usdchf-dashboard`
   - Upload all files from this folder

3. **Deploy on Streamlit Cloud**:
   - Go to https://streamlit.io/cloud
   - Click 'New App', connect GitHub, select your repo and branch
   - Set your FRED API key in Secrets (optional but recommended)

4. **View your dashboard** at the Streamlit-generated URL

