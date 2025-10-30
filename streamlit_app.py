# streamlit_app.py
import os
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_datareader.data as web
from pandas_datareader import wb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="USD/CHF 30yr Macro Dashboard")

# -------------------------
# CONFIG: change if desired
# -------------------------
FRED_SERIES = {
   "USDCHF": "DEXSZUS",
   "FEDFUNDS": "FEDFUNDS",
   "SNB_RATE": "IR3TTS01CHM156N",
   "CPI_US": "CPIAUCSL",
   "CPI_CH": "CPCHOM01CHM657N",
   "GOLD": "GOLDAMGBD228NLBM",
   "BRENT": "DCOILBRENTEU",
}

YF_TICKERS = {
   "S&P 500": "^GSPC",
   "MSCI World (ETF URTH)": "URTH",
}

WB_INDICATORS = {
   "Swiss_GDP_USD": "NY.GDP.MKTP.CD",
   "Swiss_Exports_USD": "NE.EXP.GNFS.CD"
}

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def get_fred_series(series_id, start, end, api_key=None):
   # pandas_datareader will pick up FRED_API_KEY from environ if provided
   try:
       s = pdr.DataReader(series_id, "fred", start, end)
       s.columns = [series_id]
       return s
   except Exception as e:
       st.error(f"Error fetching {series_id} from FRED: {e}")
       return pd.DataFrame(index=pd.date_range(start=start, end=end, freq="D"))

@st.cache_data(show_spinner=False)
def get_yf_series(ticker, start, end):
   try:
       df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
       if df.empty:
           return pd.DataFrame(index=pd.date_range(start=start, end=end, freq="D"))
       s = df["Close"].rename(ticker)
       return s.to_frame()
   except Exception as e:
       st.error(f"Error fetching {ticker} from Yahoo: {e}")
       return pd.DataFrame(index=pd.date_range(start=start, end=end, freq="D"))

@st.cache_data(show_spinner=False)
def get_worldbank(country_code, indicators, start_year, end_year):
   try:
       wb_df = wb.download(indicator=list(indicators.values()), country=country_code, start=start_year, end=end_year)
       # wb returns a DataFrame with columns as indicator codes
       # rename columns to keys in indicators
       rename_map = {v: k for k, v in indicators.items()}
       wb_df = wb_df.rename(columns=rename_map)
       wb_df.index = pd.to_datetime(wb_df.index.astype(int).astype(str) + "-12-31")
       return wb_df
   except Exception as e:
       st.warning(f"World Bank data fetch warning: {e}")
       return pd.DataFrame()

def daily_to_monthly_mean(df):
   # Accepts DataFrame or Series with datetime index -> returns monthly mean (end-of-month)
   if df.empty:
       return df
   return df.resample("M").mean()

def ensure_datetime_index(df):
   if not isinstance(df.index, pd.DatetimeIndex):
       df.index = pd.to_datetime(df.index)
   return df

# -------------------------
# Date range
# -------------------------
today = pd.Timestamp.today().normalize()
start_date = (today - pd.DateOffset(years=30)).replace(day=1)  # start at beginning of month 30 years ago
end_date = today

st.title("USD/CHF — 30-Year Monthly Dashboard (Automated)")

with st.sidebar:
   st.markdown("## Settings")
   fred_key = st.text_input("FRED API Key (optional; set as env FRED_API_KEY if preferred)", value=os.getenv("FRED_API_KEY") or "")
   start_input = st.date_input("Start date", value=start_date.date(), min_value=(today - pd.DateOffset(years=50)).date(), max_value=today.date())
   end_input = st.date_input("End date", value=end_date.date(), min_value=start_input, max_value=today.date())
   months_rolling = st.slider("Rolling window for correlations (months)", min_value=1, max_value=36, value=12)
   selected_vars = st.multiselect("Variables to display", options=list(FRED_SERIES.keys()) + list(YF_TICKERS.keys()) + list(WB_INDICATORS.keys()), default=["USDCHF","FEDFUNDS","GOLD","BRENT","S&P 500"])
   st.caption("Notes: You can set FRED_API_KEY in environment or paste above. World Bank data is annual and forward-filled to monthly.")

# unify start/end to first/last day
start = pd.to_datetime(start_input)
end = pd.to_datetime(end_input) + pd.offsets.MonthEnd(0)

# -------------------------
# Fetch FRED series
# -------------------------
fred_dfs = []
for name, series_id in FRED_SERIES.items():
   df = get_fred_series(series_id, start - pd.DateOffset(days=10), end + pd.DateOffset(days=10), api_key=fred_key or None)
   df = ensure_datetime_index(df)
   df = daily_to_monthly_mean(df)
   df = df.rename(columns={series_id: name})
   fred_dfs.append(df)

# -------------------------
# Fetch Yahoo tickers
# -------------------------
yf_dfs = []
for nice_name, ticker in YF_TICKERS.items():
   s = get_yf_series(ticker, start - pd.DateOffset(days=60), end + pd.DateOffset(days=1))
   s = ensure_datetime_index(s)
   s = daily_to_monthly_mean(s.rename(columns={ticker: nice_name}))
   yf_dfs.append(s)

# -------------------------
# Fetch World Bank indicators (annual) and expand to monthly
# -------------------------
wb_df = get_worldbank("CHE", WB_INDICATORS, start.year, end.year)
if not wb_df.empty:
   # rename to friendly keys already applied
   # forward-fill annual values monthly
   wb_monthly = wb_df.resample("M").ffill()
   # keep only date range
   wb_monthly = wb_monthly[(wb_monthly.index >= start) & (wb_monthly.index <= end)]
else:
   wb_monthly = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="M"))

# -------------------------
# Merge everything
# -------------------------
sources = fred_dfs + yf_dfs + [wb_monthly]
if len(sources) == 0:
   st.error("No data sources available.")
   st.stop()

merged = pd.concat(sources, axis=1)
merged = merged[(merged.index >= start) & (merged.index <= end)]
# interpolate/ffill small gaps
merged = merged.sort_index().ffill().interpolate(limit=3)

if "USDCHF" not in merged.columns:
   st.error("USDCHF series missing. Check FRED access or series code.")
   st.stop()

# compute returns and pct changes
merged_pct = merged.pct_change().replace([np.inf, -np.inf], np.nan)

# compute rolling correlations
rolling_corrs = {}
for col in merged.columns:
   if col == "USDCHF":
       continue
   try:
       rc = merged_pct["USDCHF"].rolling(window=months_rolling).corr(merged_pct[col])
       rolling_corrs[col] = rc
   except Exception:
       rolling_corrs[col] = pd.Series(index=merged_pct.index)

# -------------------------
# Layout: top stats
# -------------------------
col1, col2, col3 = st.columns(3)
with col1:
   last = merged["USDCHF"].dropna().iloc[-1]
   first = merged["USDCHF"].dropna().iloc[0]
   total_pct = (last - first) / first * 100
   st.metric("USD/CHF (last)", f"{last:.4f}", delta=f"{total_pct:.2f}% since start")
with col2:
   last_mom = merged_pct["USDCHF"].iloc[-1]
   st.metric("USD/CHF MoM % (last month)", f"{last_mom*100:.2f}%")
with col3:
   st.metric("Data range", f"{merged.index.min().date()} → {merged.index.max().date()}")

# -------------------------
# Main charts
# -------------------------
st.markdown("## Time series")
plot_cols = ["USDCHF"] + [c for c in merged.columns if c in selected_vars and c != "USDCHF"]
ts_df = merged[plot_cols].dropna(how="all")

fig = go.Figure()
for c in ts_df.columns:
   fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[c], name=c, mode="lines"))
fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Heatmap: monthly returns by year
# -------------------------
st.markdown("## Monthly returns heatmap (USD/CHF)")
ret = merged_pct["USDCHF"].dropna()
heat = ret.to_frame("ret")
heat["year"] = heat.index.year
heat["month"] = heat.index.month
pivot = heat.pivot_table(index="year", columns="month", values="ret", aggfunc="mean").sort_index(ascending=False)

fig2 = px.imshow(pivot * 100,
                labels=dict(x="Month", y="Year", color="Monthly %"),
                x=[1,2,3,4,5,6,7,8,9,10,11,12],
                y=pivot.index.astype(str))
fig2.update_layout(height=600)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Correlation matrix (full period)
# -------------------------
st.markdown("## Full-period correlations (variables)")
corr_df = merged_pct.dropna().corr()
fig3 = px.imshow(corr_df, text_auto=True)
fig3.update_layout(height=600)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Rolling correlations with USD/CHF
# -------------------------
st.markdown(f"## Rolling correlations with USD/CHF (window = {months_rolling}m)")
rc_df = pd.DataFrame(rolling_corrs)
rc_df = rc_df.dropna(how="all")
if not rc_df.empty:
   fig4 = go.Figure()
   for col in rc_df.columns:
       fig4.add_trace(go.Scatter(x=rc_df.index, y=rc_df[col], name=col))
   fig4.add_hline(y=0, line_dash="dash", line_color="gray")
   fig4.update_layout(height=420)
   st.plotly_chart(fig4, use_container_width=True)
else:
   st.info("Not enough data to compute rolling correlations.")

# -------------------------
# Scatter: pick variable vs USD/CHF
# -------------------------
st.markdown("## Scatter: USD/CHF vs variable (monthly)")
var_choice = st.selectbox("Choose variable for scatter", options=[c for c in merged.columns if c != "USDCHF"])
scatter_df = pd.concat([merged["USDCHF"], merged[var_choice]], axis=1).dropna()
if not scatter_df.empty:
   fig5 = px.scatter(scatter_df, x=var_choice, y="USDCHF", trendline="ols", labels={var_choice: var_choice})
   fig5.update_layout(height=450)
   st.plotly_chart(fig5, use_container_width=True)
else:
   st.warning("No overlapping data for scatter.")

# -------------------------
# Data & download
# -------------------------
st.markdown("## Data table (monthly)")
show_df = merged.copy()
show_df.index = show_df.index.to_period("M").to_timestamp("M")
st.dataframe(show_df.tail(200))

@st.cache_data
def to_csv(df):
   return df.to_csv().encode("utf-8")

csv = to_csv(show_df)
st.download_button("Download merged data (CSV)", data=csv, file_name="usdchf_merged_monthly.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with FRED, Yahoo Finance (yfinance), World Bank. Set FRED_API_KEY in your environment for more reliable FRED requests.")
