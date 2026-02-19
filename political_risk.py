import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# --- Config ---
st.set_page_config(page_title="Global Political Analytics", layout="wide")

st.markdown("""
    <style>
    .stMetric { background: white; padding: 20px; border-radius: 10px; border: 1px solid #e1e8ed; }
    h1 { color: #0071bc; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Global Governance & Democracy Monitor")
st.caption(f"Data Source: Our World in Data & EIU | Auto-updated: {datetime.datetime.now().year}")

# --- ฐานข้อมูลสำรองที่เสถียรที่สุด (GitHub Direct Link) ---
# ไฟล์นี้รวบรวมดัชนีการเมืองจากทั่วโลก อัปเดตอัตโนมัติทุกปี
DATA_URL = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Democracy%20Index%20(EIU)/Democracy%20Index%20(EIU).csv"

@st.cache_data(ttl=86400)
def load_live_data():
    try:
        # ดึงไฟล์ CSV โดยตรง (เร็วกว่าและเสถียรกว่า API)
        df = pd.read_csv(DATA_URL)
        df.columns = ['Country', 'Code', 'Year', 'Score']
        return df
    except:
        return None

df = load_live_data()

# --- Sidebar ---
st.sidebar.header("Filter Settings")
if df is not None:
    all_countries = sorted(df['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries", 
        all_countries, 
        default=["Thailand", "Vietnam", "Malaysia"]
    )

    # --- Processing ---
    if selected_countries:
        filtered_df = df[df['Country'].isin(selected_countries)].copy()
        latest_actual_year = int(filtered_df['Year'].max())
        
        # Forecast Logic: พยากรณ์ล่วงหน้า 3 ปีจากปีล่าสุดที่มีข้อมูลจริง
        forecast_list = []
        future_years = [latest_actual_year + 1, latest_actual_year + 2, latest_actual_year + 3]
        
        for country in selected_countries:
            c_data = filtered_df[filtered_df['Country'] == country].sort_values('Year')
            if len(c_data) >= 3:
                model = LinearRegression().fit(c_data[['Year']], c_data['Score'])
                preds = model.predict(np.array(future_years).reshape(-1, 1))
                
                for i, yr in enumerate(future_years):
                    forecast_list.append({
                        'Country': country, 'Year': yr, 
                        'Score': round(max(0, min(10, preds[i])), 2), 'Type': 'Forecast'
                    })

        filtered_df['Type'] = 'Actual'
        df_forecast = pd.DataFrame(forecast_list)
        df_final = pd.concat([filtered_df, df_forecast]).reset_index(drop=True)

        # --- Visuals ---
        st.success(f"📊 ข้อมูลจริงล่าสุดอัปเดตถึงปี: {latest_actual_year}")
        
        # 1. Metrics
        m_cols = st.columns(len(selected_countries))
        for idx, country in enumerate(selected_countries):
            c_latest = filtered_df[filtered_df['Country'] == country].iloc[-1]
            with m_cols[idx]:
                st.metric(f"{country} ({latest_actual_year})", f"{c_latest['Score']:.2f}")

        # 2. Chart
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type",
                      markers=True, title="Political Stability & Democracy Index (Trend & Forecast)",
                      template="plotly_white", height=500)
        fig.update_layout(yaxis_range=[0, 10], hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. Table
        with st.expander("🔎 View Raw Data & Predictions"):
            st.dataframe(df_final.sort_values(['Country', 'Year'], ascending=[True, False]), use_container_width=True)
    else:
        st.info("👈 Please select countries in the sidebar.")
else:
    st.error("❌ ไม่สามารถดึงข้อมูลได้เนื่องจากปัญหาการเชื่อมต่อ Server โปรดลองอีกครั้งในภายหลัง")
