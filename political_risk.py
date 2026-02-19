import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Config ---
st.set_page_config(page_title="Dynamic Political Monitor", layout="wide")

st.title("🏛️ Worldwide Governance Indicators (WGI)")
st.caption(f"ระบบดึงข้อมูลอัตโนมัติจาก World Bank | อัปเดตล่าสุด: {datetime.datetime.now().year}")

# --- Sidebar ---
st.sidebar.header("Settings")
selected_countries = st.sidebar.multiselect(
    "Select Countries", 
    ["THA", "VNM", "MYS", "SGP", "IDN", "PHL", "USA", "CHN"], 
    default=["THA", "VNM"]
)

# --- Backend: Auto-Update Fetching ---
@st.cache_data(ttl=86400) # ให้รีเฟรชข้อมูลทุก 24 ชั่วโมง
def fetch_auto_data(countries):
    try:
        this_year = datetime.datetime.now().year
        # ดึงข้อมูลจากปี 2010 จนถึงปีปัจจุบันของคอมพิวเตอร์
        df = wb.data.DataFrame('WGI.PV', countries, time=range(2010, this_year + 1), labels=True)
        if df is None or df.empty: return None
        
        # ปรับโครงสร้างและกรองข้อมูลที่ยังไม่เกิดขึ้น (NaN)
        df_long = df.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Score')
        df_long['Year'] = df_long['Year'].str.replace('YR', '').astype(int)
        return df_long.dropna(subset=['Score']).sort_values(['Country', 'Year'])
    except:
        return None

# --- Main App ---
if selected_countries:
    data = fetch_auto_data(selected_countries)
    
    if data is not None and not data.empty:
        latest_year = data['Year'].max()
        st.success(f"✅ ข้อมูลจริงล่าสุดในฐานข้อมูลคือปี: {latest_year}")

        # Forecast AI: พยากรณ์ต่อจากปีล่าสุดไปอีก 3 ปี
        forecast_results = []
        for c in selected_countries:
            c_data = data[data['Country'] == c]
            if len(c_data) >= 3:
                model = LinearRegression().fit(c_data[['Year']], c_data['Score'])
                # ทำนายปีถัดไป 3 ปี
                future = np.array([latest_year+1, latest_year+2, latest_year+3]).reshape(-1, 1)
                preds = model.predict(future)
                for i, yr in enumerate(future.flatten()):
                    forecast_results.append({'Country': c, 'Year': yr, 'Score': preds[i], 'Type': 'Forecast'})

        data['Type'] = 'Actual'
        df_final = pd.concat([data, pd.DataFrame(forecast_results)])

        # Chart
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type", markers=True,
                      title=f"Political Stability Trend (Updated up to {latest_year})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("ไม่สามารถเชื่อมต่อฐานข้อมูล World Bank ได้ในขณะนี้")
