import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --- Config ---
st.set_page_config(page_title="Governance Dashboard", layout="wide")

st.title("🏛️ Political & Governance Monitor")
st.caption("Local Data Storage Version | ระบบจะพยากรณ์ไปจนถึงปี 2026")

# --- การดึงข้อมูลจากไฟล์ใน Repository ของตัวเอง ---
@st.cache_data
def load_data():
    file_path = 'political_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

df = load_data()

if df is not None:
    # Sidebar
    countries = sorted(df['Country'].unique())
    selected = st.sidebar.multiselect("Select Countries", countries, default=["Thailand", "Vietnam"])
    
    if selected:
        filtered = df[df['Country'].isin(selected)].copy()
        
        # Forecast Logic: พยากรณ์ล่วงหน้าถึงปี 2026
        forecast_data = []
        for c in selected:
            c_df = filtered[filtered['Country'] == c]
            model = LinearRegression().fit(c_df[['Year']], c_df['Score'])
            
            # พยากรณ์ปี 2024, 2025, 2026
            future = np.array([2024, 2025, 2026]).reshape(-1, 1)
            preds = model.predict(future)
            
            for i, yr in enumerate([2024, 2025, 2026]):
                forecast_data.append({'Country': c, 'Year': yr, 'Score': round(preds[i], 2), 'Type': 'Forecast'})
        
        filtered['Type'] = 'Actual'
        df_final = pd.concat([filtered, pd.DataFrame(forecast_data)])

        # Chart
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type", markers=True,
                      title="Trend and Forecast to 2026", template="plotly_white")
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ ระบบทำงานปกติ: ดึงข้อมูลจากฐานข้อมูลภายในสำเร็จ")
    else:
        st.info("👈 กรุณาเลือกประเทศที่ Sidebar")
else:
    st.error("❌ ไม่พบไฟล์ political_data.csv ใน Repository ของคุณ")
