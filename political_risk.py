import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Page Setup ---
st.set_page_config(page_title="WGI Governance Explorer", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border-radius: 10px; border: 1px solid #dce1e6; background: white; padding: 15px; }
    h1 { color: #1a4e8a; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Worldwide Governance Indicators (WGI)")
st.caption("Data provided by World Bank Group | Intelligent Forecasting System")

# --- Sidebar Filters ---
st.sidebar.header("Filter Settings")

# รายชื่อดัชนี WGI 6 ด้านหลัก
indicator_map = {
    'WGI.PV': 'Political Stability and Absence of Violence',
    'WGI.CC': 'Control of Corruption',
    'WGI.VA': 'Voice and Accountability',
    'WGI.GE': 'Government Effectiveness',
    'WGI.RQ': 'Regulatory Quality',
    'WGI.RL': 'Rule of Law'
}
selected_ind_label = st.sidebar.selectbox("Select Indicator", list(indicator_map.values()))
selected_ind_code = [k for k, v in indicator_map.items() if v == selected_ind_label][0]

selected_countries = st.sidebar.multiselect(
    "Select Countries", 
    ["THA", "VNM", "IDN", "SGP", "MYS", "USA", "CHN", "JPN", "GBR", "IND"],
    default=["THA", "VNM"]
)

# ปรับช่วงปีเริ่มต้นคงที่ แต่ให้โปรแกรมหาปีสิ้นสุดเอง
start_year = st.sidebar.number_input("Start Year", 2000, 2020, 2010)
this_year = datetime.datetime.now().year

# --- Backend: ดึงข้อมูลและจัดการ Data Gap ---
@st.cache_data(ttl=86400)
def fetch_and_process(countries, ind_code, start_yr, end_yr):
    try:
        # ดึงข้อมูลย้อนหลังเผื่อไว้จนถึงปีปัจจุบัน
        df = wb.data.DataFrame(ind_code, countries, time=range(start_yr, end_yr + 1), labels=True)
        if df.empty: return None
        
        # แปลงเป็น Long Format
        df_melted = df.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Score')
        df_melted['Year'] = df_melted['Year'].str.replace('YR', '').astype(int)
        
        # ลบค่าที่เป็น NaN (ปีที่ยังไม่มีข้อมูล)
        df_clean = df_melted.dropna(subset=['Score']).sort_values(['Country', 'Year'])
        return df_clean
    except Exception as e:
        return None

# --- Main App Logic ---
if selected_countries:
    # ดึงข้อมูลย้อนหลังตั้งแต่ปีที่เลือกจนถึงปัจจุบัน (ระบบจะกรองปีที่ไม่มีข้อมูลออกเอง)
    data = fetch_and_process(selected_countries, selected_ind_code, start_year, this_year)
    
    if data is not None and not data.empty:
        # หาวันที่ล่าสุดที่มีข้อมูลจริง
        latest_available_year = data['Year'].max()
        st.success(f"✅ ดึงข้อมูลสำเร็จ: ข้อมูลล่าสุดที่มีคือปี {latest_available_year}")

        # --- Forecasting 3 Years from Latest Available Data ---
        forecast_results = []
        for country in selected_countries:
            c_data = data[data['Country'] == country]
            if len(c_data) > 2: # ต้องมีข้อมูลอย่างน้อย 3 ปีถึงจะพยากรณ์ได้
                X = c_data['Year'].values.reshape(-1, 1)
                y = c_data['Score'].values
                model = LinearRegression().fit(X, y)
                
                # พยากรณ์ล่วงหน้า 3 ปีจากปีล่าสุดที่มีข้อมูล
                future_years = np.array([latest_available_year + 1, latest_available_year + 2, latest_available_year + 3]).reshape(-1, 1)
                preds = model.predict(future_years)
                
                for idx, yr in enumerate(future_years.flatten()):
                    forecast_results.append({
                        'Country': country,
                        'Year': int(yr),
                        'Score': round(preds[idx], 3),
                        'Type': 'Forecast'
                    })

        data['Type'] = 'Actual'
        df_forecast = pd.DataFrame(forecast_results)
        df_final = pd.concat([data, df_forecast]).reset_index(drop=True)

        # --- Display Results ---
        # 1. Metrics
        m_cols = st.columns(len(selected_countries))
        for idx, country in enumerate(selected_countries):
            c_latest = data[data['Country'] == country].iloc[-1]
            with m_cols[idx]:
                st.metric(f"{country} Score ({latest_available_year})", f"{c_latest['Score']:.2f}")

        # 2. Chart
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type",
                      title=f"Trend & Prediction: {selected_ind_label}",
                      markers=True, template="plotly_white", height=500)
        fig.update_layout(yaxis_range=[-2.5, 2.5]) # มาตรฐาน WGI คือ -2.5 ถึง 2.5
        st.plotly_chart(fig, use_container_width=True)

        # 3. Data Table
        with st.expander("🔎 ดูตารางข้อมูลทั้งหมด"):
            st.dataframe(df_final.pivot(index='Year', columns='Country', values='Score').sort_index(ascending=False), use_container_width=True)
            
    else:
        st.error("❌ ไม่พบข้อมูลในฐานข้อมูล World Bank สำหรับกลุ่มประเทศที่เลือก โปรดลองเปลี่ยน Start Year หรือเพิ่มประเทศ")
else:
    st.info("👈 กรุณาเลือกประเทศที่แถบเมนูด้านซ้าย")
