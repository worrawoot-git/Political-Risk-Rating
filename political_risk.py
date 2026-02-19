import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="WGI Governance Analytics", layout="wide")

# ปรับ CSS ให้สะอาดตาแบบ Dashboard สากล
st.markdown("""
    <style>
    .stMetric { background: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e1e8ed; }
    h1 { color: #0071bc; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Worldwide Governance Indicators (WGI)")
st.caption("ระบบวิเคราะห์ข้อมูลธรรมาภิบาลและความเสี่ยงทางการเมือง (ฉบับปรับปรุงการเชื่อมต่อ)")

# --- Sidebar ---
st.sidebar.header("⚙️ ตัวเลือกข้อมูล")

indicator_map = {
    'WGI.PV': 'Political Stability',
    'WGI.CC': 'Control of Corruption',
    'WGI.VA': 'Voice and Accountability',
    'WGI.GE': 'Government Effectiveness',
    'WGI.RQ': 'Regulatory Quality',
    'WGI.RL': 'Rule of Law'
}
selected_ind_label = st.sidebar.selectbox("เลือกตัวชี้วัด", list(indicator_map.values()))
selected_ind_code = [k for k, v in indicator_map.items() if v == selected_ind_label][0]

selected_countries = st.sidebar.multiselect(
    "เลือกประเทศ", 
    ["THA", "VNM", "IDN", "SGP", "MYS", "PHL", "USA", "CHN", "JPN", "GBR", "IND", "KOR"],
    default=["THA", "VNM"]
)

# ให้เลือกเฉพาะปีเริ่ม ส่วนปีจบจะถูกคำนวณอัตโนมัติจากฐานข้อมูลจริง
start_year = st.sidebar.slider("เลือกปีเริ่มต้นฐานข้อมูล", 2000, 2015, 2010)

# --- Backend: ดึงข้อมูลแบบปลอดภัย (Safe Fetch) ---
@st.cache_data(ttl=86400)
def fetch_wgi_safe(countries, ind_code, start_yr):
    try:
        # ดึงข้อมูลโดยไม่ระบุปีจบ (API จะส่งปีล่าสุดที่มีจริงมาให้เองโดยไม่ Error)
        df = wb.data.DataFrame(ind_code, countries, time=range(start_yr, 2025), labels=True)
        if df is None or df.empty:
            return None
        
        # ปรับโครงสร้างข้อมูล
        df_melted = df.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Score')
        df_melted['Year'] = df_melted['Year'].str.replace('YR', '').astype(int)
        
        # กรองเอาเฉพาะที่มีข้อมูลจริง
        return df_melted.dropna(subset=['Score']).sort_values(['Country', 'Year'])
    except:
        return None

# --- Main Logic ---
if selected_countries:
    data = fetch_wgi_safe(selected_countries, selected_ind_code, start_year)
    
    if data is not None and not data.empty:
        # หาปีล่าสุดที่ธนาคารโลกมีข้อมูลจริง
        latest_yr = data['Year'].max()
        st.success(f"📊 เชื่อมต่อสำเร็จ: ข้อมูลจริงล่าสุดจาก World Bank คือปี {latest_yr}")

        # --- การพยากรณ์ล่วงหน้า 3 ปี ---
        forecast_list = []
        for c in selected_countries:
            c_data = data[data['Country'] == c]
            if len(c_data) >= 3:
                model = LinearRegression().fit(c_data[['Year']], c_data['Score'])
                
                # พยากรณ์ต่อจากปีล่าสุดที่มีข้อมูลจริง
                f_years = np.array([latest_yr + 1, latest_yr + 2, latest_yr + 3]).reshape(-1, 1)
                preds = model.predict(f_years)
                
                for i, yr in enumerate(f_years.flatten()):
                    forecast_list.append({'Country': c, 'Year': int(yr), 'Score': round(preds[i], 3), 'Type': 'Forecast'})

        data['Type'] = 'Actual'
        df_final = pd.concat([data, pd.DataFrame(forecast_list)]).reset_index(drop=True)

        # --- การแสดงผล ---
        # 1. Metrics
        cols = st.columns(len(selected_countries))
        for idx, c in enumerate(selected_countries):
            latest_score = data[data['Country'] == c]['Score'].iloc[-1]
            with cols[idx]:
                st.metric(f"{c} ({latest_yr})", f"{latest_score:.2f}")

        # 2. Graph
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type",
                      markers=True, template="plotly_white", height=500)
        fig.update_layout(yaxis_range=[-2.5, 2.5], hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. Table
        with st.expander("🔎 ตรวจสอบข้อมูลดิบและการพยากรณ์"):
            st.dataframe(df_final.pivot(index='Year', columns='Country', values='Score').sort_index(ascending=False))

    else:
        st.error("⚠️ ไม่สามารถดึงข้อมูลได้ในช่วงปีที่เลือก โปรดลองปรับ 'ปีเริ่มต้นฐานข้อมูล' ให้ย้อนหลังมากขึ้น")
else:
    st.info("👈 กรุณาเลือกประเทศที่ Sidebar")
