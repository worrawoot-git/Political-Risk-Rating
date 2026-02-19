import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="WGI Governance & Forecast", layout="wide")

# ปรับแต่ง CSS ให้ดูเป็นมืออาชีพ
st.markdown("""
    <style>
    .main { background-color: #f9fbfd; }
    .stMetric { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eef2f6; }
    h1 { color: #1e40af; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Worldwide Governance Indicators (WGI)")
st.caption("ระบบวิเคราะห์และพยากรณ์ธรรมาภิบาลโลก | ข้อมูลอัปเดตอัตโนมัติจาก World Bank API")

# --- ส่วนของการตั้งค่า (Sidebar) ---
st.sidebar.header("🔍 ตัวเลือกการแสดงผล")

indicator_map = {
    'WGI.PV': 'Political Stability and Absence of Violence',
    'WGI.CC': 'Control of Corruption',
    'WGI.VA': 'Voice and Accountability',
    'WGI.GE': 'Government Effectiveness',
    'WGI.RQ': 'Regulatory Quality',
    'WGI.RL': 'Rule of Law'
}
selected_ind_label = st.sidebar.selectbox("เลือกตัวชี้วัด (Indicator)", list(indicator_map.values()))
selected_ind_code = [k for k, v in indicator_map.items() if v == selected_ind_label][0]

selected_countries = st.sidebar.multiselect(
    "เลือกประเทศ", 
    ["THA", "VNM", "IDN", "SGP", "MYS", "PHL", "USA", "CHN", "JPN", "GBR", "IND", "KOR"],
    default=["THA", "VNM"]
)

# ให้เลือกปีเริ่มต้นเท่านั้น ส่วนปีสิ้นสุดระบบจะหาค่าล่าสุดให้เอง
start_year = st.sidebar.number_input("ปีที่เริ่มดึงข้อมูล (Start Year)", 2000, 2020, 2010)

# --- ฟังก์ชันดึงข้อมูล (Backend) ---
@st.cache_data(ttl=86400)
def fetch_wgi_data(countries, ind_code, start_yr):
    try:
        # ดึงข้อมูลตั้งแต่อดีตจนถึงปัจจุบัน (API จะส่งปีล่าสุดที่มีมาให้เอง)
        df = wb.data.DataFrame(ind_code, countries, time=range(start_yr, datetime.datetime.now().year + 1), labels=True)
        if df is None or df.empty:
            return None
        
        # ปรับโครงสร้างข้อมูล (Transform)
        df_melted = df.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Score')
        df_melted['Year'] = df_melted['Year'].str.replace('YR', '').astype(int)
        
        # ลบค่าว่าง (dropna) เพื่อป้องกันปัญหาไม่มีข้อมูลของปีล่าสุด
        df_clean = df_melted.dropna(subset=['Score']).sort_values(['Country', 'Year'])
        return df_clean
    except Exception as e:
        st.sidebar.error(f"API Connection Error: {e}")
        return None

# --- การประมวลผลและแสดงผลหลัก ---
if selected_countries:
    data = fetch_wgi_data(selected_countries, selected_ind_code, start_year)
    
    if data is not None and not data.empty:
        # หาปีล่าสุดที่มีข้อมูลจริงในระบบ
        latest_actual_year = data['Year'].max()
        st.info(f"💡 ข้อมูลจริงล่าสุดที่มีในฐานข้อมูลธนาคารโลกคือปี: **{latest_actual_year}** (ระบบจะเริ่มพยากรณ์จากปีนี้)")

        # --- ส่วนการพยากรณ์ (Forecasting Logic) ---
        forecast_results = []
        for country in selected_countries:
            c_data = data[data['Country'] == country]
            if len(c_data) >= 3: # ต้องมีข้อมูลอย่างน้อย 3 จุดเพื่อสร้าง Trend
                X = c_data['Year'].values.reshape(-1, 1)
                y = c_data['Score'].values
                model = LinearRegression().fit(X, y)
                
                # พยากรณ์ไปอีก 3 ปีข้างหน้า
                future_years = np.array([latest_actual_year + 1, latest_actual_year + 2, latest_actual_year + 3]).reshape(-1, 1)
                preds = model.predict(future_years)
                
                for i, yr in enumerate(future_years.flatten()):
                    forecast_results.append({
                        'Country': country, 'Year': int(yr), 
                        'Score': round(preds[i], 3), 'Type': 'Forecast'
                    })

        data['Type'] = 'Actual'
        df_forecast = pd.DataFrame(forecast_results)
        df_final = pd.concat([data, df_forecast]).reset_index(drop=True)

        # --- ส่วนแสดงผลบนหน้าเว็บ ---
        # 1. กล่องคะแนน (Metrics)
        cols = st.columns(len(selected_countries))
        for idx, country in enumerate(selected_countries):
            c_latest = data[data['Country'] == country].iloc[-1]
            with cols[idx]:
                st.metric(f"{country} ({latest_actual_year})", f"{c_latest['Score']:.2f}")

        # 2. กราฟ (Interactive Chart)
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type",
                      title=f"แนวโน้มและการพยากรณ์: {selected_ind_label}",
                      markers=True, template="plotly_white", height=500,
                      color_discrete_sequence=px.colors.qualitative.Safe)
        
        fig.update_layout(yaxis_range=[-2.5, 2.5], hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. ตารางข้อมูล (Data Table)
        with st.expander("📊 ตรวจสอบตารางข้อมูลดิบและการพยากรณ์"):
            pivot_df = df_final.pivot(index='Year', columns='Country', values='Score').sort_index(ascending=False)
            st.dataframe(pivot_df, use_container_width=True)
            
        # ปุ่มดาวน์โหลด
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("📥 ดาวน์โหลดข้อมูล (CSV)", csv, "wgi_analysis.csv", "text/csv")

    else:
        st.error("⚠️ ไม่พบข้อมูลสำหรับประเทศที่เลือกในช่วงปีนี้ โปรดเลือกประเทศอื่นหรือปรับปีเริ่มต้น")
else:
    st.warning("👈 กรุณาเลือกประเทศที่แถบเมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์")
