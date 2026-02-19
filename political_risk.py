import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="WGI Global Analytics", layout="wide")

st.markdown("""
    <style>
    .stMetric { background: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e1e8ed; }
    h1 { color: #0071bc; font-family: sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Worldwide Governance Indicators (WGI)")
st.caption("วิเคราะห์ดัชนีธรรมาภิบาลโลกและการพยากรณ์ล่วงหน้า (ฉบับแก้ไขการเชื่อมต่อ)")

# --- Sidebar ---
st.sidebar.header("⚙️ ตั้งค่าการดึงข้อมูล")

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

# ให้เลือกเฉพาะปีเริ่มต้น ส่วนปีจบระบบจะหาค่าล่าสุดให้เองอัตโนมัติ
start_year = st.sidebar.slider("เลือกปีเริ่มต้นฐานข้อมูล", 2000, 2018, 2010)

# --- ฟังก์ชันดึงข้อมูลแบบปลอดภัย (Robust Fetching) ---
@st.cache_data(ttl=86400)
def fetch_wgi_robust(countries, ind_code, start_yr):
    try:
        # ดึงข้อมูลโดยไม่ระบุปีจบ (API จะส่งปีล่าสุดที่มีจริงมาให้เองโดยไม่ Error)
        # เราดึงเผื่อย้อนหลังมาจนถึงปีปัจจุบัน (2026) เพื่อให้ระบบกรองเอง
        df = wb.data.DataFrame(ind_code, countries, time=range(start_yr, 2027), labels=True)
        if df is None or df.empty:
            return None
        
        # ปรับโครงสร้างข้อมูล (Reshape)
        df_long = df.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Score')
        df_long['Year'] = df_long['Year'].str.replace('YR', '').astype(int)
        
        # กรองเฉพาะแถวที่มีตัวเลขจริง (กำจัดปีที่ยังไม่มีข้อมูล)
        return df_long.dropna(subset=['Score']).sort_values(['Country', 'Year'])
    except Exception as e:
        return None

# --- ส่วนประมวลผลหลัก ---
if selected_countries:
    data = fetch_wgi_robust(selected_countries, selected_ind_code, start_year)
    
    if data is not None and not data.empty:
        # หาปีล่าสุดที่มีข้อมูลจริง
        latest_actual_year = data['Year'].max()
        st.success(f"📊 เชื่อมต่อสำเร็จ: ข้อมูลจริงล่าสุดคือปี {latest_actual_year}")

        # --- การพยากรณ์ล่วงหน้า 3 ปี (Statistical Forecasting) ---
        forecast_results = []
        for country in selected_countries:
            c_df = data[data['Country'] == country]
            if len(c_df) >= 3:
                # ใช้ Linear Regression คำนวณแนวโน้ม
                X = c_df[['Year']].values
                y = c_df['Score'].values
                model = LinearRegression().fit(X, y)
                
                # พยากรณ์ต่อจากปีล่าสุดที่มีข้อมูลไปอีก 3 ปี
                future_yrs = np.array([latest_actual_year+1, latest_actual_year+2, latest_actual_year+3]).reshape(-1, 1)
                preds = model.predict(future_yrs)
                
                for i, yr in enumerate(future_yrs.flatten()):
                    forecast_results.append({'Country': country, 'Year': int(yr), 'Score': round(preds[i], 3), 'Type': 'Forecast'})

        data['Type'] = 'Actual'
        df_forecast = pd.DataFrame(forecast_results)
        df_final = pd.concat([data, df_forecast]).reset_index(drop=True)

        # --- การแสดงผล (UI) ---
        # 1. แสดงตัวเลขสรุป
        m_cols = st.columns(len(selected_countries))
        for idx, country in enumerate(selected_countries):
            last_score = data[data['Country'] == country].iloc[-1]['Score']
            with m_cols[idx]:
                st.metric(f"{country} ({latest_actual_year})", f"{last_score:.2f}")

        # 2. กราฟเส้นแนวโน้ม
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type",
                      markers=True, template="plotly_white", height=550,
                      title=f"แนวโน้มและคาดการณ์: {selected_ind_label}")
        
        # ปรับแต่งกราฟ
        fig.update_layout(yaxis_range=[-2.5, 2.5], hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. ตารางข้อมูล
        with st.expander("🔎 ตรวจสอบตารางข้อมูลและค่าพยากรณ์"):
            pivot_df = df_final.pivot(index='Year', columns='Country', values='Score').sort_index(ascending=False)
            st.dataframe(pivot_df, use_container_width=True)
            
        # ปุ่มดาวน์โหลด
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("📥 ดาวน์โหลดข้อมูล (CSV)", csv, "wgi_analysis_export.csv", "text/csv")

    else:
        st.error("❌ ไม่สามารถดึงข้อมูลได้ โปรดลองปรับ 'ปีเริ่มต้น' หรือเปลี่ยนประเทศ")
else:
    st.info("👈 กรุณาเลือกประเทศที่เมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์")
