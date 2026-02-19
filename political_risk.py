import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="WGI Governance Expert", layout="wide")

st.markdown("""
    <style>
    .stMetric { background: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e1e8ed; }
    h1 { color: #0071bc; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Worldwide Governance Indicators (WGI)")
st.caption("ระบบวิเคราะห์ความเสี่ยงการเมืองและธรรมาภิบาล (Stable Version)")

# --- Sidebar ---
st.sidebar.header("⚙️ การตั้งค่าข้อมูล")

indicator_map = {
    'WGI.PV': 'Political Stability',
    'WGI.CC': 'Control of Corruption',
    'WGI.VA': 'Voice and Accountability',
    'WGI.GE': 'Government Effectiveness',
    'WGI.RQ': 'Regulatory Quality',
    'WGI.RL': 'Rule of Law'
}
selected_ind_label = st.sidebar.selectbox("เลือกดัชนีชี้วัด", list(indicator_map.values()))
selected_ind_code = [k for k, v in indicator_map.items() if v == selected_ind_label][0]

selected_countries = st.sidebar.multiselect(
    "เลือกประเทศ", 
    ["THA", "VNM", "IDN", "SGP", "MYS", "PHL", "USA", "CHN", "JPN", "GBR", "IND", "KOR"],
    default=["THA", "VNM"]
)

# ล็อกปีเริ่มต้นให้ย้อนหลังพอสมควรเพื่อให้มีข้อมูลเพียงพอต่อการพยากรณ์
start_year = st.sidebar.slider("จุดเริ่มต้นของข้อมูลจริง", 2000, 2018, 2010)

# --- Backend: ดึงข้อมูลแบบปลอดภัย (Locked Year Fetching) ---
@st.cache_data(ttl=86400)
def fetch_wgi_fixed(countries, ind_code, start_yr):
    try:
        # **จุดสำคัญ**: ล็อกการดึงข้อมูลไว้ที่ปี 2023 เพื่อไม่ให้ API Error จากปีที่ไม่มีข้อมูล
        # เราจะใช้วิธีพยากรณ์เพื่อดูปี 2024, 2025, 2026 แทน
        df = wb.data.DataFrame(ind_code, countries, time=range(start_yr, 2024), labels=True)
        
        if df is None or df.empty:
            return None
        
        df_long = df.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Score')
        df_long['Year'] = df_long['Year'].str.replace('YR', '').astype(int)
        
        return df_long.dropna(subset=['Score']).sort_values(['Country', 'Year'])
    except:
        return None

# --- Main App Logic ---
if selected_countries:
    data = fetch_wgi_fixed(selected_countries, selected_ind_code, start_year)
    
    if data is not None and not data.empty:
        # แจ้งสถานะข้อมูลล่าสุดที่มีจริง
        last_actual_year = data['Year'].max()
        st.success(f"📊 ข้อมูลจริงล่าสุดจาก World Bank (ปี {last_actual_year}) ถูกดึงมาเรียบร้อยแล้ว")

        # --- การพยากรณ์ล่วงหน้าไปจนถึงปี 2026 ---
        forecast_list = []
        target_years = [2024, 2025, 2026] # ปีที่เราจะสร้างขึ้นเองผ่านสถิติ
        
        for country in selected_countries:
            c_data = data[data['Country'] == country]
            if len(c_data) >= 3:
                # คำนวณแนวโน้มด้วย Linear Regression
                model = LinearRegression().fit(c_data[['Year']], c_data['Score'])
                
                # ทำนายค่าสำหรับปีที่หายไป
                f_years = np.array(target_years).reshape(-1, 1)
                preds = model.predict(f_years)
                
                for i, yr in enumerate(target_years):
                    forecast_list.append({
                        'Country': country, 
                        'Year': yr, 
                        'Score': round(preds[i], 3), 
                        'Type': 'Forecast (AI Prediction)'
                    })

        data['Type'] = 'Actual Data'
        df_final = pd.concat([data, pd.DataFrame(forecast_list)]).reset_index(drop=True)

        # --- การแสดงผล (Visuals) ---
        # 1. กล่องคะแนน Metric
        m_cols = st.columns(len(selected_countries))
        for idx, country in enumerate(selected_countries):
            latest_score = data[data['Country'] == country]['Score'].iloc[-1]
            with m_cols[idx]:
                st.metric(f"{country} Score", f"{latest_score:.2f}")

        # 2. กราฟเส้นแบบพยากรณ์
        fig = px.line(df_final, x="Year", y="Score", color="Country", line_dash="Type",
                      markers=True, template="plotly_white", height=550,
                      title=f"แนวโน้มย้อนหลังและการพยากรณ์ถึงปี 2026: {selected_ind_label}")
        
        fig.update_layout(yaxis_range=[-2.5, 2.5], hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. ตารางข้อมูล
        with st.expander("🔎 ตารางข้อมูลสรุป (ปี 2000 - 2026)"):
            st.dataframe(df_final.pivot(index='Year', columns='Country', values='Score').sort_index(ascending=False), use_container_width=True)

    else:
        st.error("❌ ไม่สามารถดึงข้อมูลได้ โปรดลองปรับปีเริ่มต้นให้ย้อนหลังมากขึ้น หรือเปลี่ยนดัชนีชี้วัด")
else:
    st.info("👈 กรุณาเลือกประเทศที่แถบเมนูด้านซ้ายเพื่อเริ่มการแสดงผล")
