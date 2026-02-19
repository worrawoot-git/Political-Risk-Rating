import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="WGI Interactive Dashboard Clone", layout="wide")

# --- Custom CSS เพื่อให้หน้าตาเหมือน World Bank ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 5px; border-left: 5px solid #0071bc; }
    h1 { color: #0071bc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🏛️ Worldwide Governance Indicators (WGI)")
st.caption("Interactive Data Access Clone | Source: World Bank Group")

# --- Sidebar Filters (เหมือนหน้าเว็บ WGI) ---
st.sidebar.header("Data Selection")

# 1. Select Indicators (อ้างอิงตาม WGI 6 ตัวหลัก)
indicator_map = {
    'WGI.VA': 'Voice and Accountability',
    'WGI.PV': 'Political Stability and Absence of Violence/Terrorism',
    'WGI.GE': 'Government Effectiveness',
    'WGI.RQ': 'Regulatory Quality',
    'WGI.RL': 'Rule of Law',
    'WGI.CC': 'Control of Corruption'
}
selected_ind_label = st.sidebar.selectbox("Select Indicator", list(indicator_map.values()))
selected_ind_code = [k for k, v in indicator_map.items() if v == selected_ind_label][0]

# 2. Select Countries
all_countries = ["THA", "SGP", "VNM", "MYS", "IDN", "PHL", "USA", "GBR", "CHN", "JPN"]
selected_countries = st.sidebar.multiselect("Select Country/Territory", all_countries, default=["THA"])

# 3. Select Year Range
this_year = datetime.datetime.now().year
year_range = st.sidebar.slider("Select Year Range", 2000, this_year, (2015, 2024))

# --- Data Fetching ---
@st.cache_data
def fetch_wgi_data(codes, indicator, years):
    try:
        # ดึงข้อมูลจาก WB API
        df = wb.data.DataFrame(indicator, codes, time=range(years[0], years[1]+1), labels=True)
        return df
    except:
        return None

# --- Main Display ---
if selected_countries:
    data = fetch_wgi_data(selected_countries, selected_ind_code, year_range)
    
    if data is not None:
        # ส่วนแสดง Score ล่าสุด (Metric Boxes)
        st.subheader(f"Current Status: {selected_ind_label}")
        cols = st.columns(len(selected_countries))
        
        # ปรับ Data ให้อยู่ในรูปที่วาดกราฟง่าย (Long Format)
        df_melted = data.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Estimate')
        df_melted['Year'] = df_melted['Year'].str.replace('YR', '').astype(int)
        
        for idx, country in enumerate(selected_countries):
            latest_val = df_melted[df_melted['Country'] == country].sort_values('Year').iloc[-1]['Estimate']
            with cols[idx]:
                st.metric(label=country, value=f"{latest_val:.2f}")

        # ส่วนแสดง Chart
        st.divider()
        tab1, tab2 = st.tabs(["📈 Time Series Chart", "📊 Comparison Table"])
        
        with tab1:
            fig = px.line(df_melted, x="Year", y="Estimate", color="Country",
                          title=f"Trend: {selected_ind_label}",
                          labels={"Estimate": "Governance Score (-2.5 to 2.5)"},
                          markers=True, line_shape="linear")
            fig.update_layout(hovermode="x unified", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.dataframe(data, use_container_width=True)
            
        # ปุ่ม Download (เหมือนหน้าเว็บจริง)
        csv = data.to_csv().encode('utf-8')
        st.download_button("📥 Export to Excel/CSV", csv, "wgi_data.csv", "text/csv")

    else:
        st.error("ไม่สามารถดึงข้อมูลได้ โปรดลองเลือกประเทศใหม่อีกครั้ง")
else:
    st.info("👈 Please select at least one country in the sidebar to view the dashboard.")

# --- Footer ข้อมูลอธิบาย ---
with st.expander("ℹ️ About the Indicators"):
    st.write("""
    **Estimate:** ให้คะแนนระหว่าง -2.5 (แย่ที่สุด) ถึง 2.5 (ดีที่สุด)
    - **Voice and Accountability:** สิทธิเสรีภาพและการมีส่วนร่วม
    - **Control of Corruption:** การควบคุมคอร์รัปชัน
    - **Political Stability:** เสถียรภาพทางการเมือง
    """)
