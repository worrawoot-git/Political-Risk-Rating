import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Config ---
st.set_page_config(page_title="Global Governance & Forecast", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    h1, h2 { color: #0071bc; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🏛️ Worldwide Governance & Political Forecast")
st.caption(f"Data Source: World Bank WGI | System Date: {datetime.datetime.now().year}")

# --- Sidebar ---
st.sidebar.header("Filter Settings")
this_year = datetime.datetime.now().year

indicator_map = {
    'WGI.PV': 'Political Stability',
    'WGI.CC': 'Control of Corruption',
    'WGI.VA': 'Voice & Accountability',
    'WGI.GE': 'Government Effectiveness'
}
selected_ind_label = st.sidebar.selectbox("Select Indicator", list(indicator_map.values()))
selected_ind_code = [k for k, v in indicator_map.items() if v == selected_ind_label][0]

selected_countries = st.sidebar.multiselect(
    "Select Countries", 
    ["THA", "VNM", "IDN", "SGP", "MYS", "PHL", "USA", "CHN", "JPN", "GBR"],
    default=["THA", "VNM"]
)

year_range = st.sidebar.slider("Historical Range", 2005, this_year, (2015, this_year))

# --- Function: Fetch Data ---
@st.cache_data(ttl=86400)
def get_clean_data(codes, ind_code, years):
    try:
        # ดึงข้อมูลจาก WB
        df = wb.data.DataFrame(ind_code, codes, time=range(years[0], years[1]+1), labels=True)
        if df.empty: return None
        # ปรับโครงสร้างข้อมูล
        df_long = df.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Score')
        df_long['Year'] = df_long['Year'].str.replace('YR', '').astype(int)
        df_long = df_long.dropna(subset=['Score'])
        return df_long
    except:
        return None

# --- Main Logic ---
if selected_countries:
    raw_data = get_clean_data(selected_countries, selected_ind_code, year_range)
    
    if raw_data is not None and not raw_data.empty:
        # 1. Visualization
        st.subheader(f"📈 Trend Analysis: {selected_ind_label}")
        
        # ส่วนการพยากรณ์
        forecast_data = []
        for c in selected_countries:
            c_df = raw_data[raw_data['Country'] == c]
            if len(c_df) > 1:
                X = c_df['Year'].values.reshape(-1, 1)
                y = c_df['Score'].values
                model = LinearRegression().fit(X, y)
                
                # พยากรณ์ 3 ปี
                last_y = c_df['Year'].max()
                future_yrs = np.array([last_y+1, last_y+2, last_y+3]).reshape(-1, 1)
                preds = model.predict(future_yrs)
                
                for idx, fy in enumerate(future_yrs.flatten()):
                    forecast_data.append({'Country': c, 'Year': int(fy), 'Score': preds[idx], 'Type': 'Forecast'})

        # รวมข้อมูลจริงและพยากรณ์
        raw_data['Type'] = 'Actual'
        df_all = pd.concat([raw_data, pd.DataFrame(forecast_data)])
        
        fig = px.line(df_all, x="Year", y="Score", color="Country", line_dash="Type", 
                      markers=True, template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # 2. Summary Table (Fixed Error)
        st.divider()
        st.subheader("📋 Data Summary")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # แสดงตารางแบบไม่มี Gradient เพื่อความเสถียร แต่ใช้การจัดฟอร์แมตตัวเลขแทน
            st.dataframe(df_all.sort_values(['Country', 'Year'], ascending=[True, False]), use_container_width=True)
        
        with col2:
            st.info("""
            **Score Guide:**
            - **+2.5**: Strong / High Stability
            - **0.0**: Average
            - **-2.5**: Weak / Low Stability
            """)
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "governance_data.csv", "text/csv")
            
    else:
        st.error("❌ ไม่พบข้อมูลสำหรับประเทศหรือช่วงเวลาที่เลือก โปรดลองขยายช่วงปี (Historical Range) ให้กว้างขึ้น")
else:
    st.info("👈 กรุณาเลือกประเทศที่ Sidebar เพื่อเริ่มต้น")
