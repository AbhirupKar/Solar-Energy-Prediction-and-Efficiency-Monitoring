import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
from io import BytesIO

# --- Page Config ---
st.set_page_config(page_title="Solar Efficiency Dashboard", layout="wide")

# --- Title ---
st.title("🔆 Solar Energy Efficiency Dashboard")
st.markdown("Track solar panel **efficiency trends**, **SHAP feature impacts**, and **simulate performance** over time.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Mi\Downloads\merged_solar_data.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

data = load_data()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("📅 Filter Date Range")
    start_date = st.date_input("Start Date", value=data['Date'].min())
    end_date = st.date_input("End Date", value=data['Date'].max())

filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))].copy()

# --- Simulation + Efficiency Graph (Main) ---
st.subheader("📉 Efficiency Over Time")

# Create placeholder for the chart
eff_chart_placeholder = st.empty()

simulate = st.checkbox("▶️ Start Live Simulation")
speed = st.slider("⏱️ Speed (sec/day)", 0.1, 2.0, 0.5)

if simulate:
    st.markdown("🧪 Simulation in Progress... Updating graph below:")
    for i in range(1, len(filtered_data) + 1):
        sim_data = filtered_data.iloc[:i]
        fig_sim = px.line(sim_data, x='Date', y='Efficiency (%)',
                          title='Live Efficiency (%) vs Date',
                          markers=True)
        eff_chart_placeholder.plotly_chart(fig_sim, use_container_width=True)
        time.sleep(speed)
else:
    fig_static = px.line(filtered_data, x='Date', y='Efficiency (%)',
                         title='Efficiency (%) vs Date', markers=True)
    eff_chart_placeholder.plotly_chart(fig_static, use_container_width=True)

# --- Tabs Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Impact Features", "📈 SHAP Analysis", "📉 Correlation", "📤 Export"])

# --- Tab 1: Impact Features ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔥 Top Impactful Features (SHAP)")
        top_features = filtered_data['Most Impactful Feature'].value_counts().reset_index()
        top_features.columns = ['Feature', 'Count']
        fig1 = px.bar(top_features, x='Feature', y='Count', title='Most Frequently Impactful Features')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if 'Expected Efficiency Impact (%)' in filtered_data.columns:
            st.markdown("### 🎯 Expected Impact on Efficiency (%)")
            impact_df = filtered_data[['Most Impactful Feature', 'Expected Efficiency Impact (%)']].dropna()
            impact_avg = impact_df.groupby('Most Impactful Feature').mean().reset_index()
            fig2 = px.bar(impact_avg, x='Most Impactful Feature', y='Expected Efficiency Impact (%)',
                          title='Avg Expected Impact per Feature')
            st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: SHAP Value vs Efficiency ---
with tab2:
    st.markdown("### 🧬 SHAP Value vs Efficiency (%)")
    if 'SHAP Value' in filtered_data.columns:
        fig3 = px.scatter(filtered_data, x='SHAP Value', y='Efficiency (%)',
                          color='Most Impactful Feature', trendline="ols",
                          title="SHAP Value vs Efficiency (%)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No SHAP Value column found.")

# --- Tab 3: Correlation Heatmap ---
with tab3:
    st.markdown("### 🧪 Correlation Heatmap of Numeric Features")
    num_data = filtered_data.select_dtypes(include=['float64', 'int64'])
    if not num_data.empty:
        fig4, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig4)
    else:
        st.warning("No numeric data available to plot correlation heatmap.")

# --- Tab 4: Export Section ---
with tab4:
    st.markdown("### 📤 Export Data and Reports")

    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data (CSV)", data=csv, file_name='filtered_solar_data.csv', mime='text/csv')

    # Summary Report
    report_text = f"""
    Solar Efficiency Report
    -------------------------
    Date Range: {start_date} to {end_date}
    Total Records: {len(filtered_data)}

    Avg Efficiency: {filtered_data['Efficiency (%)'].mean():.2f}%
    Max Efficiency: {filtered_data['Efficiency (%)'].max():.2f}%
    Min Efficiency: {filtered_data['Efficiency (%)'].min():.2f}%

    Most Common Feature: {filtered_data['Most Impactful Feature'].mode()[0] if 'Most Impactful Feature' in filtered_data else 'N/A'}
    """

    buffer = BytesIO()
    buffer.write(report_text.encode())
    st.download_button("📄 Download Summary Report", data=buffer, file_name="solar_summary_report.txt", mime="text/plain")

# --- Low Efficiency Days ---
st.subheader("⚠️ Days with Efficiency < 50%")
low_eff_df = filtered_data[filtered_data['Efficiency (%)'] < 50]

if not low_eff_df.empty:
    st.dataframe(low_eff_df[['Date', 'Efficiency (%)', 'Most Impactful Feature', 'Recommendation']])
else:
    st.success("✅ No low-efficiency days in the selected date range.")

# --- Footer ---
st.markdown("---")
st.caption("📘 Capstone Project | Built with ❤️ using Streamlit")
