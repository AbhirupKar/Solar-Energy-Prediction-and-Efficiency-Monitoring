## 🌞 Solar Energy Efficiency Dashboard

A Streamlit-based interactive dashboard to **analyze**, **visualize**, and **simulate** the efficiency of solar panels using real-world solar sensor data and SHAP-based feature impacts.

---

### 📸 Dashboard Preview

> *Includes live simulation, SHAP impact analysis, and interactive reports.*

---

### 🔧 Features

✅ **Efficiency Over Time Visualization**
✅ **Live Simulation of Solar Efficiency**
✅ **Top Impactful Features (SHAP)**
✅ **SHAP Value vs Efficiency Correlation**
✅ **Expected Feature Impact on Efficiency (%)**
✅ **Correlation Heatmap of Numerical Parameters**
✅ **Low Efficiency Detection with Recommendations**
✅ **Export Filtered Data & Summary Reports**

---

### 🗂️ Project Structure

```
├── predict.py                 # Main Streamlit dashboard code
├── merged_solar_data.csv     # Input dataset (sample solar panel readings)
├── requirements.txt          # Python dependencies
└── README.md                 # GitHub documentation
```

---

### 🚀 Getting Started

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/solar-efficiency-dashboard.git
cd solar-efficiency-dashboard
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the Streamlit app

```bash
streamlit run predict.py
```

---

### 🧪 Dataset Overview

The dataset includes:

* `Date`: Timestamp of measurement
* `Efficiency (%)`: Power conversion efficiency of solar cells
* `Most Impactful Feature`: Top SHAP-driven factor affecting efficiency
* `SHAP Value`: Feature attribution value
* `Expected Efficiency Impact (%)`: Model-predicted contribution
* `Recommendation`: Suggested optimization action

You can replace `merged_solar_data.csv` with your own dataset following the same structure.

---

### 📤 Export Options

* **Download Filtered Data (CSV)**
* **Download Summary Report (TXT)**

Available in the "📤 Export" tab of the dashboard.

---

### 📌 Live Simulation

Watch solar efficiency build over time with adjustable simulation speed.

```python
simulate = st.checkbox("▶️ Start Live Simulation")
```

---

### 📷 Screenshots

| Live Efficiency Chart | Feature Impacts       | SHAP Analysis     |
| --------------------- | --------------------- | ----------------- |
|        |  |  |

---

### 🧑‍💻 Technologies Used

* **Streamlit** – UI and web app framework
* **Pandas** – Data manipulation
* **Plotly** – Interactive plotting
* **Seaborn / Matplotlib** – Correlation plots
* **SHAP** – Model interpretability values (preprocessed)

---

### 🙌 Acknowledgments

Capstone project for **Solar Energy Prediction & Monitoring**, built with 💡 and 📊.

---

