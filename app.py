# ---------------- IMPORT LIBRARIES ----------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Household Power Consumption Prediction",
    page_icon="⚡",
    layout="centered"
)

# ---------------- BACKGROUND IMAGE ----------------
page_bg = """
<style>
.stApp {
    background-image: url(""https://images.unsplash.com/photo-1496588152823-86ff7695a68d
"");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center; color: white;'>⚡ Household Power Consumption Prediction</h1>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("household_power_consumption.csv")
    df.replace('?', np.nan, inplace=True)

    cols = ['Global_active_power', 'Voltage', 'Global_intensity']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
    return df

df = load_data()

# ---------------- FEATURES ----------------
X = df[['Voltage', 'Global_intensity']]
y = df['Global_active_power']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Enter Input Values")

voltage = st.sidebar.number_input("Voltage", min_value=100.0, max_value=300.0, value=240.0)
intensity = st.sidebar.number_input("Global Intensity", min_value=0.0, max_value=50.0, value=10.0)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Power Consumption"):

    input_data = pd.DataFrame(
        [[voltage, intensity]],
        columns=['Voltage', 'Global_intensity']
    )

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"<h2 style='color:black;'>Predicted Power Consumption: {prediction:.3f} kW</h2>",
        unsafe_allow_html=True
    )

# ---------------- MODEL PERFORMANCE ----------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.markdown(
    f"<h4 style='color:black;'>Model R² Score: {r2:.3f}</h4>",
    unsafe_allow_html=True
)
