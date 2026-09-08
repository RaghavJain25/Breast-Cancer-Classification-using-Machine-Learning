import streamlit as st
import pandas as pd

st.title("Breast Cancer Classification")
st.write("Machine Learning based Breast Cancer Prediction")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Dataset Information")
    st.write("Number of rows:", data.shape[0])
    st.write("Number of columns:", data.shape[1])
