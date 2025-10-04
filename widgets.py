import streamlit as st
import pandas as pd
import numpy as np  


data=[
    {"name": "Alice", "age": 30, "city": "New York"},
    {"name": "Bob", "age": 25, "city": "San Francisco"},
    {"name": "Charlie", "age": 35, "city": "Los Angeles"}
]

df=pd.DataFrame(data)
st.write(df)
df.to_csv("people.csv", index=False)


st.title("Streamlit Widgets Example")
uploaded_file=st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)