import streamlit as st
import pandas as pd
import numpy as np  

st.title("Hello, Streamlit!")
st.write("Welcome to your first Streamlit app.")


df=pd.DataFrame(
    {'first column':[1,2,3,4],
     'second column':[10,20,30,40]}

)

chart=pd.DataFrame(
    np.random.randn(20,3),
    columns=['a','b','c']
)
st.line_chart(chart)