import pandas as pd
import numpy as np
import streamlit as st

df = pd.DataFrame(
    np.random.randn(100, 2) / [0.5, 0.5] + [55.5, 37.33],
    columns=['lat', 'lon'])
st.map(df)