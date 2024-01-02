import pandas as pd
import numpy as np
import streamlit as st

data = pd.read_csv('diabetes_prediction_dataset.csv')
st.table(data=data)

