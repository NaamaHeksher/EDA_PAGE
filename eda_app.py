import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample DataFrame
df = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100),
    'C': np.random.rand(100),
    'D': np.random.rand(100),
})

# Calculate metrics
rows = df.shape[0]
columns = df.shape[1]
missing_percentage = df.isnull().mean().mean() * 100

# Display metrics
st.title('Exploratory Data Analysis App')
st.metric('Rows', rows)
st.metric('Columns', columns)
st.metric('Missing Percentage', f'{missing_percentage:.2f}%')

# Column selector
column = st.selectbox('Select a column for histogram', df.columns)
st.write(f'Histogram for {column}')
plt.hist(df[column], bins=30, alpha=0.7)
st.pyplot(plt)

# Scatter plot with dual column selection
st.subheader('Scatter Plot')
x_column = st.selectbox('Select X-axis column', df.columns)
y_column = st.selectbox('Select Y-axis column', df.columns)
plt.scatter(df[x_column], df[y_column])
plt.xlabel(x_column)
plt.ylabel(y_column)
st.pyplot(plt)