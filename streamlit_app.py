import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from tree.base import DecisionTree
from metrics import *
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


st.title("Bias Variance Tradeoff")
st.write("This app plots the bias variance tradeoff for a decision tree model")
st.sidebar.header("User Input Parameters")

with st.sidebar:
    option = st.selectbox(
    'Which dataset to use',
    ('SK Learn make_regression', 'y = x**2 + 1 + eps'))
    number_of_data_points = st.slider("Number of Data Points", 10, 1000, 100)
    depth = st.slider("Tree Depth", 1, 10, 1)

if option == 'y = x**2 + 1 + eps':
    X = np.linspace(0, 10, number_of_data_points)
    y = X**2 + 1 + np.random.normal(0, 1, number_of_data_points)
    df = pd.DataFrame(X)
    trg_col = 'Y'
    df[trg_col] = y
    num_data_pnts = len(df.index)
else:
    X, y = make_regression(n_samples=number_of_data_points, n_features=1, n_informative=2, noise=20, random_state=1)
    df = pd.DataFrame(X)
    trg_col = 'Y'
    df[trg_col] = y
    num_data_pnts = len(df.index)


model = make_pipeline(DecisionTree(criterion="mse", input_Dtype="Real", output_Dtype="Real", depth=depth))
model.fit(df.drop([trg_col], axis = 1), df[trg_col])
y_pred = model.predict(df.drop([trg_col], axis = 1))
error = rmse(y_pred, df[trg_col])
st.write("RMSE: ", error)
st.write("Bias: ", (df[trg_col].mean() - y_pred.mean())**2)
st.write("Variance: ", y_pred.var())
st.write("Bias + Variance: ", (df[trg_col].mean() - y_pred.mean())**2 + y_pred.var())

st.write("The plot below shows the bias variance tradeoff for a decision tree model with varying tree depth")

#scatter plot predicted points
st.write("Predicted Points")
fig, ax = plt.subplots()
ax.scatter(df.drop([trg_col], axis = 1), y_pred, color='red', marker='o', label='Predicted')
ax.scatter(df.drop([trg_col], axis = 1), df[trg_col], color='black', marker='*', label='Actual')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
st.pyplot(fig)
