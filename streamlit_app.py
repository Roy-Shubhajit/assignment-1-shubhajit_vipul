import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from tree.base import DecisionTree
from metrics import *

st.title("Bias Variance Tradeoff")
st.write("This app plots the bias variance tradeoff for a decision tree model")
st.sidebar.header("User Input Parameters")

def user_input_features():
    depth = st.sidebar.slider("Tree Depth", 1, 10, 1)
    return depth

depth = user_input_features()
test_size = 0.3
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=20, random_state=1)
df = pd.DataFrame(X)
trg_col = 'Y'
df[trg_col] = y
num_data_pnts = len(df.index)
df_train = df.loc[0:(1-test_size)*num_data_pnts-1]
df_test = df.loc[(1-test_size)*num_data_pnts:num_data_pnts]
dt = DecisionTree(df_train, df_test, trg_col, depth, "Real", "Real", depth=depth)
dt.fit(df_train.drop([trg_col], axis = 1), df_train[trg_col])
y_pred = dt.predict(df_test.drop([trg_col], axis = 1))
error = mse(y_pred - df_test[trg_col])
st.write("MSE: ", mse)
st.write("Bias: ", (df_test[trg_col].mean() - y_pred.mean())**2)
st.write("Variance: ", y_pred.var())
st.write("Bias + Variance: ", (df_test[trg_col].mean() - y_pred.mean())**2 + y_pred.var())

st.write("The plot below shows the bias variance tradeoff for a decision tree model with varying tree depth")
st.write("The MSE is the sum of bias and variance")