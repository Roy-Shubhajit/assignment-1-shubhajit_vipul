"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.seterr(divide = 'ignore')
np.random.seed(42)
# Test case 1
# Real Input and Real Output
print("Real Input and Real Output")
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P), columns=["X1", "X2", "X3", "X4", "X5"])
y = pd.Series(np.random.randn(N), name="Y")

tree = DecisionTree(criterion=None, input_Dtype = "Real", output_Dtype = "Real", depth = 5) #Split based on Inf. Gain
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print('Criteria :', "MSE")
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print(" ")

# Test case 2
# Real Input and Discrete Output
print("Real Input and Discrete Output")
N = 30
P = 5
choice_y = ["a", "b", "c", "d", "e"]
X = pd.DataFrame(np.random.randn(N, P), columns=["X1", "X2", "X3", "X4", "X5"])
y = pd.Series(np.random.choice(choice_y, 30), dtype="category", name="Y")

for criteria in ['entropy', 'gini_index']:
    tree = DecisionTree(criterion=None, input_Dtype = "Real", output_Dtype = "Discrete", depth = 6) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    print(" ")
    for cls in y.unique():
        print(f'Precision--> Class={cls}: ', precision(y_hat, y, cls))
        print(f'Recall-->Class={cls}: ', recall(y_hat, y, cls))
        print(" ")
print(" ")

# Test case 3
# Discrete Input and Discrete Output
print("Discrete Input and Discrete Output")
N = 30
P = 5
choice_X = ["A", "B", "C", "D", "E"]
choice_y = ["a", "b", "c", "d", "e"]
X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in choice_X})
y = pd.Series(np.random.choice(choice_y, 30), dtype="category", name="Y")

for criteria in ['entropy', 'gini_index']:
    tree = DecisionTree(criterion=criteria, input_Dtype = "Discrete", output_Dtype = "Discrete", depth = 5) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    print(" ")
    for cls in y.unique():
        print(f'Precision--> Class={cls}: ', precision(y_hat, y, cls))
        print(f'Recall-->Class={cls}: ', recall(y_hat, y, cls))
        print(" ")
print(" ")

# Test case 4
# Discrete Input and Real Output
print("Discrete Input and Real Output")
N = 30
P = 5
choice_X = ["A", "B", "C", "D", "E"]
choice_y = ["a", "b", "c", "d", "e"]
X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in choice_X})
y = pd.Series(np.random.randn(N), name="Y")

for criteria in ['variance', 'std']:
    tree = DecisionTree(criterion=criteria, input_Dtype = "Discrete", output_Dtype = "Real", depth = 5) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    print(" ")
