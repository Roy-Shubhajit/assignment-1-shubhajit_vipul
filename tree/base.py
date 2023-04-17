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
import json
from tree.utils import entropy, information_gain, gini_index, get_mse

numpy.seterr(divide = 'ignore') 

np.random.seed(42)


class DecisionTree:
    def __init__(self, criterion: str, input_Dtype: str, output_Dtype: str, depth: int) -> None:
        self.criterion = criterion
        self.input_Dtype = input_Dtype
        self.output_Dtype = output_Dtype
        self.tree = {}
        self.min_samples_split = 2
        self.max_depth=10,
        self.depth=depth
        self.max_class = None
        self.mean_class = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.input_Dtype == "Discrete":
            if self.output_Dtype == "Discrete":
              self.max_class = y.value_counts().idxmax()
            else:
              self.mean_class = y.mean()
            self.tree = self._build_tree_Discrete(X, y, 0)
        else:
            self.tree = self._build_tree_Real(X, y, 0)

    def best_split(self, X: pd.DataFrame, y: pd.Series):
        """
        Funtion to best split on real features
        """
        best_feature = None
        best_value = None
        if self.output_Dtype == "Real":
            mse_base = get_mse(y)
        else:
            info_gain_base = float('-inf')
        df = pd.merge(X, y, right_index=True, left_index=True)
        for feature in list(X.columns):
            Xdf = df.dropna().sort_values(feature)
            xmeans = Xdf[feature].rolling(2).mean()
            xmeans.dropna(inplace=True)
            for value in xmeans:
                left_y = Xdf[Xdf[feature] <= value][y.name]
                right_y = Xdf[Xdf[feature] > value][y.name]
                if self.output_Dtype == "Real":
                    left_mean = left_y.mean()
                    right_mean = right_y.mean()
                    res_left = left_y - left_mean
                    res_right = right_y - right_mean
                    r = pd.concat([res_left, res_right], ignore_index=True)
                    n = len(r)
                    r = r ** 2
                    r = r.sum()
                    mse_split = r / n
                    if mse_split < mse_base:
                        best_feature = feature
                        best_value = value
                        mse_base = mse_split
                else:
                    info_gain = information_gain(pd.concat([left_y, right_y], ignore_index=True), pd.Series(
                        ["Yes"] * len(left_y) + ["No"] * len(right_y), name="Y"), criteria=self.criterion, input_Dtype=self.input_Dtype)
                    if info_gain > info_gain_base:
                        best_feature = feature
                        best_value = value
                        info_gain_base = info_gain

        return best_feature, best_value

    def _select_feature_Discrete(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Funtion to find the best discrete feature
        """
        info_gain = {}
        for attr in X.columns:
            info_gain[attr] = information_gain(
                y, X[attr], self.criterion,  input_Dtype=self.input_Dtype)
        return max(info_gain, key=lambda x: info_gain[x])

    def _build_tree_Discrete(self, X: pd.DataFrame, y: pd.Series, depth: int):
        """
        Funtion to build tree on discrete inputs
        """
        if depth <= self.depth:
            if y.nunique() == 1:
                return y.iloc[0]
            if len(X.columns) == 0:
                return y.value_counts().idxmax()
            best_feature = self._select_feature_Discrete(X, y)
            tree = {best_feature: {}}
            for value in X[best_feature].unique():
                sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
                sub_y = y[X[best_feature] == value]
                if len(sub_X) > 0:
                    tree[best_feature][value] = self._build_tree_Discrete(
                        sub_X, sub_y, depth+1)
                else:
                    tree[best_feature][value] = y.value_counts().idxmax()
        return tree

    def _build_tree_Real(self, X: pd.DataFrame, y: pd.Series, depth: int):
        """
        Funtion to build tree on real inputs
        """
        if depth < self.depth and len(y) >= self.min_samples_split:
            if self.output_Dtype == "Real":
                if len(X.columns) == 0:
                    return y.mean()
            else:
                if len(X.columns) == 0:
                    return y.value_counts().idxmax()
                if y.nunique() == 1:
                    return y.iloc[0]
            best_feature, best_value = self.best_split(X, y)
            if best_feature is not None:
                tree = {best_feature: {best_value: {}}}
                left_X = X[X[best_feature] <= best_value]
                right_X = X[X[best_feature] > best_value]
                left_y = y[X[best_feature] <= best_value]
                right_y = y[X[best_feature] > best_value]
                tree[best_feature][best_value][" <= "] = self._build_tree_Real(
                    left_X, left_y, depth+1)
                tree[best_feature][best_value][" > "] = self._build_tree_Real(
                    right_X, right_y, depth+1)

                return tree
        else:
            if self.output_Dtype == "Real":
                return y.mean()
            else:
                return y.value_counts().idxmax()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        current_node = self.tree
        predictions = []
        for i, row in X.iterrows():
            current_node = self.tree
            # Keep looping until you reach a leaf node
            while(True):
                if isinstance(current_node, dict):
                    split_feature = list(current_node.keys())
                    split_val = row[split_feature]
                    if self.input_Dtype == "Discrete":
                        if split_val.values[0] in list(current_node[split_feature[0]].keys()):
                            current_node = current_node[split_feature[0]][split_val.values[0]]
                        else:
                            if self.output_Dtype == "Discrete":
                              predictions.append(self.max_class)
                            else:
                              predictions.append(self.mean_class)
                            break
                    else:
                        if split_val.values[0] <= list(current_node[split_feature[0]].keys())[0]:
                            current_node = current_node[split_feature[0]][list(current_node[split_feature[0]].keys())[0]]["L"]
                        else:
                            current_node = current_node[split_feature[0]][list(current_node[split_feature[0]].keys())[0]]["R"]
                else:
                    predictions.append(current_node)
                    break

        return pd.Series(predictions)

        return pd.Series(predictions)

    def plot(self) -> None:
        """
        Funtion to plot the decision tree
        """
        print(json.dumps(self.tree, indent=4))
