import pandas as pd
import numpy as np

def entropy(Y: pd.Series) -> float:
    count_class = Y.value_counts()
    prob_class = count_class/count_class.sum()
    entropy = - prob_class * np.log2(prob_class)
    return entropy.sum()

def gini_index(Y: pd.Series) -> float:
    count_class = Y.value_counts()
    prob_class = count_class/count_class.sum()
    gini_index = 1 - (prob_class**2).sum()
    return gini_index


def information_gain(Y: pd.Series, attr: pd.Series, criteria: str, input_Dtype: str) -> float:
    Y = Y
    parent_impurity = entropy(Y)
    subsets_impurities = {}
    subsets_size = {}
    merged = pd.merge(attr, Y, right_index = True, left_index = True)
    for value in attr.unique():
        subset = merged.groupby(merged.columns[0]).get_group(value)[merged.columns[1]]
        if criteria == 'gini_index':
            subsets_impurities[value] = gini_index(subset)
        elif criteria == "entropy":
            subsets_impurities[value] = entropy(subset)
        elif criteria == "variance":
            subsets_impurities[value] = subset.var()
        elif criteria == "std":
            subsets_impurities[value] = subset.std()
        subsets_size[value] = len(subset)/len(attr)
    if criteria == 'gini_index' or criteria == "variance" or criteria == "std":
        information_gain = sum(subsets_impurities[k]*subsets_size[k] for k in subsets_impurities)
    else:
        if input_Dtype == "Discrete":
            information_gain = parent_impurity - sum(subsets_impurities[k]*subsets_size[k] for k in subsets_impurities)
        else:
            information_gain = -sum(subsets_impurities[k]*subsets_size[k] for k in subsets_impurities)
    return information_gain

def get_mse(Y: pd.Series) -> float:
    n = len(Y)
    r = Y - Y.mean() 
    r = r ** 2
    r = r.sum()
    return r / n
