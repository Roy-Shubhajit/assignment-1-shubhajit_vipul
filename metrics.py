from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    y = y.to_numpy()
    return (y_hat == y).sum()/len(y_hat)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    if cls not in y_hat.values:
        return 0.0   
    y = y.to_numpy()  
    tp = sum((y==cls)&(y_hat==cls))
    fp = sum((y!=cls)&(y_hat==cls))
    return tp/(tp+fp)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    if cls not in y_hat.values:
        return 0.0  
    y = y.to_numpy()
    tp = sum((y==cls)&(y_hat==cls))
    fn = sum((y==cls)&(y_hat!=cls))
    return tp/(tp+fn)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    n = len(y)
    r = y - y_hat
    r = r ** 2
    r = r.sum()
    r = r / n

    return r**(1/2)
def mse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-squared-error(mse)
    """
    n = len(y)
    r = y - y_hat
    r = r ** 2
    r = r.sum()
    r = r / n

    return r

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    n = len(y)
    r = y - y_hat
    r = r.abs()
    r = r.sum()
    r = r / n

    return r

