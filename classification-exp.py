from sklearn.datasets import make_classification, make_regression, make_blobs
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

numpy.seterr(divide = 'ignore') 

np.random.seed(42)

test_size = 0.3
depth = 5
X, y = make_classification(
n_features=10, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
df = pd.DataFrame(X)
trg_col = 'Y'
df[trg_col] = y
num_data_pnts = len(df.index)
df_train = df.loc[0:(1-test_size)*num_data_pnts-1]
df_test = df.loc[(1-test_size)*num_data_pnts:num_data_pnts]
dt = DecisionTree(criterion = 'entropy', input_Dtype = 'Real', output_Dtype = 'Discrete', depth = depth)
dt.fit(df_train.drop([trg_col], axis = 1), df_train[trg_col])
predictions = pd.Series(dt.predict(df_test.drop([trg_col], axis=1)))
print('Accuracy: ',accuracy(predictions, df_test[trg_col]))
print(" ")
for cls in df[trg_col].unique(): 
  print('Precision--> Class='+str(cls),precision(predictions, df_test[trg_col], cls = cls))
  print('Recall--> Class='+str(cls),recall(predictions, df_test[trg_col], cls = cls))
  print(" ")

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

max_depth = 10
test_size = 0.3
k_fold = 5

df = pd.DataFrame(X)
df_num_pnts = df.shape[0]
trg_col = 'Y'
df[trg_col] = y

k_fold_out = int(np.ceil(k_fold*test_size))
k_fold_in = int(k_fold - k_fold_out)

num_data_pnts = len(df.index)
subset_k_in = int(df_num_pnts/k_fold)
subset_k_out = k_fold_out*subset_k_in
acc = np.zeros([max_depth, k_fold_in, k_fold_out])
for k_out in range(k_fold_out):
    df_test = df[k_out*subset_k_out:(k_out+1)*subset_k_out]
    df_train_val = df[0:k_out*subset_k_out]
    df_train_val = pd.concat([df_train_val,df[(k_out+1)*subset_k_out:df_num_pnts]])
    df_trn_vl_pnts = df_train_val.shape[0]
    for k_in in range(k_fold_in):
        df_val_k = df_train_val[k_in*subset_k_in:(k_in+1)*subset_k_in]
        df_train_k = df_train_val[0:k_in*subset_k_in]
        df_train_k = pd.concat([df_train_k,df_train_val[(k_in+1)*subset_k_in:df_trn_vl_pnts]])
        for d in range(max_depth):
            dt = DecisionTree(criterion = 'entropy', input_Dtype = 'Real', output_Dtype = 'Discrete', depth = d)
            dt.fit(df_train_k.drop([trg_col], axis=1), df_train_k[trg_col])
            predictions = pd.Series(dt.predict(df_val_k.drop([trg_col], axis=1)))
            acc[d,k_in, k_out] = accuracy(predictions, df_val_k[trg_col])

acc = acc.mean(axis=1).mean(axis=1)
opt_depth = np.where(acc == acc.max())[0][0]
print("Best dept = ",opt_depth)
plt.plot(acc)
plt.ylabel('Validation Accuracy')
plt.xlabel('Depth')
plt.title('Optimal Depth of Decision Tree')
plt.show()
