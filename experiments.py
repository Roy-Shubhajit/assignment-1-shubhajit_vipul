
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification, make_regression

numpy.seterr(divide = 'ignore') 
plt.rcParams['figure.figsize'] = [15, 30]

np.random.seed(42)
num_average_time = 100
def sample_run_time(criterion: str, input_Dtype: str, output_Dtype: str, test_size: float, depth: int, M: int, N: int):
  train_time = []
  test_time = []
  n_samp = []

  if input_Dtype == "Real" and output_Dtype == "Real":
    columns = []
    for i in range(M):
      columns.append(f"X{i+1}")
    X = pd.DataFrame(np.random.randn(N, M), columns=columns)
    y = pd.Series(np.random.randn(N), name="Y")
  elif input_Dtype == "Real" and output_Dtype == "Discrete":
    columns = []
    choice_y = []
    for i in range(M):
      columns.append(f"X{i+1}")
      choice_y.append(f"y{i+1}")
    X = pd.DataFrame(np.random.randn(N, M), columns=columns)
    y = pd.Series(np.random.choice(choice_y, N), dtype="category", name="Y")
  elif input_Dtype == "Discrete" and output_Dtype == "Discrete":
    columns = []
    choice_y = []
    for i in range(M):
      columns.append(f"X{i+1}")
      choice_y.append(f"y{i+1}")
    X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in columns})
    y = pd.Series(np.random.choice(choice_y, N), dtype="category", name="Y")
  else:
    columns = []
    for i in range(M):
      columns.append(f"X{i+1}")
    X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in columns})
    y = pd.Series(np.random.randn(N), name="Y")

  for i in range(2, N+2):
      df = df = pd.merge(X[:i], y[:i], right_index=True, left_index=True)
      trg_col = y.name
      num_data_pnts = len(df.index)
      df_train = df.loc[0:(1-test_size)*num_data_pnts-1]
      df_test = df.loc[(1-test_size)*num_data_pnts:num_data_pnts]
      dt = DecisionTree(criterion = criterion, input_Dtype = input_Dtype, output_Dtype = output_Dtype, depth = depth)
      start_time = time.time()
      dt.fit(df_train.drop([trg_col], axis=1), df_train[trg_col])
      train_time.append((time.time() - start_time))
      start_time = time.time()
      predictions = dt.predict(df_test.drop([trg_col], axis=1))
      test_time.append((time.time() - start_time))
      n_samp.append(i)
  return train_time, test_time, n_samp

def feature_run_time(criterion: str, input_Dtype: str, output_Dtype: str, test_size: float, depth: int, M: int, N: int):
  train_time = []
  test_time = []
  n_samp = []

  if input_Dtype == "Real" and output_Dtype == "Real":
    columns = []
    for i in range(M):
      columns.append(f"X{i+1}")
    X = pd.DataFrame(np.random.randn(N, M), columns=columns)
    y = pd.Series(np.random.randn(N), name="Y")
  elif input_Dtype == "Real" and output_Dtype == "Discrete":
    columns = []
    choice_y = []
    for i in range(M):
      columns.append(f"X{i+1}")
      choice_y.append(f"y{i+1}")
    X = pd.DataFrame(np.random.randn(N, M), columns=columns)
    y = pd.Series(np.random.choice(choice_y, N), dtype="category", name="Y")
  elif input_Dtype == "Discrete" and output_Dtype == "Discrete":
    columns = []
    choice_y = []
    for i in range(M):
      columns.append(f"X{i+1}")
      choice_y.append(f"y{i+1}")
    X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in columns})
    y = pd.Series(np.random.choice(choice_y, N), dtype="category", name="Y")
  else:
    columns = []
    for i in range(M):
      columns.append(f"X{i+1}")
    X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in columns})
    y = pd.Series(np.random.randn(N), name="Y")

  for i in range(1, M+1):
      df = pd.merge(X[X.columns[:i]], y, right_index=True, left_index=True)
      trg_col = y.name
      num_data_pnts = len(df.index)
      df_train = df.loc[0:(1-test_size)*num_data_pnts-1]
      df_test = df.loc[(1-test_size)*num_data_pnts:num_data_pnts]
      dt = DecisionTree(criterion = criterion, input_Dtype = input_Dtype, output_Dtype = output_Dtype, depth = depth)
      start_time = time.time()
      dt.fit(df_train.drop([trg_col], axis = 1), df_train[trg_col])
      train_time.append((time.time() - start_time))
      start_time = time.time()
      predictions = dt.predict(df_test.drop([trg_col], axis = 1))
      test_time.append((time.time() - start_time))
      n_samp.append(i)
  return train_time, test_time, n_samp
fig, axs = plt.subplots(4, 2,squeeze=False)
train_time, test_time, n_samp = sample_run_time(input_Dtype = 'Real', output_Dtype = 'Real', test_size=0.5, depth = 5, M=10, N=50, criterion = None)
axs[0,0].plot(n_samp, train_time)
axs[0,0].set_title('Real - Real')
train_time, test_time, n_samp = feature_run_time(input_Dtype = 'Real', output_Dtype = 'Real', test_size=0.5, depth = 5, M=10, N=50, criterion = None)
axs[0,1].plot(n_samp, train_time)
axs[0,1].set_title('Real - Real')

train_time, test_time, n_samp = sample_run_time(input_Dtype = 'Real', output_Dtype = 'Discrete', test_size=0.5, depth = 5, M=10, N=50, criterion = "entropy")
axs[1,0].plot(n_samp, train_time)
axs[1,0].set_title('Real - Discrete')
train_time, test_time, n_samp = feature_run_time(input_Dtype = 'Real', output_Dtype = 'Discrete', test_size=0.5, depth = 5, M=10, N=50, criterion = "entropy")
axs[1,1].plot(n_samp, train_time)
axs[1,1].set_title('Real - Discrete')

train_time, test_time, n_samp = sample_run_time(input_Dtype = 'Discrete', output_Dtype = 'Real', test_size=0.5, depth = 5, M=10, N=50, criterion = "variance")
axs[2,0].plot(n_samp, train_time)
axs[2,0].set_title('Discrete - Real')
train_time, test_time, n_samp = feature_run_time(input_Dtype = 'Discrete', output_Dtype = 'Real', test_size=0.5, depth = 5, M=10, N=50, criterion = "variance")
axs[2,1].plot(n_samp, train_time)
axs[2,1].set_title('Discrete - Real')

train_time, test_time, n_samp = sample_run_time(input_Dtype = 'Discrete', output_Dtype = 'Discrete', test_size=0.5, depth = 5, M=10, N=50, criterion = "entropy")
axs[3,0].plot(n_samp, train_time)
axs[3,0].set_title('Discrete - Discrete')
train_time, test_time, n_samp = feature_run_time(input_Dtype = 'Discrete', output_Dtype = 'Discrete', test_size=0.5, depth = 5, M=10, N=50, criterion = "entropy")
axs[3,1].plot(n_samp, train_time)
axs[3,1].set_title('Discrete - Discrete')

plt.setp(axs[:,1], xlabel="Features")
plt.setp(axs[:,0], xlabel="Samples")
plt.setp(axs[:,:], ylabel="Time(Sec)")
fig.tight_layout()
