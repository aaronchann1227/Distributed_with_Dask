
import joblib
import pandas as pd
import numpy as np
import math
from collections import Counter

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import time
import ctypes

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor as rf
###############################################################
def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)
###############################################################

n_node = 4
n_processes = 3

cluster = SLURMCluster(cores = 90, queue = 'cpu', memory = '100GB', walltime = '24:00:00', processes = n_processes)
cluster.scale(jobs = n_node)

client = Client(cluster)
print(client)

client.run(trim_memory)

ready = False
while (ready == False):
    time.sleep(3)
    if (len(client.ncores()) == n_node * n_processes):
        ready = True
        print('READY TO GO')
        
###############################################################     
data = pd.read_csv("LB_index_01_train.csv",low_memory=False)

X = data.drop(columns=['log_ratio_residual'])
y = data['log_ratio_residual']

from sklearn.impute import SimpleImputer
y = np.array(y)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X)
X_I = imp.transform(X)

X = X_m.to_numpy()
X = X.astype(np.float32)
y = y.astype(np.float32)
print('DATA READY')
###############################################################

params = {'n_estimators': [800, 900, 1000, 1200, 1500], # Number of trees in random forest
         'criterion': ['squared_error', 'absolute_error'],
         'max_depth': [27, 28, 29, 30], # Maximum number of levels in tree
         'min_samples_split': [2, 4, 6, 8, 10]}

with joblib.parallel_backend('dask'):
    clf = rf(n_jobs = -1)
    search = GridSearchCV(clf, params, cv = 10, n_jobs = -1)
    start = time.time()
    search.fit(X, y)
    end = time.time()
    result = pd.DataFrame({'param': search.cv_results_["params"], 'score': search.cv_results_["mean_test_score"]})
    result.to_csv('GridSearchTestResult.csv',index=False, sep=',')
    
    s = f"\n--- {end - start}s total 24 combinations---"
    f = open('GridSearchTestResult.txt', 'at', encoding='utf-8')
    f.write(s)
    f.close()
