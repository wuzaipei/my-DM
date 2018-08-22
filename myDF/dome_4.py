# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf


data_train = pd.read_csv('./tmp/x_train_libsvm_xin.csv',header=0,sep='\t', nrows=10000)

# print(data.shape,'\n',data.columns,'\n',data.head())

def sigmoid(x):
    import numpy as np
    return 1.0 / (1 + np.exp(-float(x)))

def split_libsvm(data):
    y_train = data.iloc[:,0]
    data.drop(data.columns[0], axis=1, inplace=True)
    n,m = data.shape
    xi_train = np.zeros((n,m))
    xv_train = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            xi_train[i,j] = data.iloc[i,j].split(":")[0]
            xv_train[i,j] = sigmoid(data.iloc[i,j].split(":")[1])
    return xi_train,xv_train,y_train

xi_train,xv_train,y_train= split_libsvm(data=data_train)

data_test = pd.read_csv('./tmp/x_test_libsvm_xin.csv',header=0,sep='\t')
xi_test,xv_test,y_test= split_libsvm(data=data_test)
print(xv_test.shape)
# print(xv_train.shape,xv_train[:5,:])

xv_train = xv_train[:10000,:]
xi_train = xi_train[:10000,:]
print(xi_train.shape)
y_train = y_train.iloc[:10000]

print(y_train.shape)

from DeepFM import DeepFM

dfm_model = DeepFM(feature_size=18,field_size=18,embedding_size=10,epoch=20)
dfm_model.fit(Xi_train=xi_train,Xv_train=xv_train,y_train=y_train)
get_predictionResult= dfm_model.predict(xi_test, xv_test)

import matplotlib.pyplot as plt

plt.scatter(range(len(get_predictionResult)),get_predictionResult)
plt.show()




# y_pred = dfm_model.predict(xi_test,xv_test)
# print(y_pred)
# print(y_pred.shape)

# print(xi_test.shape,xv_test.shape,y_test.shape)
# loss = dfm_model.fit_on_batch(xi_test,xv_test,y_test)

