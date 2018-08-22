# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# test数据集处理保存：
# data = pd.read_csv('./tmp/test.csv',header=0,index_col=0) #,nrows=100000# 我们可以从nrows这个参数来设置数据大小
# print(data.shape,'\n',data.columns)
# data.drop(['SearchQuery','SearchParams','Params'],axis=1,inplace=True)
# data['HistCTR']=data['HistCTR'].fillna(data['HistCTR'].mean())
# print(data.shape,data.isnull().sum())
# data_train = data.copy()
# data_train.to_csv("tmp/xin_test.csv")



# train数据集处理保存
data = pd.read_csv('./tmp/train.csv',header=0,index_col=0,nrows=1000000) # 我们可以从nrows这个参数来设置数据大小
print(data.shape,'\n',data.columns)
data.drop(['SearchQuery','SearchParams','Params'],axis=1,inplace=True)
data['IsClick'] = data['IsClick'].fillna(100)
li = data[data['IsClick']==100].index.tolist()
data = data.drop(li,axis=0)

print(data.shape,'\n',data.isnull().sum())
# data_train = data.copy()
# data_train.to_csv("tmp/xin_test.csv")
