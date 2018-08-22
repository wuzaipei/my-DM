# -*- coding:utf-8 -*-

import os,re
import pandas as pd
# 封装读取tsv的方法
def read_tsv(file_path):
    train=[];name=[]
    path_tsv = os.listdir(file_path)
    for i in path_tsv:
        if i.endswith('.tsv'): #re.findall("\.tsv$",i)==['.tsv']
            index_head = i.split('.tsv')[0]
            data = pd.read_csv(file_path+i,sep='\t',nrows=1700000,header=0,encoding='utf-8') #,index_col=0
            if len(data.iloc[:,1]) == 1700000:
                train.append(data)
                name.append(index_head)
    return train,name

x_train,name =read_tsv(r'./date/')

print(x_train[1].columns,print(),name)
# 把具有相同列的表合拼起来使用 pd.merge
x_train_1 = pd.merge(x_train[1],x_train[2],on='UserID',how='left') #,x_train[5],x_train[6],
x_train_2 = pd.merge(x_train_1,x_train[5],on='UserID',how='left')
x_train_3 = pd.merge(x_train_2,x_train[6],on='UserID',how='left')
x_train4 = pd.merge(x_train_1,x_train[0],on='AdID')
X_train = pd.merge(x_train4,x_train[3],on='SearchID') #3

print(X_train.shape,X_train.columns)  #(183099, 23)


# 保存合并过后的表格式为 train.csv
# X_train.to_csv("tmp/test.csv")
