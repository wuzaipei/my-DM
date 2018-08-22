# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

data = pd.read_csv('./tmp/train.csv',header=0,index_col=0,nrows=30000) # 我们可以从nrows这个参数来设置数据大小

data_copy = data.copy()
print(data.shape,data.columns)

data_copy['Price'][data_copy['Price']>0.1*1e8] = 0.1*1e8

plt.plot(data_copy['Price'])
plt.show()


# 对数据进行处理
#  searchid 用均值 IsUserLoggedOn用0来替换 均值：LocationID_x，CategoryID_x 文本：SearchParams 字典：Params
data.drop(['Price','Title','IsContext'],axis=1,inplace=True) #删除不需要使用one_hot表示的列
data['SearchID'] = data['SearchID'].fillna(data['SearchID'].mean()) # SearchID使用均值来填空值
data['IsUserLoggedOn'] = data['IsUserLoggedOn'].fillna(0)           # IsUserLoggedOn使用0来填空值

print(data.shape,data.columns)

# # 对其进行one编码
# x_train=pd.DataFrame()
# for i in data.columns:
#     data_test = pd.get_dummies(data[i], prefix=i)
#     x_train = pd.concat((x_train, data_test), axis=1)
#
# y_train = data_copy['Price']
# v_train =[]
# print(data_dummies.shape)
# 保存编码过后的one_hot数据
# data_dummies.to_csv('tmp/train_one_hot.csv')

# from jiangzilong.DeepFM import DeepFM
#
# dfm = DeepFM(feature_size=80,field_size=40,embedding_size=10)
# dfm.fit(Xi_train=x_train,Xv_train=v_train,y_train=y_train)
# dfm.predict()
# loss = dfm.fit_on_batch
