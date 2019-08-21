#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#加载数据集


# In[3]:


data=pd.read_csv('D:/kaggle/cancer/data.csv')


# In[4]:


#因为数据集中列比较多，需要把dataframe中的列全部显示出来
pd.set_option('display.max_columns',None)


# In[5]:


data.columns


# In[6]:


data.head()


# In[7]:


data.describe()


# In[8]:


# 将特征字段分成三组
features_mean=list(data.columns[2:12])
features_se=list(data.columns[12:22])
features_worst=list(data.columns[22:32])


# In[9]:


#数据清洗


# In[10]:


data.drop('id',axis=1,inplace=True)


# In[11]:


#将B良性替换为0，M恶性替换为1
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})


# In[12]:


#将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'],label='Count')


# In[13]:


#用热力图呈现features_mean字段之间的相关性
corr=data[features_mean].corr()


# In[14]:


plt.figure(figsize=(14,14))


# In[15]:


sns.heatmap(corr,annot=True)#annot=True显示每个方格的数据


# In[16]:


#特征选择    特征选择的目的是降维，用少量的特征代表数据的特性，这样可以增强分类器的泛化能力，避免数据过度拟合
features_remain=['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean']


# In[17]:


#抽取30%的数据作为测试集，其余作为训练集
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)
train_X=train[features_remain]
train_Y=train['diagnosis']
teat_X=test[features_remain]
test_Y=test['diagnosis']


# In[18]:


#采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
train_X=ss.fit_transform(train_X)
teat_X=ss.fit_transform(teat_X)
#创建SVM分类器
from sklearn import svm
model=svm.SVC()


# In[19]:


#用训练集做预测
model.fit(train_X,train_Y)


# In[20]:


#用测试集做预测
prediction=model.predict(teat_X)
from sklearn.metrics import accuracy_score
print('准确率: ', accuracy_score(prediction,test_Y))


# In[ ]:




