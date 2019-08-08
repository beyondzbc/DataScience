#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


#加载数据集


# In[4]:


data=pd.read_csv('D:/kaggle/cancer/data.csv')


# In[5]:


from pandas import DataFrame,Series


# In[6]:


#因为数据集中列比较多，需要把dataframe中的列全部显示出来


# In[7]:


pd.set_option('display.max_columns',None)


# In[8]:


print(data.columns)


# In[9]:


print(data.head(5))


# In[10]:


print(data.describe())


# In[11]:


#将特征字段分成三组


# In[12]:


features_mean=list(data.columns[2:12])


# In[13]:


features_se=list(data.columns[12:22])


# In[14]:


features_worst=list(data.columns[22:32])


# In[15]:


#数据清洗


# In[16]:


#ID没有用，删除


# In[17]:


data.drop('id',axis=1,inplace=True)


# In[18]:


#将B良性替换为0，M恶性替换为1


# In[19]:


data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})


# In[20]:


#将肿瘤诊断结果可视化


# In[21]:


import seaborn as sns


# In[22]:


sns.countplot(data['diagnosis'],label='Count')


# In[23]:


import matplotlib.pyplot as plt


# In[24]:


plt.show()


# In[25]:


#用热力图呈现features_mean字段之间的相关性


# In[26]:


corr=data[features_mean].corr()


# In[27]:


plt.figure(figsize=(14,14))


# In[28]:


#annot=True显示每个方格的数据


# In[29]:


sns.heatmap(corr,annot=True)


# In[30]:


#特征选择    特征选择的目的是降维，用少量的特征代表数据的特性，这样可以增强分类器的泛化能力，避免数据过度拟合


# In[31]:


features_remain=['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean']


# In[32]:


#抽取30%的数据作为测试集，其余作为训练集


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


train,test=train_test_split(data,test_size=0.3)


# In[35]:


#抽取特征选择的数值作为训练和测试数据


# In[36]:


train_X=train[features_remain]


# In[37]:


train_Y=train['diagnosis']


# In[38]:


teat_X=test[features_remain]


# In[39]:


test_Y=test['diagnosis']


# In[40]:


#采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1


# In[41]:


from sklearn.preprocessing import StandardScaler


# In[42]:


ss=StandardScaler()


# In[43]:


train_X=ss.fit_transform(train_X)


# In[44]:


teat_X=ss.fit_transform(teat_X)


# In[45]:


#创建SVM分类器


# In[46]:


from sklearn import svm


# In[47]:


model=svm.SVC()


# In[48]:


#用训练集做预测


# In[49]:


model.fit(train_X,train_Y)


# In[50]:


#用测试集做预测


# In[51]:


prediction=model.predict(teat_X)


# In[52]:


from sklearn.metrics import accuracy_score


# In[54]:


print('准确率: ', accuracy_score(prediction,test_Y))


# In[ ]:




