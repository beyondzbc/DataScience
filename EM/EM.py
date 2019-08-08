#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-


# In[2]:


import pandas as pd


# In[3]:


import csv


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


from sklearn.mixture import GaussianMixture


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


#数据加载，避免中文乱码问题


# In[9]:


data_ori=pd.read_csv('D:/kaggle/EM/heros.csv',encoding='gb18030')


# In[10]:


features=[u'最大生命',u'生命成长',u'初始生命',u'最大法力', u'法力成长',u'初始法力',u'最高物攻',u'物攻成长',u'初始物攻',u'最大物防',u'物防成长',u'初始物防', u'最大每5秒回血', u'每5秒回血成长', u'初始每5秒回血', u'最大每5秒回蓝', u'每5秒回蓝成长', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']


# In[11]:


data=data_ori[features]


# In[12]:


#对英雄属性之间的关系进行可视化分析


# In[13]:


#设置plt正确显示中文


# In[14]:


plt.rcParams['font.sans-serif']=['SimHei']


# In[15]:


plt.rcParams['axes.unicode_minus']=False


# In[16]:


#用热力图呈现features_mean字段之间的相关性


# In[17]:


corr=data[features].corr()


# In[18]:


plt.figure(figsize=(14,14))


# In[19]:


#annot=True显示每个方格的数据 


# In[20]:


sns.heatmap(corr,annot=True)


# In[21]:


#相关性大的属性保留一个，因此可以对属性进行降维


# In[22]:


features_remain=[u'最大生命', u'初始生命', u'最大法力', u'最高物攻', u'初始物攻', u'初始物攻', u'最大物防', u'初始物防', u'最大每5秒回血', u'最大每5秒回蓝', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']


# In[23]:


data = data_ori[features_remain]


# In[24]:


data[u'最大攻速'] = data[u'最大攻速'].apply(lambda x: float(x.strip('%'))/100)


# In[25]:


data[u'攻击范围']=data[u'攻击范围'].map({'远程':1,'近战':0})


# In[26]:


# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1


# In[27]:


ss=StandardScaler()


# In[28]:


data=ss.fit_transform(data)


# In[29]:


#构造GMM聚类


# In[30]:


gmm=GaussianMixture(n_components=30,covariance_type='full')


# In[31]:


gmm.fit(data)


# In[32]:


#训练数据


# In[33]:


prediction=gmm.predict(data)


# In[34]:


print(prediction)


# In[35]:


#将训练结果输出到CSV文件中


# In[36]:


data_ori.insert(0,'分组',prediction)


# In[37]:


data_ori.to_csv('D:/kaggle/EM/hero_out.csv',index=False,sep=',')

