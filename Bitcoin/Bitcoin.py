#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding:utf-8 -*-


# In[2]:


#比特币走势预测，使用时间序列ARMA


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


from statsmodels.tsa.arima_model import ARMA


# In[7]:


import  warnings


# In[8]:


from itertools import product


# In[9]:


from datetime import datetime


# In[10]:


warnings.filterwarnings('ignore')


# In[11]:


#数据加载


# In[12]:


df=pd.read_csv('D:/kaggle/bitcoin/bitcoin_2012-01-01_to_2018-10-31.csv')


# In[13]:


#将时间作为df的索引


# In[14]:


df.Timestamp=pd.to_datetime(df.Timestamp)


# In[15]:


df.index=df.Timestamp


# In[16]:


df.head()


# In[17]:


#按照月，季度，年来统计


# In[18]:


df_month=df.resample('M').mean()


# In[19]:


df_Q=df.resample('Q-DEC').mean()


# In[20]:


df_year=df.resample('A-DEC').mean()


# In[21]:


#按照天，月，季度，年来显示比特币的走势


# In[22]:


fig=plt.figure(figsize=[15,7])


# In[23]:


plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签


# In[24]:


plt.suptitle('比特币金额（美金）',fontsize=20)


# In[25]:


plt.subplot(221)


# In[26]:


plt.plot(df.Weighted_Price,'-',label='按天')


# In[27]:


plt.legend()


# In[28]:


plt.subplot(222)


# In[29]:


plt.plot(df_month.Weighted_Price,'-',label='按月')


# In[30]:


plt.legend()


# In[31]:


plt.subplot(223)


# In[33]:


plt.plot(df_Q.Weighted_Price,'-',label='按季度')


# In[35]:


plt.legend()


# In[36]:


plt.subplot(224)


# In[37]:


plt.plot(df_year.Weighted_Price,'-',label='按年')


# In[38]:


plt.legend()


# In[39]:


plt.show()


# In[40]:


#设置参数范围


# In[41]:


ps=range(0,3)


# In[42]:


qs=range(0,3)


# In[43]:


parameters=product(ps,qs)


# In[44]:


parameters_list=list(parameters)


# In[45]:


#寻找最优ARMA模型参数，即best_aic最小


# In[46]:


results=[]


# In[47]:


best_aic=float('inf')#正无穷


# In[48]:


for param in parameters_list:
    try:
        model=ARMA(df_month.Weighted_Price,order=(param[0],param[1])).fit()
    except ValueError:
        print('参数错误：',param)
        continue
    aic=model.aic
    if aic<best_aic:
        best_model=model
        best_aic=aic
        best_param=param
    results.append([param,model.aic])


# In[49]:


#输出最优模型


# In[50]:


results_table=pd.DataFrame(results)


# In[51]:


results_table.columns=['parameters','aic']


# In[52]:


print('最优模型：',best_model.summary())


# In[53]:


#比特币预测


# In[54]:


df_month2=df_month[['Weighted_Price']]


# In[56]:


date_list=[datetime(2018,11,30),datetime(2018,12,31),datetime(2019,1,31),datetime(2019,2,28),datetime(2019,3,31),datetime(2019,4,30),datetime(2019,5,31),datetime(2019,6,30)]


# In[57]:


future=pd.DataFrame(index=date_list,columns=df_month.columns)


# In[58]:


df_month2=pd.concat([df_month2,future])


# In[59]:


df_month2['forecast']=best_model.predict(start=0,end=91)


# In[60]:


#比特币预测结果显示


# In[61]:


plt.figure(figsize=(20,7))


# In[62]:


df_month2.Weighted_Price.plot(label='实验金额')


# In[63]:


df_month2.forecast.plot(color='r',ls='--',label='预测金额')


# In[64]:


plt.legend()


# In[65]:


plt.title('比特币金额（月）')


# In[66]:


plt.xlabel('时间')


# In[67]:


plt.ylabel('美金')


# In[68]:


plt.show()


# In[ ]:




