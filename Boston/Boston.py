#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import AdaBoostRegressor


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.metrics import mean_squared_error


# In[4]:


from sklearn.datasets import load_boston


# In[5]:


#加载数据


# In[6]:


data=load_boston()


# In[7]:


#分割数据


# In[8]:


train_x,test_x,train_y,test_y=train_test_split(data.data,data.target,test_size=0.3,random_state=33)


# In[9]:


#使用AdaBoost回归模型


# In[11]:


regressor=AdaBoostRegressor()


# In[12]:


regressor.fit(train_x,train_y)


# In[13]:


pred_y=regressor.predict(test_x)


# In[14]:


mse=mean_squared_error(test_y,pred_y)


# In[15]:


print('房价预测结果',pred_y)


# In[17]:


print('均方误差=',round(mse,2))


# In[18]:


#使用决策树回归模型


# In[21]:


from sklearn.tree import DecisionTreeRegressor


# In[22]:


dec_regressor=DecisionTreeRegressor()


# In[23]:


dec_regressor.fit(train_x,train_y)


# In[24]:


pred_y=dec_regressor.predict(test_x)


# In[25]:


mse=mean_squared_error(test_y,pred_y)


# In[26]:


print('决策树均方差=',round(mse,2))


# In[27]:


#使用KNN回归模型


# In[29]:


from sklearn.neighbors import KNeighborsRegressor


# In[30]:


knn_regressor=KNeighborsRegressor()


# In[31]:


knn_regressor.fit(train_x,train_y)


# In[32]:


pred_y=knn_regressor.predict(test_x)


# In[33]:


mse=mean_squared_error(test_y,pred_y)


# In[34]:


print('KNN均方误差=',round(mse,2))


# In[ ]:




