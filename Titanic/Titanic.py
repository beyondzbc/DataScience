#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#导入数据


# In[3]:


train_data=pd.read_csv('D:/kaggle/Titanic/train.csv')


# In[4]:


test_data=pd.read_csv('D:/kaggle/Titanic/test.csv')


# In[5]:


#数据探索


# In[6]:


#了解数据表的基本情况：行数、列数、每列的数据类型、数据完整度


# In[7]:


print(train_data.info())


# In[8]:


#了解数据表的统计情况：总数、平均值、标准差、最小值、最大值


# In[9]:


print(train_data.describe())


# In[10]:


#查看字符串类型（非数字）的整体情况


# In[11]:


print(train_data.describe(include=['O']))


# In[12]:


#查看前几行数据（默认是前5行）


# In[13]:


print(train_data.head())


# In[14]:


#查看后几行数据（默认是后5行）


# In[15]:


print(test_data.tail())


# In[16]:


#使用平均年龄来填充年龄中的NAN值


# In[17]:


train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)


# In[18]:


test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)


# In[19]:


#使用票价的均值填充票价的NAN值


# In[20]:


train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)


# In[21]:


test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)


# In[22]:


#Cabin有大量的缺失值，无法补齐；Embarked有少量的缺失值，可以补齐


# In[23]:


print(train_data['Embarked'].value_counts())


# In[24]:


#使用登录最多的港口来填充登录港口的nan值


# In[25]:


train_data['Embarked'].fillna('S',inplace=True)


# In[26]:


test_data['Embarked'].fillna('S',inplace=True)


# In[27]:


#特征选择


# In[28]:


features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']


# In[29]:


train_features=train_data[features]


# In[30]:


train_labels=train_data['Survived']


# In[31]:


test_features=test_data[features]


# In[32]:


#将特征值中的字符串转换成数值类型


# In[33]:


#选用sklearn特征选择中的DictVectorizer类，将符号转换成数字0/1表示


# In[34]:


from sklearn.feature_extraction import DictVectorizer


# In[35]:


dvec=DictVectorizer(sparse=False)


# In[36]:


train_features=dvec.fit_transform(train_features.to_dict(orient='record'))


# In[37]:


#fit_transform这个函数可以将特征向量转化为特征值矩阵


# In[38]:


print(dvec.feature_names_)


# In[39]:


#使用ID3算法


# In[60]:


from sklearn.tree import DecisionTreeClassifier


# In[41]:


#构造ID3决策树


# In[42]:


clf=DecisionTreeClassifier(criterion='entropy')


# In[43]:


test_features=dvec.fit_transform(test_features.to_dict(orient='record'))


# In[44]:


#决策树训练


# In[45]:


clf.fit(train_features,train_labels)


# In[46]:


#决策树预测


# In[47]:


pred_labels=clf.predict(test_features)


# In[48]:


#得到决策树准确率    ----决策树提供了score函数可以直接得到准确率


# In[49]:


acc_decision_tree=round(clf.score(train_features,train_labels),6)


# In[50]:


print(u'score准确率为%.4lf' %acc_decision_tree)


# In[51]:


#交叉验证    ----sklearn的model_selection模型中提供了cross_val_score函数


# In[52]:


import numpy as np


# In[53]:


from sklearn.model_selection import cross_val_score


# In[54]:


#使用K折交叉验证   统计决策树准确率


# In[55]:


print(u'cross_val_score准确率为 %.4lf' %np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))


# In[56]:


PassengerId=np.array(test_data['PassengerId']).astype(int)


# In[57]:


predictions=pd.DataFrame(pred_labels,PassengerId,columns=['Survived'])


# In[58]:


predictions.to_csv('D:/kaggle/Titanic/predictions.csv',index_label=['PassengerId'])


# In[ ]:




