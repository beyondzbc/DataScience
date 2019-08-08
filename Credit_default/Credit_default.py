#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding:utf_8 -*-


# In[2]:


#信用卡违约率分析


# In[3]:


import pandas as pd


# In[4]:


from sklearn.model_selection import learning_curve,train_test_split,GridSearchCV


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


from sklearn.pipeline import Pipeline


# In[7]:


from sklearn.metrics import accuracy_score


# In[8]:


from sklearn.svm import SVC


# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[11]:


from sklearn.neighbors import KNeighborsClassifier


# In[12]:


from matplotlib import pyplot as plt


# In[13]:


import seaborn as sns


# In[14]:


#数据加载


# In[15]:


data=pd.read_csv('D:/kaggle/credit_default/UCI_Credit_Card.csv')


# In[16]:


#数据探索


# In[17]:


data.shape


# In[18]:


data.describe()


# In[19]:


#查看下个月违约率情况


# In[20]:


next_month=data['default.payment.next.month'].value_counts()


# In[21]:


next_month


# In[22]:


df=pd.DataFrame({'default.payment.next.month':next_month.index,'values':next_month.values})


# In[23]:


plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文 标签


# In[24]:


plt.figure(figsize=(6,6))


# In[25]:


plt.title('信用卡违约率客户\n (违约:1,守约:0)')


# In[26]:


sns.set_color_codes('pastel')


# In[27]:


sns.barplot(x='default.payment.next.month',y='values',data=df)


# In[28]:


#特征选择，去掉ID字段、最后一个结果字段


# In[29]:


data.drop(['ID'],inplace=True,axis=1)


# In[30]:


target=data['default.payment.next.month'].values


# In[31]:


columns=data.columns.tolist()


# In[32]:


columns.remove('default.payment.next.month')


# In[33]:


features=data[columns].values


# In[34]:


#30%作为测试集，其余为训练集


# In[35]:


train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=0.3,stratify=target,random_state=1)


# In[36]:


#构造各种分类器


# In[37]:


classifiers=[
    SVC(random_state=1,kernel='rbf'),
    DecisionTreeClassifier(random_state=1,criterion='gini'),
    RandomForestClassifier(random_state=1,criterion='gini'),
    KNeighborsClassifier(metric='minkowski'),
]


# In[38]:


#分类器名称


# In[39]:


classifier_names=[
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier',
]


# In[40]:


#分类器参数


# In[41]:


classifier_param_grid=[
    {'svc__C':[1],'svc__gamma':[0.01]},
    {'decisiontreeclassifier__max_depth':[6,9,11]},
    {'randomforestclassifier__n_estimators':[3,5,6]},
    {'kneighborsclassifier__n_neighbors':[4,6,8]},
]


# In[42]:


#对具体的分类器进行GridSearchCV参数调优


# In[43]:


def GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,param_grid,score='accuracy'):
    response={}
    gridsearch=GridSearchCV(estimator=pipeline,param_grid=param_grid,scoring=score)
    #寻找最优的参数和最优的准确率分数
    search=gridsearch.fit(train_x,train_y)
    print('GridSearchCV最优参数:',search.best_params_)
    print('GridSearchCV最优分数: %.4lf' %search.best_score_)
    predict_y=gridsearch.predict(test_x)
    print('准确率 %.4lf' %accuracy_score(test_y,predict_y))
    response['predict_y']=predict_y
    response['accuracy_score']=accuracy_score(test_y,predict_y)
    return response


# In[44]:


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')


# In[ ]:




