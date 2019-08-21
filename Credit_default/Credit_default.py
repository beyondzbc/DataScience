#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import learning_curve,train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('D:/kaggle/credit_default/UCI_Credit_Card.csv')
data.shape
data.describe()


# In[3]:


#查看下个月违约率情况
next_month=data['default.payment.next.month'].value_counts()
next_month


# In[4]:


df=pd.DataFrame({'default.payment.next.month':next_month.index,'values':next_month.values})
plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文 标签


# In[5]:


plt.figure(figsize=(6,6))
plt.title('信用卡违约率客户\n (违约:1,守约:0)')
sns.set_color_codes('pastel')
sns.barplot(x='default.payment.next.month',y='values',data=df)


# In[6]:


#特征选择，去掉ID字段、最后一个结果字段
data.drop(['ID'],inplace=True,axis=1)
target=data['default.payment.next.month'].values
columns=data.columns.tolist()
columns.remove('default.payment.next.month')
features=data[columns].values


# In[7]:


#30%作为测试集，其余为训练集
train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=0.3,stratify=target,random_state=1)


# In[8]:


#构造各种分类器
classifiers=[
    SVC(random_state=1,kernel='rbf'),
    DecisionTreeClassifier(random_state=1,criterion='gini'),
    RandomForestClassifier(random_state=1,criterion='gini'),
    KNeighborsClassifier(metric='minkowski'),
]


# In[9]:


#分类器名称
classifier_names=[
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier',
]


# In[10]:


#分类器参数
classifier_param_grid=[
    {'svc__C':[1],'svc__gamma':[0.01]},
    {'decisiontreeclassifier__max_depth':[6,9,11]},
    {'randomforestclassifier__n_estimators':[3,5,6]},
    {'kneighborsclassifier__n_neighbors':[4,6,8]},
]


# In[11]:


#对具体的分类器进行GridSearchCV参数调优
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


# In[12]:


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')


# In[ ]:




