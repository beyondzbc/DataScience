#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding:utf-8 -*-


# In[2]:


#使用逻辑回归对信用卡欺诈进行分类


# In[3]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


import seaborn as sns


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import itertools


# In[8]:


from sklearn.linear_model import LogisticRegression


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


from sklearn.metrics import confusion_matrix,precision_recall_curve


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


import warnings


# In[13]:


warnings.filterwarnings('ignore')


# In[14]:


#混淆矩阵可视化


# In[15]:


def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0)
    plt.yticks(tick_marks,classes)
    
    thresh=cm.max()/2.
    for i ,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(i,j,cm[i,j],
                horizontalalignment='center',
                color='white' if cm[i,j]>thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[16]:


#显示模型评估结果


# In[17]:


def show_metrics():
    tp=cm[1,1]
    fn=cm[1,0]
    fp=cm[0,1]
    tn=cm[0,0]
    print('精确率:{:.3f}'.format(tp/(tp+fp)))
    print('召回率:{:.3f}'.format(tp/(tp+fn)))
    print('F1值:{:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))))


# In[18]:


#绘制精确率-召回率曲线


# In[19]:


def plot_precision_recall():
    plt.step(recall,precision,color='b',alpha=0.2,where='post')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    plt.plot(recall,precision,linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率 曲线')
    plt.show()


# In[20]:


#数据加载


# In[21]:


data=pd.read_csv('D:/kaggle/credit_fraud/creditcard.csv')


# In[22]:


data.describe()


# In[23]:


plt.rcParams['font.sans-serif']=['SimHei']


# In[24]:


plt.figure()


# In[25]:


ax=sns.countplot(x='Class',data=data)


# In[26]:


#显示交易笔数，欺诈交易笔数


# In[27]:


num=len(data)


# In[28]:


num_fraud=len(data[data['Class']==1])


# In[29]:


print('总交易笔数：',num)


# In[30]:


print('欺诈交易笔数：',num_fraud)


# In[31]:


print('诈骗交易比例:{:.6f}'.format(num_fraud/num))


# In[32]:


#欺诈和正常交易可视化


# In[33]:


f,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(15,8))
bins=50
ax1.hist(data.Time[data.Class==1],bins=bins,color='deeppink')
ax1.set_title('诈骗交易')
ax2.hist(data.Time[data.Class==0],bins=bins,color='deepskyblue')
ax2.set_title('正常交易')
plt.xlabel('时间')
plt.ylabel('交易次数')


# In[34]:


#对Amount进行数据规范化


# In[35]:


data['Amount_Norm']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))


# In[36]:


#特征选择


# In[37]:


y=np.array(data.Class.tolist())


# In[38]:


data=data.drop(['Time','Amount','Class'],axis=1)


# In[39]:


x=np.array(data.as_matrix())


# In[40]:


#准备训练集和测试集


# In[41]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.1,random_state=33)


# In[42]:


#逻辑回归分类


# In[43]:


clf=LogisticRegression()


# In[44]:


clf.fit(train_x,train_y)


# In[45]:


predict_y=clf.predict(test_x)


# In[46]:


#预测样本的置信分数


# In[47]:


score_y=clf.decision_function(test_x)


# In[48]:


#计算混淆矩阵，并显示


# In[49]:


cm=confusion_matrix(test_y,predict_y)


# In[50]:


class_names=[0,1]


# In[51]:


#显示混淆矩阵


# In[52]:


plot_confusion_matrix(cm,classes=class_names,title='逻辑回归 混淆矩阵')


# In[53]:


plot_confusion_matrix(cm,classes=class_names,title='逻辑回归 混淆矩阵')


# In[54]:


show_metrics()


# In[55]:


#计算精确率，召回率，阈值用于可视化


# In[56]:


precision,recall,thresholds=precision_recall_curve(test_y,score_y)


# In[57]:


plot_precision_recall()


# In[ ]:




