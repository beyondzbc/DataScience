#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#读取数据


# In[3]:


data=pd.read_csv('D:/kaggle/kobe/data.csv')


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.head()


# In[8]:


#保留标签不为缺失值的数据,有缺失值數據作為測試數據


# In[9]:


kobe=data[pd.notnull(data['shot_made_flag'])]


# In[10]:


kobe.shape


# In[11]:


#分配画布大小
#alpha值表示透明程度
plt.figure(figsize=(10,10))
#loc_x and loc_y
plt.subplot(1,2,1)
plt.scatter(kobe.loc_x,kobe.loc_y,color='g',alpha=0.02)
plt.title('loc_x and loc_y')
#lat and lon
plt.subplot(1,2,2)
plt.scatter(kobe.lon,kobe.lat,color='b',alpha=0.02)
plt.title('lat and lon')


# In[12]:


data['dist']=np.sqrt(data['loc_x']**2 + data['loc_y']**2)

loc_x_zero =data['loc_x'] == 0
#print (loc_x_zero)
data['angle'] = np.array([0]*len(data))
data['angle'][~loc_x_zero] = np.arctan(data['loc_y'][~loc_x_zero] / data['loc_x'][~loc_x_zero])
data['angle'][loc_x_zero] = np.pi / 2 


# In[13]:


data['remain_time']=data['minutes_remaining']*60+data['seconds_remaining']


# In[14]:


#显示某一特征里的独一无二值


# In[15]:


print(kobe['shot_type'].unique())
print(kobe['shot_type'].value_counts())
print(kobe['season'].unique())
print(kobe['team_id'].unique())
print(kobe['team_name'].unique())


# In[16]:


data['season']=data['season'].apply(lambda x:int(x.split('-')[1]))
data['season'].unique()


# In[17]:


pd.DataFrame({'matchup':kobe.matchup,'opponent':kobe.opponent})


# In[18]:


plt.figure(figsize=(5,5))

plt.scatter(data.dist, data.shot_distance, color='blue')
plt.title('dist and shot_distance')


# In[19]:


gs = kobe.groupby('shot_zone_area')
print (kobe['shot_zone_area'].value_counts())
print (len(gs))


# In[20]:


import matplotlib.cm as cm


# In[21]:


plt.figure(figsize=(20,10))
def scatterbygroupby(feature):
    alpha=0.1
    gb=data.groupby(feature)
    cl=cm.rainbow(np.linspace(0,1,len(gb)))
    for g,c in zip(gb,cl):
        plt.scatter(g[1].loc_x,g[1].loc_y,color=c,alpha=alpha)
        
plt.subplot(1,3,1)
scatterbygroupby('shot_zone_basic')

plt.subplot(1,3,2)
scatterbygroupby('shot_zone_range')

plt.subplot(1,3,3)
scatterbygroupby('shot_zone_area')


# In[22]:


drops=['shot_id','team_id','team_name','shot_zone_area','shot_zone_range', 'shot_zone_basic',          'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining',          'shot_distance', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    data= data.drop(drop, 1)


# In[23]:


print (data['combined_shot_type'].value_counts())
pd.get_dummies(data['combined_shot_type'], prefix='combined_shot_type')[0:2]


# In[24]:


categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
 #使用one-hot编码，将a中的特征里的属性值都当作新的特征附在数据的列上，特征名为前缀prefix加上该属性名
for var in categorical_vars:
    data = pd.concat([data, pd.get_dummies(data[var], prefix=var)], 1)
    data = data.drop(var, 1)


# In[25]:


train_kobe = data[pd.notnull(data['shot_made_flag'])]


# In[26]:


train_label = train_kobe['shot_made_flag']


# In[27]:


train_kobe = train_kobe.drop('shot_made_flag', 1)


# In[28]:


test_kobe = data[pd.isnull(data['shot_made_flag'])]
test_kobe = test_kobe.drop('shot_made_flag', 1)


# In[29]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix,log_loss
import time


# In[30]:


import numpy as np
range_m = np.logspace(0,2,num=5).astype(int)
range_m


# In[31]:


# 找出隨機森林分類最優n_estimators
print('Finding best n_estimators for RandomForestClassifier...')
min_score = 100000
best_n = 0
scores_n = []
range_n = np.logspace(0,2,num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()
    
    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    k_fold=KFold(n_splits=10,shuffle=True)
    for train_k, test_k in k_fold.split(train_kobe,train_label):
        rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
        #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train_kobe.iloc[test_k])
        rfc_score += log_loss(train_label.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n
        
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2-t1))
print(best_n, min_score)


#  找出隨機森林分類max_depth
print('Finding best max_depth for RandomForestClassifier...')
min_score = 100000
best_m = 0
scores_m = []
range_m = np.logspace(0,2,num=3).astype(int)
for m in range_m:
    print("the max depth : {0}".format(m))
    t1 = time.time()
    
    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    k_fold=KFold(n_splits=10,shuffle=True)
    for train_k, test_k in k_fold.split(train_kobe,train_label):
        rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
        #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train_kobe.iloc[test_k])
        rfc_score += log_loss(train_label.iloc[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m
    
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2-t1))
print(best_m, min_score)


# In[32]:


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(range_n, scores_n)
plt.ylabel('score')
plt.xlabel('number of trees')

plt.subplot(122)
plt.plot(range_m, scores_m)
plt.ylabel('score')
plt.xlabel('max depth')


# In[33]:


model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(train_kobe, train_label)


# In[34]:


pred=model.predict(test_kobe)


# In[36]:


result=pd.read_csv('D:kaggle/kobe/sample_submission.csv')
result['shot_made_flag']=pred
result.to_csv('D:kaggle/kobe/result.csv',index=False)


# In[ ]:




