#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding:utf-8 -*-


# In[2]:


#用PageRank挖掘希拉里邮箱中的重要任务关系


# In[3]:


import pandas as pd


# In[4]:


import networkx as nx


# In[5]:


import numpy as np


# In[6]:


from collections import defaultdict


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


#数据加载


# In[9]:


emails=pd.read_csv('D:/kaggle/email/emails.csv')


# In[10]:


#读取别名文件


# In[11]:


file=pd.read_csv('D:/kaggle/email/Aliases.csv')


# In[12]:


aliases={}


# In[13]:


for index,row in file.iterrows():
    aliases[row['Alias']]=row['PersonId']


# In[14]:


#读取人名文件


# In[15]:


file=pd.read_csv('D:/kaggle/email/Persons.csv')


# In[16]:


persons={}


# In[17]:


for index,row in file.iterrows():
    persons[row['Id']]=row['Name']


# In[18]:


#针对别名进行转换


# In[19]:


def unity_name(name):
    #姓名统一小写
    name=str(name).lower()
    #去掉，和@后面的内容
    name=name.replace(',','').split('@')[0]
    #别名转换
    if name in aliases.keys():
        return persons[aliases[name]]
    return name


# In[20]:


#画网络图


# In[21]:


def show_graph(graph):
    #使用spring layout布局，类似中心放射状
    positions=nx.spring_layout(graph)
    #设置网络图中的节点大小，大小与PageRank值相关，因为PageRank值很小，所以需要*2000
    nodesize=[x['pagerank']*2000 for v,x in graph.nodes(data=True)]
    #设置网络图中的边长度
    edgesize=[np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
    #绘制节点
    nx.draw_networkx_nodes(graph,positions,node_size=nodesize,alpha=0.4)
    #绘制边
    nx.draw_networkx_edges(graph,positions,edge_size=edgesize,alpha=0.2)
    #绘制节点的label
    nx.draw_networkx_labels(graph,positions,font_size=10)
    #输出希拉里邮件中的所有任务关系图
    plt.show()


# In[22]:


#将寄件人和收件人的姓名进行规范化


# In[23]:


emails.MetadataFrom=emails.MetadataFrom.apply(unity_name)


# In[24]:


emails.MetadataTo=emails.MetadataTo.apply(unity_name)


# In[25]:


#设置边的权重等于发邮件的次数


# In[26]:


edges_weights_temp=defaultdict(list)


# In[27]:


for row in zip(emails.MetadataFrom,emails.MetadataTo,emails.RawText):
    temp=(row[0],row[1])
    if temp not in edges_weights_temp:
        edges_weights_temp[temp]=1
    else:
        edges_weights_temp[temp]=edges_weights_temp[temp]+1


# In[28]:


#转换格式（from,to）,weight=>from,to,weight


# In[29]:


edges_weights=[(key[0],key[1],val) for key ,val in edges_weights_temp.items()]


# In[30]:


#创建一个有向图


# In[31]:


graph=nx.DiGraph()


# In[32]:


#设置有向图中的路径和权重（from，to，weight）


# In[33]:


graph.add_weighted_edges_from(edges_weights)


# In[34]:


#计算每个节点（人）的PR值，并作为节点的PageRank属性


# In[35]:


pagerank=nx.pagerank(graph)


# In[36]:


#获取每个节点的PageRank数值


# In[37]:


pagerank_list={node:rank for node,rank in pagerank.items()}


# In[38]:


#将pagerank数值作为节点的属性


# In[39]:


nx.set_node_attributes(graph,name='pagerank',values=pagerank_list)


# In[40]:


#画网络图


# In[41]:


show_graph(graph)


# In[43]:


#将完整的图谱进行精简


# In[44]:


#设置PR值的阈值，筛选大于阈值的重要核心节点


# In[45]:


pagerank_threshold=0.005


# In[46]:


#复制一份计算好的网络图


# In[47]:


small_graph=graph.copy()


# In[48]:


#剪掉PR值小于pagerank_threshold的节点


# In[49]:


for n,p_rank in graph.nodes(data=True):
    if p_rank['pagerank']<pagerank_threshold:
        small_graph.remove_node(n)


# In[50]:


#画网络图


# In[51]:


show_graph(small_graph)


# In[ ]:




