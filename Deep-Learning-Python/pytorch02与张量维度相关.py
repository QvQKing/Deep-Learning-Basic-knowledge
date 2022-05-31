#!/usr/bin/env python
# coding: utf-8

# In[4]:


# 与张量维度相关的方法
import torch
t = torch.randn(3,4,5) # 产生一个3×4×5的张量
t.ndimension() # 获取维度的数目


# In[5]:


t.nelement() # 获取该张量的总元素数目


# In[6]:


t.size() # 获取该张量每个维度的大小，调用方法


# In[7]:


t.shape # 获取该张量每个维度的大小，访问属性


# In[8]:


t.size(0) # 获取该张量维度0的大小，调用方法


# In[12]:


t = torch.randn(12) # 产生大小为12的向量
t.view(3, 4) # 向量改变形状为3×4的矩阵


# In[13]:


t.view(4, 3) # 向量改变形状为4×3的矩阵


# In[14]:


t.view(-1, 4) # 第一个维度为-1，PyTorch会自动计算该维度的具体值


# In[24]:


# view方法不改变底层数据，改变view后张量会改变原来的张量
t.view(4, 3)[0, 0] = 1.0
t.data_ptr() # 获取张量的数据指针


# In[25]:


t.view(3,4).data_ptr() # 数据指针不改变


# In[26]:


t.view(4,3).data_ptr() # 同上，不改变


# In[27]:


t.view(3,4).contiguous().data_ptr() # 同上，不改变


# In[28]:


t.view(4,3).contiguous().data_ptr() # 同上，不改变


# In[52]:


t.view(3,4).transpose(0,1).data_ptr() # transpose方法交换两个维度的步长


# In[60]:


t.view(4,3).transpose(0,1).contiguous().data_ptr() # 步长和维度不兼容，重新生成张量


# In[41]:


t.reshape(2,6)


# In[42]:


t.reshape(2,6).data_ptr()

