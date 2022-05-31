#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1.通过torch.tensor函数创建张量—Python列表和Numpy数组转换为PyTorch张量


# In[ ]:


import torch
print(torch.__version__) # 打印出当前PyTorch版本


# In[1]:


import numpy as np # 导入numpy包
import torch # 导入torch包
torch.tensor([1,2,3,4]) # 转换Python列表为PyTorch张量


# In[ ]:


torch.tensor([1,2,3,4]).dtype # 查看张量数据类型


# In[ ]:


torch.tensor([1,2,3,4], dtype=torch.float32) # 指定数据类型为32位浮点数


# In[ ]:


torch.tensor([1,2,3,4], dtype=torch.float32).dtype  # 查看张量数据类型


# In[ ]:


torch.tensor(range(10)) # 转换迭代器为张量


# In[ ]:


np.array([1,2,3,4]).dtype # 查看numpy数组类型


# In[ ]:


torch.tensor(np.array([1,2,3,4])) # 转换numpy数组为PyTorch张量


# In[ ]:


torch.tensor(np.array([1,2,3,4])).dtype # 转换后PyTorch张量的类型


# In[ ]:


torch.tensor([1.0, 2.0, 3.0, 4.0]).dtype # PyTorch默认浮点类型为32位单精度


# In[ ]:


torch.tensor(np.array([1.0, 2.0, 3.0, 4.0])).dtype # numpy默认浮点类型为64位双精度


# In[ ]:


torch.tensor([[1,2], [3,4,5]]) # 列表嵌套创建张量，错误：子列表大小不一致


# In[ ]:


torch.tensor([[1,2,3], [4,5,6]]) # 列表嵌套创建张量，正确：2×3的矩阵


# In[ ]:


torch.randn(3,3).to(torch.int) # 从torch.float 转换到 torch.int，也可以调用.int()方法


# In[ ]:


torch.randint(0, 5, (3,3)).to(torch.float) # 从torch.int64到torch.float，也可以调用.float()方法


# In[8]:


# 2.通过PyTorch内置函数创建张量
torch.rand(3,3) # 生成3×3的矩阵，矩阵元素服从[0, 1)上的均匀分布


# In[9]:


torch.randn(2,3,4) # 生成2×3×4的张量，张量元素服从标准正态分布


# In[10]:


torch.zeros(2,2,2) # 生成 2×2×2的张量，张量元素全为0


# In[11]:


torch.ones(1,2,3) # 生成1×2×3的张量，张量元素全为1


# In[12]:


torch.eye(3) # 生成3×3的单位矩阵


# In[13]:


torch.randint(0, 10, (3,3)) # 生成0（包含）到10（不含）之间均匀分布整数的3×3矩阵


# In[17]:


# 3.通过已知张量创建
t = torch.randn(3,3) # 生成一个随机正态分布的张量t
print(t)


# In[18]:


torch.zeros_like(t) # 生成一个元素全为0的张量，形状和给定张量t相同


# In[19]:


torch.ones_like(t) # 生成一个元素全为1的张量，形状和给定张量t相同


# In[20]:


torch.rand_like(t) # 生成一个元素服从[0, 1)上的均匀分布的张量，形状和给定张量t相同


# In[21]:


torch.randn_like(t) # 生成一个元素服从标准正态分布的张量，形状和给定张量t相同


# In[22]:


t.new_tensor([1,2,3]).dtype # 根据Python列表生成张量，注意这里输出的是单精度浮点数


# In[23]:


t.new_zeros(3, 3) # 生成相同类型且元素全为0的张量


# In[24]:


t.new_ones(3,3) # 生成相同类型且元素全为1的张量

