#!/usr/bin/env python
# coding: utf-8

# Nguyễn Như Quỳnh - 11194482

# Ex 2

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[3]:


dataset = pd.read_csv('data_linear.csv')
dataset


# In[7]:


x = np.array(dataset['Diện tích'])
y = np.array(dataset['Giá'])


# In[6]:


x


# In[8]:


y


# In[9]:


model = LinearRegression()


# In[11]:


x = x.reshape(-1, 1)


# In[12]:


model.fit(x,y)


# In[13]:


y_predict = model.predict(x)


# In[14]:


# Vẽ model dự đoán 
plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()


# In[16]:


# Dự đoán giá các căn nhà có diện tích 50, 100, 150
new_x = np.array([50, 100, 150])
new_x = new_x.reshape(-1, 1)
model.predict(new_x)


# Ex 3

# In[18]:


header_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
              'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT', 'PRICE']


# In[19]:


housing = pd.read_csv('housing.csv', names=header_names, delim_whitespace=True)
housing.head()


# In[21]:


housing.info()


# In[23]:


housing.isnull().sum()


# In[25]:


X = housing.drop('PRICE', axis=1)
X


# In[26]:


Y = housing['PRICE']
Y


# In[27]:


linear = LinearRegression()


# In[28]:


linear.fit(X, Y)


# In[30]:


Y_pred = linear.predict(X)

