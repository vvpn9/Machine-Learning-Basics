#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv('churn.csv')
col_names = df.columns.tolist()
print('Column names:')
print(col_names)
print(df.shape)


# In[2]:


df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df['Churn?'].value_counts()


# In[6]:


df['CustServ Calls'].value_counts()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((2,3),(0,0))
df['Churn?'].value_counts().plot(kind='bar')
plt.title('Customer Loss Percent')
plt.ylabel('Number')
plt.xlabel('Whether loss or not')
plt.subplot2grid((2,3),(0,2))
df['CustServ Calls'].value_counts().plot(kind='bar')
plt.title('Customer Service Call Statistics')
plt.ylabel('Call Numbers')
plt.xlabel('Frequency')
plt.show()


# In[8]:


plt.subplot2grid((2,5),(0,0))
df['Day Mins'].plot(kind='kde')
plt.xlabel('Mins')
plt.ylabel('Density')
plt.title('Dis for day mins')
plt.subplot2grid((2,5),(0,2))

df['Day Calls'].plot(kind='kde')
plt.xlabel("Call")
plt.ylabel("Density")
plt.title("Dis for day calls")
plt.subplot2grid((2,5),(0,4))

df['Day Charge'].plot(kind='kde')
plt.xlabel("Charge")
plt.ylabel("density")
plt.title("dis for day charge")
plt.show()


# In[9]:


# We can see that, all plots above show the shape of normal distribution.
# Here, kde stands for kernel density estimation


# In[10]:


# Here, we wanna discover the relationship between the Int'I Plan and Churn?


# In[11]:


int_yes = df['Churn?'][df['Int\'l Plan'] == 'yes'].value_counts()
int_yes


# In[12]:


int_no = df['Churn?'][df['Int\'l Plan'] == 'no'].value_counts()
int_no


# In[13]:


type(int_yes)


# In[14]:


df_int=pd.DataFrame({u'int plan':int_yes, u'no int plan':int_no})
df_int


# In[15]:


# Then, if we want to visualize the crosstale above
df_int.plot(kind='bar', stacked=True)
plt.title('International Long Distance Plan Whether Lost or Not')
plt.xlabel('Int or not')
plt.ylabel('Number')
plt.show()


# In[16]:


cus_0 = df['CustServ Calls'][df['Churn?'] == 'False.'].value_counts()
cus_1 = df['CustServ Calls'][df['Churn?'] == 'True.'].value_counts()
df_cus=pd.DataFrame({u'churn':cus_1,u'retain':cus_0})
df_cus


# In[17]:


df_cus.plot(kind='bar',stacked=True)
plt.title('Relationship between Customer Service Complaint Phone Number and'
          'Customer Churn Rate')
plt.xlabel('Call Service')
plt.ylabel('Num')
plt.show()


# In[18]:


# So data is not suitable for further analysis, so we need to tranform
# them into 0/1 type

ds_result = df['Churn?']
y = np.where(ds_result == 'True.',1,0) # here not 1.0, but 1, 0
dummies_int = pd.get_dummies(df["Int\'l Plan"], prefix="_Int\' Plan")
dummies_voice = pd.get_dummies(df['VMail Plan'], prefix='VMail')
ds_tmp=pd.concat([df, dummies_int, dummies_voice], axis=1)
ds_tmp.head()


# In[19]:


y


# In[20]:


to_drop = ['State','Area Code','Phone','Churn?', 'Int\'l Plan', 'VMail Plan']
df = ds_tmp.drop(to_drop,axis=1)


# In[21]:


print("after trans: ")
df.head(5)


# In[22]:


X = df.values.astype(np.float)
X


# In[23]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# In[24]:


X.shape


# In[25]:


print("Feature space holds %d observations and %d features" % X.shape)


# In[27]:


print("Unique target labels:", np.unique(y))


# In[28]:


print(X[0])


# In[34]:


y[0]


# In[ ]:




