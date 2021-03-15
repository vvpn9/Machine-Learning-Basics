#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.linear_model import LogisticRegression


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.metrics import classification_report


# In[5]:


titanic_df = pd.read_csv('train.csv')


# In[6]:


titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[7]:


titanic_df.info()


# In[8]:


titanic_df.head(10)


# In[9]:


titanic_df['AgeIsMissing'] = 0


# In[10]:


titanic_df.loc[titanic_df.Age.isnull(), 'AgeIsMissing'] = 1


# In[11]:


titanic_df.head(10)


# In[12]:


age_mean = round(titanic_df['Age'].mean())


# In[13]:


age_mean


# In[14]:


titanic_df.Age.fillna(age_mean, inplace=True)


# In[15]:


titanic_df.info()


# In[16]:


titanic_df.Embarked.fillna('S', inplace=True)


# In[17]:


titanic_df.info()


# In[18]:


cut_points = [0,18,25,40,60,100]


# In[19]:


titanic_df['Age_bin'] = pd.cut(titanic_df.Age, bins=cut_points)


# In[20]:


titanic_df.head()


# In[21]:


titanic_df['Fare_bin'] = pd.qcut(titanic_df.Fare, 5)


# In[22]:


titanic_df.Fare_bin.unique()


# In[23]:


titanic_df.head()


# In[24]:


titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1


# In[25]:


titanic_df.head()


# In[26]:


titanic_df['IsAlone'] = 0


# In[27]:


titanic_df.loc[titanic_df['FamilySize'] == 1, 'IsAlone'] = 1


# In[28]:


titanic_df.head()


# In[29]:


pd.crosstab(titanic_df.Survived,titanic_df.IsAlone).apply(lambda a:a/a.sum(),axis=0)


# In[30]:


titanic_df['IsMother'] = 0


# In[31]:


titanic_df.loc[(titanic_df['Sex']=='female') & (titanic_df['Parch']>0) & (titanic_df['Age']>20),
'IsMother'] = 1


# In[32]:


titanic_df.head()


# In[33]:


pd.crosstab(titanic_df.Survived,titanic_df.IsMother).apply(lambda a:a/a.sum(),axis=0)


# In[34]:


titanic_df['SexAge_Combo'] = titanic_df['Sex'] + "_" + titanic_df['Age_bin'].astype(str)


# In[35]:


titanic_df.head()


# In[36]:


titanic_df.info()


# In[37]:


Pclass = pd.get_dummies(titanic_df.Pclass,prefix='Pclass')


# In[38]:


Sex = pd.get_dummies(titanic_df.Sex,prefix='Sex')


# In[39]:


Embarked = pd.get_dummies(titanic_df.Embarked,prefix='Embarked')


# In[40]:


Age_bin = pd.get_dummies(titanic_df.Age_bin,prefix='Age_bin')


# In[41]:


Fare_bin = pd.get_dummies(titanic_df.Fare_bin,prefix='Fare_bin')


# In[42]:


FamilySize = pd.get_dummies(titanic_df.FamilySize,prefix='FamilySize')


# In[43]:


SexAge_Combo = pd.get_dummies(titanic_df.SexAge_Combo,prefix='SexAge_Combo')


# In[44]:


TrainData=pd.concat([titanic_df[['Survived','AgeIsMissing','IsAlone','IsMother']],Pclass,Sex,Embarked,Age_bin,Fare_bin,FamilySize,SexAge_Combo],axis=1)


# In[45]:


TrainData.head(10)


# In[46]:


TrainData.info()


# In[47]:


# as such we prepare the data for training


# In[48]:


TrainData_X = TrainData.drop(['Survived'], axis=1)


# In[49]:


TrainData_X.head()


# In[50]:


TrainData_y = TrainData.Survived


# In[51]:


TrainData_y


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(TrainData_X, TrainData_y, test_size = 0.3, random_state=123456)


# In[53]:


lr = LogisticRegression(solver='liblinear')


# In[54]:


lr.fit(X_train, y_train)


# In[55]:


y_test_pred = lr.predict(X_test)


# In[56]:


print(classification_report(y_test, y_test_pred))


# In[57]:


# Now we already have a model to predict

