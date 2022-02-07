#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION - INTERNSHIP 
# # DATA SCIENCE AND BUSINESS ANALYTICS
# # NAME:AYUSH GUPTA

# ## TASK1 - PREDICTION USING SUPERVISED MACHINE LEARNING
# ### PROBLEM STATEMENT: The given dataset contains the score of students with respect to their study time. It is required to perform EDA and simple linear regression on the dataset to find out the score of a student who studies 9.5 hrs/day
# 
# 
# ## Dataset used:http://bit.ly/w-data

# ### Importing the required librabries

# In[49]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading the dataset

# In[50]:


df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# ### Displaying the top 5 values

# In[51]:


df.head()


# ### Displaying the last 5 values

# In[52]:


df.tail()


# ### Basic info of the dataset

# In[53]:


df.info()


# ### Dimensions of the dataset

# In[54]:


df.shape


# ### Checking for duplicated values

# In[55]:


df.duplicated()


# In[56]:


df.duplicated().sum()


# ### Checking for null values

# In[57]:


df.isnull()


# In[58]:


df.isnull().sum()


# In[59]:


df.duplicated().sum()


# ### Description of the dataset

# In[60]:


df.describe()


# ### Visualization of the dataset

# #### Distribution plot of "Hours"

# In[68]:


sns.distplot(df['Hours'], kde = True, hist = True, rug= False, bins= 30)


# #### Distribution plot of "Scores"

# In[69]:


sns.distplot(df['Scores'], kde = True, hist = True, rug= False, bins= 30)


# #### Jointplot of the dataset

# In[71]:


sns.jointplot(df['Hours'],df['Scores'])


# #### Hex-Jointplot of the dataset

# In[72]:


sns.jointplot(df['Hours'],df['Scores'],kind='hex')


# #### Kernal density estimation plot

# In[73]:


sns.jointplot(df['Hours'],df['Scores'],kind='kde')


# #### Pairplot

# In[74]:


sns.pairplot(df)


# #### Regplot

# In[78]:


f, ax = plt.subplots(figsize = (8,4))
sns.regplot(x = 'Hours', y = 'Scores', data = df, ax = ax)


# #### Linear model plot

# In[79]:


sns.lmplot(x = 'Hours', y= 'Scores', data = df)


# #### Histplot

# In[14]:


sns.histplot(df)


# ### Plotting the dataset

# In[36]:


plt.plot(df)


# #### Boxplot

# In[86]:


plt.boxplot(df, notch=None, vert=None, patch_artist=None, widths=None)


# #### Scatter plot

# In[15]:


#scatter plot of the feature and label
df.plot(kind='scatter',x='Hours',y='Scores')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### Assigning the feature and label values to variables

# In[16]:


feature_cols=['Hours']
x=df[feature_cols]
y=df.Scores


# ### Train test split

# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# ### performing the Linear Regression

# In[18]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)


# ### Intercept and coefficient of the model

# In[19]:


print(lm.intercept_)
print(lm.coef_)


# ### Plotting the least squares line

# In[20]:


#Minimum and maximum values of 'Hours'
X_new = pd.DataFrame({'Hours': [df.Hours.min(), df.Hours.max()]})
X_new.head()


# In[21]:


#predicting the values for above dataframe
preds=lm.predict(X_new)


# In[22]:


preds


# In[23]:


# first, plot the observed data
df.plot(kind='scatter', x='Hours', y='Scores')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)


# ### Predicting the values of test data

# In[24]:


predicted=lm.predict(x_test)


# In[25]:


predicted


# In[26]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})  
df 


# ### Bar chart showing the distribution of Actual and predicted values

# In[38]:


df.plot(kind = 'bar')


# ### Predicting the score for a student who studies 9.5 hrs/day

# In[27]:


new_data=pd.DataFrame({'Hours':[9.5]})


# In[28]:


new_data


# In[29]:


prediction=lm.predict(new_data)


# In[30]:


prediction


# ### MSE of the model

# In[87]:


#mean absolute error of the model
from sklearn import metrics  
print('Mean Absolute Error:', 
metrics.mean_absolute_error(y_test, predicted)) 


# ### OLS model

# In[88]:


#creating an Ordinary least squares model
import statsmodels.formula.api as smf
lm = smf.ols(formula='Hours ~ Scores', data=df).fit()
lm.conf_int()


# In[89]:


#pvalues
lm.pvalues


# In[90]:


#rsquare/ accuracy
lm.rsquared


# In[ ]:




