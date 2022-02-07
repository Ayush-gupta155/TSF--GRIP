#!/usr/bin/env python
# coding: utf-8

# # Prediction using Unsupervised ML
# # Ayush Gupta 

# ### Problem statement: From the given statement ,predict the optimum number of clusters and represent it virtually 
# 
# ### Dataset: https://bit.ly/3kXTdox

# ![iris_image.png](attachment:iris_image.png)

# ### Importing the requried libraries

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='darkgrid')


# ### Reading the dataset

# In[2]:


iris=pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')


# ### Displaying the top and lower 5 values

# In[3]:


iris.head()


# In[4]:


iris.tail()


# ### Description of the dataset

# In[5]:


iris.describe()


# ### Basic info of the dataset

# In[6]:


iris.info()


# ### Checking for any null values

# In[7]:


iris.isnull().sum()


# ### Checking for any duplicate values

# In[8]:


iris.duplicated().sum()


# ### Shape of the dataset

# In[9]:


iris.shape


# ### Columns of the dataset

# In[10]:


iris.columns


# ### Correlation b/w features

# In[11]:


iris.corr()


# ### Using pandas profiling to create a report of the dataset

# In[12]:


import pandas_profiling
iris.profile_report()


# ### Visualizing the dataset to extract more info and find trends

# In[13]:


sns.scatterplot(x = 'sepal_length', y = 'petal_length', data = iris)


# In[14]:


sns.pairplot(iris)


# In[15]:


g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels = 10)


# In[16]:


ax = iris.plot(figsize=(15,8), title='Iris Dataset')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')


# In[17]:


df = iris.drop(['species'], axis = 1)


# In[18]:


df.iloc[0]


# In[19]:


df.iloc[0].plot(kind='bar')


# In[20]:


iris.plot.hist()


# In[21]:


iris.plot(kind = 'hist', stacked = False, bins = 100)


# In[22]:


iris['sepal_width'].diff()


# In[23]:


iris['sepal_width'].diff().plot(kind = 'hist', stacked = True, bins = 50)


# In[24]:


df.diff().hist(color = 'b', alpha = 0.1, figsize=(10,10))


# In[25]:


df.plot.scatter(x = 'sepal_length', y = 'petal_length', c = 'sepal_width')


# In[26]:


df.plot.hexbin(x = 'sepal_length', y = 'petal_length', gridsize = 5, C = 'sepal_width')


# In[27]:


d = df.iloc[0]
d


# In[28]:


d.plot.pie(figsize = (10,10))


# In[30]:


d = df.head(3).T
d


# In[31]:


d.plot.pie(subplots = True, figsize = (20, 20))


# In[32]:


from pandas.plotting import scatter_matrix


# In[33]:


scatter_matrix(df, figsize= (8,8), diagonal='kde', color = 'b')
plt.show()


# In[34]:


from pandas.plotting import andrews_curves


# In[35]:


andrews_curves(df, 'sepal_width')


# In[37]:


df.plot(subplots = True)


# In[38]:


#Heat Maps
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(iris.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# ### Importing the required libraries for model building

# In[39]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# ### Train test split

# In[40]:


train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[43]:


iris.head(2)


# In[45]:


train_X = train[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
train_y = train.species

test_X = test[['sepal_length', 'sepal_width','petal_length',
                 'petal_width']]
test_y = test.species


# ### Using Logistic Regression

# In[46]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[47]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))


# ### Using Support Vector

# In[48]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))


# ### Using KNN 

# In[49]:


#Using KNN 
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# ### Using Gaussian NB

# In[50]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))


# ### Using Decision Tree

# In[51]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))


# ### Results - accuracy of all the models

# In[52]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# ### Conclusion

# In this notebook, I have approached by anayzing data sets to summarize their main characteristics by using statistical graphics and other data visualization methods and built different ML models on the dataset. We can see that Logistic Regression, SVM classifier, Naive Bayes and KNN are giving an accuracy of 94.7% while Decision Tree is 92% accurate.
