#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis on dataset - Terrorism

# # GRIP(THE SPARKS FOUNDATION)Datascience-Internship
# # NAME :- AYUSH GUPTA 

# ### Problem statement: As a security/ defence analyst, try to find out the hot zones of terrorism
# ### Dataset-https://bit.ly/2TK5Xn5

# ![shutterstock_306542147.jpg](attachment:shutterstock_306542147.jpg)

# # Importing the required libraries

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'darkgrid')


# # Reading the dataset

# In[2]:


df=pd.read_csv('global_terrorism.csv',encoding='latin1')


# # Displaying the top and lower 5 values of the dataset

# In[3]:


df.head()


# In[4]:


df.tail()


# # Number of rows and columns in the dataset

# In[5]:


df.shape


# # Basic info of the dataset

# In[6]:


df.info()


# # Checking for any nill values

# In[7]:


df.isnull().sum()


# # Checking for any duplicated values

# In[8]:


df.duplicated().sum()


# # Description of the dataset

# In[9]:


df.describe()


# # Checking the correlation between features

# In[10]:


df.corr()


# # Columns in the dataset

# In[11]:


df.columns


# # Renaming the columns for better observation

# In[12]:


df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                       'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                       'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# # Extracting the important columns

# In[13]:


df=df[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed',
               'Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]


# In[14]:


df.head()


# # Value count of the 'year' column

# In[15]:


df['Year'].value_counts()


# # In which year did  max and min attacks took place?

# In[16]:


df['Year'].value_counts().idxmax()


# In[17]:


df['Year'].value_counts().idxmin()


# # Distplot of 'Year' column

# In[54]:


sns.distplot(df['Year'], kde = True, hist = True, rug= False, bins= 30,color='red')


# # Descending order of the type of attacks

# In[19]:


df['AttackType'].value_counts()


# # Attack types which have occured maximum/minimum time

# In[20]:


df['AttackType'].value_counts().idxmax()


# In[21]:


df['AttackType'].value_counts().idxmin()


# In[22]:


(df['AttackType'].value_counts()/df.shape[0])*100


# In[23]:


plt.figure(figsize=(15,12))
sns.histplot(df['AttackType'],color='red')
plt.title('Attack Types',fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[24]:


df.head(2)


# # Weapons used maximum/ minimum time

# In[25]:


plt.figure(figsize=(15,12))
sns.histplot(df['Weapon_type'],color='red')
plt.title('Weapon Types',fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[26]:


df['Weapon_type'].value_counts()


# In[27]:


df['Weapon_type'].value_counts().idxmax()


# In[28]:


df['Weapon_type'].value_counts().idxmin()


# In[29]:


(df['Weapon_type'].value_counts()/df.shape[0])*100


# In[30]:


df.head(2)


#   # Target type analysis

# In[31]:


df['Target_type'].value_counts()


# In[32]:


df['Target_type'].value_counts().idxmax()


# In[33]:


df['Target_type'].value_counts().idxmin()


# In[34]:


plt.figure(figsize=(15,12))
sns.histplot(df['Target_type'],color='red')
plt.title('Target Type',fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[35]:


(df['Target_type'].value_counts()/df.shape[0])*100


# In[36]:


df.head(2)


# # Motive analysis

# In[37]:


df['Motive'].value_counts()


# In[38]:


(df['Motive'].value_counts()/df.shape[0])*100


# In[39]:


df.head(2)


# # Countries with max/min terrorist activities

# In[40]:


df['Country'].value_counts()


# In[41]:


df['Country'].value_counts().idxmax()


# In[42]:


df['Country'].value_counts().idxmin()


# In[43]:


plt.figure(figsize=(40,12))
sns.histplot(df['Country'],color='red')
plt.title('Country',fontsize=30)
plt.xticks(rotation=90)
plt.show()


# In[44]:


(df['Country'].value_counts()/df.shape[0])*100


# In[45]:


df.head(2)


# # Most active terrorist groups

# In[46]:


df['Group'].value_counts()


# In[47]:


df['Group'].value_counts().idxmax()


# In[48]:


df['Group'].value_counts().idxmin()


# In[49]:


plt.figure(figsize=(10,10))
sns.barplot(df['Group'].value_counts()[1:11].values, df['Group'].value_counts()[1:11].index)
plt.title('Top 10 Terrorist Organization with Highest Terror Attacks',fontsize=15)
plt.xlabel('Number of Attacks',fontsize=15)
plt.ylabel('Terrorist Groups',fontsize=15)
plt.show()


# In[50]:


df.head(2)


# In[51]:


plt.subplot(1,2,1)
# regions with most attacks 
sns.barplot(df['Region'].value_counts().index, df['Region'].value_counts().values)
plt.title('Most Attacked Regions',fontsize=25)
plt.xlabel('Regions',fontsize=25)
plt.ylabel('Number of Attacks',fontsize=25)
plt.xticks(rotation=90)

plt.subplot(1,2,2)
sns.barplot(df['state'].value_counts().head(15).index, df['state'].value_counts().head(15).values)
plt.title('Top 10 Most Attacked States',fontsize=25)
plt.xlabel('States',fontsize=25)
plt.ylabel('Number of Attacks',fontsize=25)
plt.xticks(rotation=90)
plt.gcf().set_size_inches(15, 5)


# In[52]:


plt.figure(figsize = (12,7))
df.groupby(['Year'])['Killed'].sum().plot(kind='bar')
plt.title('Number of Deaths in different years',fontsize=25)
plt.xlabel('Years',fontsize=25)
plt.ylabel('Number of Deaths',fontsize=25)
plt.xticks(rotation=90)
plt.show()


# In[53]:


plt.figure(figsize = (12,7))
sns.kdeplot(df['Year'], hue = df['Region'])
plt.title('Terrorist Activities by Region in each Year',fontsize=25)
plt.xlabel('Years',fontsize=25)
plt.ylabel('Frequency of Attacks',fontsize=25)
plt.xticks(rotation=90)
plt.show()


# # Conclusion

# In this notebook, I have approached by analyzing data sets to summarize their main characteristics using statistical graphics and other data visualization methods to extract the useful info and find trends.
