#!/usr/bin/env python
# coding: utf-8

# # STOCK MARKET PREDICTION USING NUMERICAL AND TEXTUAL ANALYSIS
# ## OBJECTIVE:- Create a hybrid model for stock price/performance prediction using numerical analysis of historical stock prices, and sentimental analysis of news headlines
# ## NAME:- AYUSH GUPTA 

# ### PROJECT IDEA: Machine learning has significant applications in the stock price
# ### prediction. In this machine learning project, we will be talking about predicting the
# ### returns on stocks. We will analyse daily stocks status and predict for the next day.
# 
# 
# ### DATASETS:https://bit.ly/36fFPI6

# #### Exploratory data analysis of MSFT dataset

# In[1]:


#Importing the required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('MSFT.csv')#reading the dataset using pandas


# In[3]:


df.head()#Displaying the top 5 data values


# In[4]:


df.tail()#Displaying the lower 5 data values


# In[5]:


df.info() #Basic info of the dataset


# In[6]:


df.describe#Description of the dataset


# In[7]:


df.duplicated().sum()#Numebr of duplicated values in the dataset


# In[8]:


df.isnull() #Checking for any null values in the dataset


# In[9]:


df.isnull().sum()#Checking for the number of null values in the dataset


# In[10]:


df.columns #Columns present in the dataset


# In[11]:


df.shape #(number of rows, number of columns)


# In[12]:


df.columns=df.columns.str.upper() #Renaming the columns in uppercase


# In[13]:


df.head() #displaying top 5 values 


# #### It was observed that the Datatype of DATE column is object, so let us convert it into datetime.

# In[14]:


df['DATE']= pd.to_datetime(df['DATE'],format = "%Y-%m-%d") #converting 'DATE' from object to datetime Dtype


# In[15]:


df.info()# successfully converted the column to datetime


# In[16]:


df #8857 rows × 7 columns in our dataset


# #### The following plot shows the change in all the features values wrt time

# In[17]:


#plotting the scatterplot of all the features with respect to date
fig,axs=plt.subplots(2,3,sharey=True)
df.plot(kind='scatter', x='OPEN', y='DATE', ax=axs[0,0], figsize=(30, 15))
df.plot(kind='scatter', x='HIGH', y='DATE', ax=axs[0,1])
df.plot(kind='scatter', x='LOW', y='DATE', ax=axs[0,2])
df.plot(kind='scatter', x='CLOSE', y='DATE', ax=axs[1,0])
df.plot(kind='scatter', x='ADJ CLOSE', y='DATE', ax=axs[1,1])
df.plot(kind='scatter', x='VOLUME', y='DATE', ax=axs[1,2])


# ## Insights from the above plots:-
# - OPEN, HIGH, LOW, CLOSE and ADJ CLOSE features are showing similar relationship/trend with respect to time.
# - The data of the above features is constant till the year 1996, after that, a sudden growth is observed in features with    respect to time.
# - The 'VOLUME' feature is mostly distributed between the values 0.0 to 0.4, few outliers are observed.

# In[18]:


#plotting the feature values 
plt.figure(figsize=(20,25))
plotnumber=1
for column in df:
    if plotnumber<=7:
        ax=plt.subplot(3,3,plotnumber)
        plt.plot(df[column])
        plt.xlabel(column, fontsize=20)
    plotnumber+=1
plt.show()


# ### Insights from the above plots:-
# - OPEN, HIGH, LOW, CLOSE and ADJ CLOSE features are showing similar relationship/trend with respect to time.
# - Elbow is observed in the year 2000 after which the values suddenly decrease and then increase wrt time.
# - The maximum volume was achieved before the year 1988, the volumne distribution decreases with time.

# In[19]:


df=df.set_index('DATE') #Setting the Date feature as an index


# In[20]:


df


# In[21]:


#Plotting the distribution plots for all the features
plt.figure(figsize=(20,25))
plotnumber=1
for column in df:
    if plotnumber<=7:
        ax=plt.subplot(3,3,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column, fontsize=20)
    plotnumber+=1
plt.show()


# ### Insights from the above plots:-
# - The distribution of a statistical data set (or a population) is a listing or function showing all the possible values (or intervals) of the data and how often they occur. When a distribution of categorical data is organized, you see the number or percentage of individuals in each group.
# - From the above plots, we can observe that the density shows a decrease after the point 100 on the X- axis for the first 5 features.
# - Max density in volume feature is about 1.7. The density is almost constant at 0.0 and is 0 after 0.2. The data is mostly distributed between 0.0 and 0.2

# In[22]:


sns.pairplot(data=df) # relation between each and every variable present in Pandas DataFrame.


# ### Insights from the plot above:-
# - Above plot shows pairwise relationships in a dataset.
# - Most of the plots are showing linear relationship while others are not showing any specific relationship

# In[23]:


sns.heatmap(df.corr(),annot=True,cmap = "YlGnBu") #Graphical representation of data using colors to visualize the value of the matrix.


# ### Insights from the plot above:-
# - A heatmap contains values representing various shades of the same colour for each value to be plotted. Usually the darker shades of the chart represent higher values than the lighter shade. For a very different value a completely different colour can also be used.
# - It is observed that all the features are highly correlated with each other except volume.

# #### Let us check for any outliers

# In[24]:


#plotting boxplots for all the features
plt.figure(figsize=(20,25))
plotnumber=1
for column in df:
    if plotnumber<=7:
        ax=plt.subplot(3,3,plotnumber)
        sns.boxplot(df[column])
        plt.xlabel(column, fontsize=20)
    plotnumber+=1
plt.show()


# In[25]:


#Vertical visualization of the boxplot
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=df, width= 0.8,ax=ax,  fliersize=7)


# ### Insights from the plot above:-
# - Boxplots are a measure of how well distributed the data in a data set is. It divides the data set into three quartiles. This graph represents the minimum, maximum, median, first quartile and third quartile in the data set.
# - Outliers are observed in the feature 'VOLUME'.

# In[26]:


#Displaying data using a number line
plt.figure(figsize=(30,15))
sns.lineplot(data = df)
plt.title('LINE PLOT')
plt.xlabel("DATE")


# ### Insights from the plot above:-
# - The above plot shows a line plot with the possibility of several semantic groupings. The relationship is shown for different subsets of the data using different parameters.

# In[27]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in df:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(df[column])
    plotnumber+=1
plt.tight_layout()


# ### Insights from the plots above:-
# - The plot above is a graphical data anlysis technique for summarizing a univariate data set.
# - This plot shows us how data is distributed for every column.
# - The distribution is similar for the first 5 plots.
# - Dense distribution is observed from 0 to 50 for the first five plots.
# - The data is dispersed from 50 to 200.

# In[28]:


#EDA for the time series 


# In[29]:


df


# In[30]:


df=df.reset_index()


# In[31]:


df


# In[32]:


df['DATE'].plot()


# - Linear relationship is observed in the plot above

# In[33]:


df['DATE'].describe()


# In[34]:


df['DATE'].mean()


# #### Checking for randomness in the dataset

# In[35]:


import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['DATE'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['DATE'], lags=40, ax=ax2)


# - A positive correlation indicates that large current values correspond with large values at the specified lag
# - a negative correlation indicates that large current values correspond with small values at the specified lag.
# - The absolute value of a correlation is a measure of the strength of the association, with larger absolute values indicating stronger relationships.

# ## EDA of the combined dataset

# In[36]:


data=pd.read_csv('stock_data.csv') #reading the csv file


# In[37]:


data.head()#displaying the top 5 values


# In[38]:


data.tail()#displaying the lower 5 values


# In[39]:


data.describe() #description of the dataset


# In[40]:


data.shape#dimentions of the dataset


# In[41]:


data.info()#basic info of the dataset


# In[42]:


data.duplicated()#checking for any duplicated value


# In[43]:


data.duplicated().sum()# checking for the number of duplicated values


# In[44]:


data.isnull()#checking for any null value


# In[45]:


data.isnull().sum()#checking for number of null values


# In[46]:


data.columns=data.columns.str.upper() #Renaming the columns in uppercase


# In[47]:


data.head(2)#displaying top 2 values


# In[48]:


data['DATE']= pd.to_datetime(data['DATE'],format = "%Y-%m-%d") #converting 'DATE' from object to datetime Dtype


# In[49]:


data.info()#successfully converted 'DATE' dtype to datetime


# In[50]:


data.columns#columns in the dataset


# In[51]:


data['STOCK'].unique()#unique values in 'STOCK' feature


# In[52]:


data.groupby(['STOCK']).mean()#grouping by the mean of 'STOCK' feature


# In[53]:


sns.pairplot(data.groupby(['STOCK']).mean())#pairplot


# We have grouped the whole dataset with respect to companies by which we can seprate the dataset accordingly

# In[54]:


data.groupby(['STOCK']).mean().plot( kind='bar',alpha=0.3)


# In[55]:


#stripplot
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(data[column])
    plotnumber+=1
plt.tight_layout()


# Above plot is a strip plot. A strip plot is a graphical data anlysis technique for summarizing a univariate data set which shows us how the values are distributed. Above dataset shows how the features are distributed with respect to time.

# In[56]:


#Histogram of the dataset
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(data[column])
    plotnumber+=1
plt.tight_layout()


# In[57]:


plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        plt.plot(data[column])
    plotnumber+=1
plt.tight_layout()


# In[58]:


sns.pairplot(data=data)


# In[59]:


sns.heatmap(data.corr(),annot=True,cmap = "YlGnBu") #Graphical representation of data using colors to visualize the value of the matrix.


# In[60]:


#Vertical visualization of the boxplot
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=data, width= 0.8,ax=ax, fliersize=7)


# In[61]:


#Displaying data using a number line
plt.figure(figsize=(30,15))
sns.lineplot(data = data)
plt.title('LINE PLOT')
plt.xlabel("DATE")


# In[62]:


data=data.reset_index()


# In[63]:


data.head()


# In[64]:


data['DATE'].plot()
plt.title("PLOT SHOWING DISTRIBUTION OF VALUES OF 'DATE'")


# In[65]:


data['DATE'].describe()#description of the dataset


# In[66]:


data["DATE"].mean()#mean of 'DATE'


# ### Checking for randomness in the data

# In[67]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data['DATE'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data['DATE'], lags=40, ax=ax2)


# ### EDA on the basis of individual stock

# In[68]:


data.head()#displaying the first 5 values


# In[69]:


data.drop(columns=['OPENINT'], inplace=True)#Dropping the OPENINT column


# In[70]:


data.head()#column dropped


# In[71]:


stock_data=data.groupby('STOCK')#grouping the values by STOCK


# In[72]:


stock_data.head()#values grouped


# In[73]:


data['STOCK'].unique()#unique values in stock


# In[74]:


#Diving the data on the basis of company
df_AAPL = data[data.STOCK=='AAPL']
df_TSLA = data[data.STOCK=='TSLA']
df_MSFT = data[data.STOCK=='MSFT']
df_FB   =data[data.STOCK=='FB']


# In[75]:


df_AAPL.head() #displaying the dataframes created


# In[76]:


df_TSLA.head()


# In[77]:


df_MSFT.head()


# In[78]:


df_FB.head()


# In[79]:


companies=[df_AAPL, df_MSFT,df_TSLA,df_FB]
company_name=['df_AAPL', 'df_MSFT','df_TSLA','df_FB']


# In[80]:


plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companies, 1):
    plt.subplot(2, 2, i)
    company['OPEN'].plot()
    plt.ylabel('OPEN')
    plt.xlabel(None)
    plt.title(f"Opening Price of {company_name[i - 1]}")
    
plt.tight_layout()


# In[81]:


plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companies, 1):
    plt.subplot(2, 2, i)
    company['CLOSE'].plot()
    plt.ylabel('CLOSE')
    plt.xlabel(None)
    plt.title(f"Closing Price of {company_name[i - 1]}")
    
plt.tight_layout()


# In[82]:


nrow=2
ncol=2
fig, axes = plt.subplots(nrow, ncol,figsize=(15,15))

# plot counter
count=0
for r in range(nrow):
    for c in range(ncol):
        companies[count]['CLOSE'].plot(ax=axes[r,c])
        count+=1


# In[83]:


plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companies, 1):
    plt.subplot(2, 2, i)
    company['HIGH'].plot()
    plt.ylabel('HIGH')
    plt.xlabel(None)
    plt.title(f"Hghest Price of {company_name[i - 1]}")
    
plt.tight_layout()


# In[84]:


plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companies, 1):
    plt.subplot(2, 2, i)
    company['LOW'].plot()
    plt.ylabel('LOW')
    plt.xlabel(None)
    plt.title(f"Lowest Price of {company_name[i - 1]}")
    
plt.tight_layout()


# In[85]:


nrow=2
ncol=2
fig, axes = plt.subplots(nrow, ncol,figsize=(15,7))

# plot counter
count=0
for r in range(nrow):
    for c in range(ncol):
        companies[count]['VOLUME'].plot(ax=axes[r,c])
        count+=1


# In[86]:


plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companies, 1):
    plt.subplot(2, 2, i)
    company['VOLUME'].plot()
    plt.ylabel('VOLUME')
    plt.xlabel(None)
    plt.title(f"VOLUME of {company_name[i - 1]}")
    
plt.tight_layout()


# In[87]:


plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companies, 1):
    plt.subplot(2, 2, i)
    company['DATE'].plot()
    plt.ylabel('DATE')
    plt.xlabel(None)
    plt.title(f"DATE distribution of {company_name[i - 1]}")
    
plt.tight_layout()


# In[88]:


sns.set(style="ticks", rc={"lines.linewidth": 7})
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(20)
fig.set_figwidth(30)


df_AAPL[['OPEN','CLOSE','HIGH','LOW']].plot(ax=axes[0,0],alpha=1)
axes[0,0].set_title('APPLE', fontsize=50)


df_MSFT[['OPEN','CLOSE','HIGH','LOW']].plot(ax=axes[0,1],alpha=0.8)
axes[0,1].set_title('MICROSOFT', fontsize=50)


df_TSLA[['OPEN','CLOSE','HIGH','LOW']].plot(ax=axes[1,0],alpha=0.8)
axes[1,0].set_title('TESLA', fontsize=50)


df_FB[['OPEN','CLOSE','HIGH','LOW']].plot(ax=axes[1,1],alpha=0.8)
axes[1,1].set_title('FACEBOOK', fontsize=50)



fig.tight_layout()


# In[89]:


data.groupby("STOCK").hist(figsize=(12, 12));


# In[90]:


cp_AAPL=pd.DataFrame(df_AAPL['CLOSE']) 
cp_AAPL.rename(columns={'CLOSE':'CP_AAPL'},inplace=True)
cp_MSFT=pd.DataFrame(df_MSFT['CLOSE']) 
cp_MSFT.rename(columns={'CLOSE':'CP_MSFT'},inplace=True)
cp_TSLA=pd.DataFrame(df_TSLA['CLOSE']) 
cp_TSLA.rename(columns={'CLOSE':'CP_TSLA'},inplace=True)
cp_FB=pd.DataFrame(df_FB['CLOSE']) 
cp_FB.rename(columns={'CLOSE':'CP_FB'},inplace=True)


# In[91]:


closing_df= pd.concat([cp_AAPL,cp_MSFT,cp_TSLA,cp_FB], axis=1)


# In[92]:


closing_df.head()


# In[93]:


sns.heatmap(closing_df.corr(), annot=True)


# In[94]:


df_AAPL=df_AAPL.set_index(df_AAPL['DATE'])
df_AAPL.head()


# In[95]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df_AAPL['CLOSE'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()


# In[96]:


# Create a new dataframe with only the 'Close column 
data = df_AAPL.filter(['CLOSE'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len


# In[97]:


# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[98]:


# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


# In[128]:


df_TSLA


# In[ ]:





# In[147]:


df_AAPL=df_AAPL.set_index(df_AAPL['DATE'])
df_MSFT=df_MSFT.set_index(df_MSFT['DATE'])
df_TSLA=df_TSLA.set_index(df_TSLA['DATE'])
df_FB=df_FB.set_index(df_FB['DATE'])


# In[149]:


closed_AAPL=df_AAPL.filter(['CLOSE'])
closed_AAPL.columns=['AAPL_CLOSE']
closed_MSFT=df_MSFT.filter(['CLOSE'])
closed_MSFT.columns=['MSFT_CLOSE']
closed_TSLA=df_TSLA.filter(['CLOSE'])
closed_TSLA.columns=['TSLA_CLOSE']
closed_FB=df_FB.filter(['CLOSE'])
closed_FB.columns=['FB_CLOSE']


# In[156]:


closed_AAPL


# In[157]:


closed_TSLA


# In[158]:


closed_MSFT


# In[159]:


closed_FB


# In[180]:


new1=closed_AAPL.merge(closed_MSFT, left_on='DATE', right_on='DATE', how='inner')


# In[181]:


new2=closed_TSLA.merge(closed_FB, left_on='DATE', right_on='DATE', how='inner')


# In[182]:


new_df=new1.merge(new2,left_on='DATE', right_on='DATE', how='inner')


# In[184]:


new_df


# In[185]:


dataset=new_df.values


# In[186]:


dataset


# In[193]:


# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# #  MODEL BUILDING

# Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can process not only single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition,[2] speech recognition[3][4] and anomaly detection in network traffic or IDSs (intrusion detection systems).
# 
# A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.
# 
# LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications.
# SOURCE:Wikipedia

# ![LSTM3-var-GRU.png](attachment:LSTM3-var-GRU.png)

# ![1_0f8r3Vd-i4ueYND1CUrhMA.png](attachment:1_0f8r3Vd-i4ueYND1CUrhMA.png)

# # PRICE PREDICTION FOR APPLE

# In[99]:


from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[100]:


# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[101]:


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['CLOSE'])
plt.plot(valid[['CLOSE', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[102]:


valid


# # PRICE PREDICTION FOR MICROSOFT

# In[103]:


df_MSFT=df_MSFT.set_index(df_MSFT['DATE'])
df_MSFT.head()


# In[104]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df_MSFT['CLOSE'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()


# In[105]:


# Create a new dataframe with only the 'Close column 
data = df_MSFT.filter(['CLOSE'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len


# In[106]:


# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[107]:


# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


# In[108]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[109]:


# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[110]:


valid


# # PRICE PREDICTION FOR TESLA

# In[111]:


df_TSLA=df_TSLA.set_index(df_TSLA['DATE'])
df_TSLA.head()


# In[112]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df_TSLA['CLOSE'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()


# In[113]:


# Create a new dataframe with only the 'Close column 
data = df_TSLA.filter(['CLOSE'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len


# In[114]:


# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[115]:


# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


# In[116]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[117]:


# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[118]:


valid


# # PRICE PREDICTION FOR FACEBOOK

# In[119]:


df_FB=df_FB.set_index(df_FB['DATE'])
df_FB.head()


# In[120]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df_FB['CLOSE'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()


# In[121]:


# Create a new dataframe with only the 'Close column 
data = df_FB.filter(['CLOSE'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len


# In[122]:


# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[123]:


# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


# In[124]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[125]:


# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[128]:


valid


# # CONCLUSION

# In this notebook we have looked at data from the stock market with respect to different companies, particularly some technology stocks. We have used pandas to get stock information,we have used libraries like matplotlib and seaborn to visualize different aspects of it, and finally we have performed feature scaling and based on its previous performance history we have built a model to predict future stock prices through a Long Short Term Memory (LSTM) method.
