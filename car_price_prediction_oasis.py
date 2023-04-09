#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("Car.csv")


# In[5]:


data.head()


# In[6]:


data.tail()


# In[8]:


data.describe()


# In[9]:


data.isnull().sum()


# In[10]:


data.info()


# In[11]:


data.CarName.unique()


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[23]:


sns.set_style("darkgrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()


# In[28]:


import plotly.express as px

px.defaults.template = 'plotly'
fig = px.histogram(data, x='price', nbins=50)
fig.update_layout(width=800, height=500)

fig.show()


# In[29]:


print(data.corr())


# In[30]:


plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# In[32]:


# Calculate the correlation matrix
corr_matrix = data.corr()

# Create a heatmap using Plotly Express
fig = px.imshow(corr_matrix, color_continuous_scale='RdBu',
                x=corr_matrix.columns, y=corr_matrix.columns)

# Customize the heatmap
fig.update_layout(width=700, height=700, title='Correlation Matrix',
                  yaxis={'autorange': 'reversed'}, margin=dict(l=50, r=50, b=100, t=100))

# Add text annotations to the heatmap
text = np.around(corr_matrix.values, decimals=2)
fig.update_traces(text=text, hoverinfo='skip')

# Show the plot
fig.show()


# In[33]:


# Calculate the correlation matrix
corr = data.corr()

# Set the figure size
fig = px.imshow(corr, color_continuous_scale='RdBu', 
                labels=dict(x="Variable", y="Variable", color="Correlation"))

# Set the ticklabels
fig.update_xaxes(tickfont=dict(size=10))
fig.update_yaxes(tickfont=dict(size=10))

# Set the colorbar range
fig.update_traces(zmin=-1, zmax=1)

# Show the plot
fig.show()


# In[34]:


predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[35]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


# In[36]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[37]:


from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# In[39]:


model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)


# In[40]:


predictions = model.predict(xtest)


# In[41]:


mae = mean_absolute_error(ytest, predictions)
print(f"Mean absolute error: {mae:.2f}")


# In[42]:


plt.figure(figsize=(10, 8))
plt.scatter(ytest, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()


# In[ ]:




