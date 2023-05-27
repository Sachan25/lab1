#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset

# The famous and historic titanic dataset is used to learn statistics in data analysis/science/ML/AI.

# In[72]:


#install if you do not have it already
#pip install matplotlib         


# In[1]:


#libraries
import pandas as pd #pandas for dataframes
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #nicer plots, read https://seaborn.pydata.org


# In[74]:


#reads and creates dataframe(df) from titanic dataset
#Read below for info on variables on the data
#https://notebook.community/vikramsjn/Investigate-Titanic-Dataset/Analysis%20of%20Titanic%20Dataset#
#https://trenton3983.github.io/files/titanic.html

df = pd.read_csv('TitanicData.csv')
df


# In[75]:


#gives the first 5 rows, similarily, tail() gives the last 5 rows
df.head()


# In[76]:


#describe gives statistics, such as quartiles, mean, and std
df.describe()


# In[77]:


#check the info and datatype
df.info()


# In[78]:


np.mean(df) #mean


# In[79]:


df.median() #median


# In[80]:


np.std(df) #standard deviation


# In[81]:


# groupby() aggregates 
df.groupby('Sex')[['Survived']].aggregate(['sum'])


# In[82]:


df.groupby('Embarked')['Name'].count()


# In[83]:


df.groupby('Pclass')['Name'].count()


# In[84]:


# can seperate data by adding a list
df.groupby(['Sex','Survived'])['Age'].count()


# In[85]:


df.groupby('Pclass')[['Fare','Age']].aggregate(['min','mean','max'])


# In[97]:


#seris of ages, not very meaningful, just an example
plt.figure(figsize=(8,4))
plt.plot(df['PassengerId'],df['Age'])
plt.title('Age distribution with respect to PassengerID')


# In[87]:


# histogram
df.Age.hist()
plt.xlabel('Age')
plt.ylabel('Passengers')
plt.title('Passengers vs Age')


# In[88]:


plt.boxplot(df['Age'])
plt.title('boxplot')


# In[101]:


#plot box plot
plt.boxplot(df.Age)
plt.title('boxplot')


# In[102]:


sns.boxplot(x='Pclass',y='Age',data=df)


# In[91]:


#distributions layered
sns.distplot(df['Age'])
sns.distplot(df['Fare'])


# In[92]:


#male and female ages
male_ages = (df[df.Sex == 'male'])['Age']
female_ages = (df[df.Sex == 'female'])['Age']


# In[93]:


male_ages.hist(label='Male')
female_ages.hist(label='Female')

plt.xlabel('Age')
plt.ylabel('Passengers')
plt.title('Male & Female passenger ages')
plt.legend(loc='best')


# In[94]:


#plots data and adds colour coded label
sns.lmplot('Age','Fare', data=df, hue='Sex')


# In[95]:


#read https://seaborn.pydata.org/examples/wide_form_violinplot.html for information regarding what this means
sns.violinplot(df['Age'])


# In[96]:


sns.barplot(x='Pclass', y='Survived', data=df, hue='Sex').set_title('Gender Survival by Class')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




