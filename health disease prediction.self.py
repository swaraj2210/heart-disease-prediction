#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import pandas_profiling as pp
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("heart.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe().transpose()


# In[7]:


for f in ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']:
    plt.figure(figsize=(5, 5))
    sns.boxplot(df[f])
    plt.title(f)


# In[8]:


df=df.loc[(df['chol']<350)&(df['chol']>120)]
df=df.loc[(df['trestbps']<170)]
df=df.loc[(df['thalach']>90)]
df=df.loc[(df['oldpeak']<4.2)]
df=df.loc[(df['ca']<2.2)]
df=df.loc[(df['thal']>0.5)]


# In[9]:


for f in ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']:
    plt.figure(figsize=(5, 5))
    sns.boxplot(df[f])
    plt.title(f)


# In[10]:


pp.ProfileReport(df)


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[ ]:





# In[16]:


## dependent and independent features

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

## converting X and y into numpy array's

X = X.values
y = y.values


# In[17]:


## scaling the data 

sc = StandardScaler()
X = sc.fit_transform(X)


# In[18]:


## splitting the dataset into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[19]:


## logistic Regression

lg = LogisticRegression()


# In[20]:


lg.fit(X_train, y_train)


# In[22]:


preds = lg.predict(X_test)
pd.DataFrame({'Acutual': y_test, 'Predicted': preds})


# In[23]:


print("------------------------------------------------Accuracy Score-------------------------------------------------------")
print(accuracy_score(y_test, preds))

print("------------------------------------------Classfication Report--------------------------------------------------------")
print(classification_report(y_test, preds))

print("----------------------------------------------------------Confusion Matrix---------------------------------------------")
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, preds), annot=True);

