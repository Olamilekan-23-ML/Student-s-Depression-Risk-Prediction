#!/usr/bin/env python
# coding: utf-8

# In[1]:


#____IMPORTING DEPENDENCIES____#
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#___LOADING DATASET___#
depression = pd.read_csv('student_depression_dataset.csv')


# In[3]:


#___CHECKING THE FIRST 5 ROW OF THE DATA___#
depression.head()


# In[4]:


#___CHECKING THE DATA INFO___#
depression.info()


# In[5]:


#___DESCRIPTIVE STATISTICS___#
depression.describe()


# In[6]:


#___SHAPE OF THE DATA___#
depression.shape


# In[7]:


#___CHECKING THE LABEL COUNT___#
#_Not Depressed--> 0
#_Depressed--> 1
depression['Depression'].value_counts()


# In[8]:


#___CHECKING FOR MISSING VALUE___#
depression.isnull().sum()


# In[9]:


#___DROPPING FEATURES THAT ARE NOT NEEDED___#
column_to_drop = ['City', 'Job Satisfaction', 'id','Profession','Degree','Work Pressure']
depression = depression.drop(columns=column_to_drop, axis=1)


# In[10]:


depression.head()


# In[11]:


#___ENCODING CATEGORICAL FEATURES___#
depression['Sleep Duration'] = depression['Sleep Duration'].map({
    "'Less than 5 hours'": 0, "'5-6 hours'": 1, 
    "'7-8 hours'": 2, "'More than 8 hours'": 3, 
    'Others': 4
})
depression['Dietary Habits'] = depression['Dietary Habits'].map({
    'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2, 'Others': 3
})
depression['Have you ever had suicidal thoughts ?'] = (depression['Have you ever had suicidal thoughts ?'] == 'Yes').astype(int)
depression['Family History of Mental Illness'] = (depression['Family History of Mental Illness'] == 'Yes').astype(int)
depression['Gender'] = (depression['Gender'] == 'Male').astype(int)


# In[12]:


depression.head()


# In[13]:


depression.info()


# In[14]:


#___CORRELATION OF THE DATA___#
depression.corr()


# In[15]:


#___PLOTTING THE CORRELATION___#
plt.figure(figsize=(10,8))
sns.heatmap(depression.corr(), annot = True, cmap='Blues', fmt = '.2f')


# In[16]:


#___SEPERATING THE TARGET AND THE FEATURES___#
X = depression.drop(columns='Depression', axis=1)
Y = depression['Depression']


# In[17]:


print(X.shape)


# In[18]:


print(Y.shape)


# In[19]:


#___SPLITTING THE DATA___#
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, stratify= Y, random_state=3)


# In[20]:


print(X.shape,X_train.shape, X_test.shape)


# In[21]:


model = LogisticRegression(max_iter=1000)


# In[23]:


#___CROSS VAL ON TRAINING DATA___#
cv_score_lr = cross_val_score(model, X_train, y_train, cv=5)
print(cv_score_lr)
mean = sum(cv_score_lr)/len(cv_score_lr)
mean = mean*100
print(mean)


# In[24]:


#___TRAINING MODEL___#
model.fit(X_train, y_train)


# In[25]:


#___EVALUATION METRICS OF THE TEST DATA___#
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, y_test)
print(test_accuracy)


# In[26]:


#___TESTING MODEL WITH DATA___#
input_data = (1,33,5,8.97,2,1,2,1,3,1,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print("The Student is Depressed")
else:
    print('The Student is Not-Depressed')


# ## SAVING MODEL 

# In[27]:


import pickle 


# In[28]:


filename = 'train_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[29]:


load_model = pickle.load(open('train_model.sav', 'rb'))


# In[30]:


input_data = (1,33,5,8.97,2,1,2,1,3,1,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = load_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print("The Student is Depressed")
else:
    print('The Student is Not-Depressed')


# In[ ]:




