#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder , OneHotEncoder


# In[2]:


import keras
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD


# In[7]:


df=pd.read_csv('C:\\Users\\SAI SREEKAR KAJA\\Desktop\\python\\hackethon\\dfd.csv')
df.head()


# In[11]:


dfd=df
dfd


# In[9]:


dummie_list1=['power_of_shot','game_season','area_of_shot','shot_basics','range_of_shot']
def credum(df,dummie_list):
    for i in dummie_list:
        dummies=pd.get_dummies( df[i], prefix=i, dummy_na=False )
        df=df.drop(i,1)
        df=pd.concat([df,dummies],axis=1)
    return df 


# In[13]:


dfd=dfd.iloc[:,0:11]
dfd


# In[14]:


dfd3=credum(dfd,dummie_list1)
dfd3.head(10)


# In[15]:


fin_train2=dfd3.dropna(subset=['is_goal'])
fin_train2.shape


# In[16]:


fin_test2=dfd3[dfd3['is_goal'].isna()]
fin_test2.head()


# In[19]:


classifier=Sequential()
classifier.add(Dense(units=20, kernel_initializer='uniform',activation='relu',input_dim=49))


# In[20]:


classifier.add(Dense(units=10,
                     kernel_initializer='uniform',
                     activation='relu'))


# In[21]:


classifier.add(Dense(units=20,
                     kernel_initializer='uniform',
                     activation='relu'))


# In[22]:


#o/p layer 
classifier.add(Dense(units=1,
                     kernel_initializer='uniform',
                     activation='sigmoid'))


# In[25]:


classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['mae','acc'])


# In[26]:


print(classifier.metrics)


# In[27]:


Xf=fin_train2.iloc[:,2:].values
yf=fin_train2.iloc[:,1].values


# In[29]:


history=classifier.fit(Xf,yf,batch_size=20,epochs=200)


# In[30]:



Xt=fin_test2.iloc[:,2:].values
y_pred=classifier.predict(Xt)


# In[31]:


y_pred


# In[32]:


score=classifier.evaluate(Xf,yf)
print(score)


# In[ ]:




