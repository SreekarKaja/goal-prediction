#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder , OneHotEncoder


# In[2]:


df=pd.read_csv('C:\\Users\\SAI SREEKAR KAJA\\Desktop\\python\\hackethon\\fin_train1.csv')
df.head()


# In[6]:


fin_train1=df


# In[4]:


from sklearn.model_selection import train_test_split


# In[7]:


X=fin_train1.iloc[:,2:].values
y=fin_train1.iloc[:,1].values


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[11]:


from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[17]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}


# In[18]:


from sklearn.model_selection import GridSearchCV


# In[19]:


grid = GridSearchCV(SVC(probability=True),param_grid,refit=True,verbose=3)


# In[20]:


grid.fit(X_train,y_train)


# In[21]:


grid.best_params_


# In[22]:


grid.best_estimator_


# In[44]:


grid_predictions = grid.predict(X_test)


# In[24]:


print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


# In[43]:





# In[41]:


grid_predictions=grid_predictions.reshape(1,-1 )
y_test=y_test.reshape(1,-1)


# In[47]:


grid_predictions.shape


# In[42]:


y_test


# In[46]:


y_test.shape


# In[48]:


yt=y_test[0]


# In[51]:


gp=grid_predictions[np.newaxis,:]
gp.shape


# In[52]:





# In[53]:


fin_test1=pd.read_csv('C:\\Users\\SAI SREEKAR KAJA\\Desktop\\python\\hackethon\\fin_test1.csv')
fin_test1.head()


# In[54]:


Xk=fin_test1.iloc[:,2:].values


# In[ ]:





# In[56]:


grid.predict_proba(Xk,probability=True)


# In[63]:


from sklearn import svm
clf=svm.SVC(C=10, gamma=0.01, kernel='rbf', probability=True)


# In[65]:


clf.fit(X_train,y_train)
clf.predict_proba(Xk)


# In[68]:


y_probt=clf.predict_proba(Xk)[:,1]
y_probt


# In[69]:


sub2=pd.DataFrame()
sub2


# In[71]:


sub2['shot_id_number']=fin_test1['shot_id_number']
sub2.head()


# In[72]:


sub2['is_goal']=y_probt
sub2.head()


# In[73]:


export_csv=sub2.to_csv(r'C:\Users\SAI SREEKAR KAJA\Desktop\python\hackethon\sub2.csv', index=None,header=True)
print(sub2)


# In[ ]:




