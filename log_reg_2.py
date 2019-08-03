#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder , OneHotEncoder


# In[5]:


df=pd.read_csv('C:\\Users\\SAI SREEKAR KAJA\\Desktop\\python\\hackethon\\dfd.csv')
df.head()


# In[7]:


dfd=df
dfd


# In[8]:


fin_train1=dfd.dropna(subset=['is_goal'])
fin_train1.shape


# In[83]:


X=fin_train1.iloc[:,[2,5]].values
y=fin_train1.iloc[:,1].values


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)


# In[86]:


from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()


# In[88]:


log_reg.fit(X_train,y_train)
log_reg.score(X_test,y_test)
#log_reg.score(X_train,y_train)


# In[59]:


dfd2=dfd.iloc[:,1:11]


# In[ ]:





# In[60]:


dummie_list1=['power_of_shot','game_season','area_of_shot','shot_basics','range_of_shot']
def credum(df,dummie_list):
    for i in dummie_list:
        dummies=pd.get_dummies( df[i], prefix=i, dummy_na=False )
        df=df.drop(i,1)
        df=pd.concat([df,dummies],axis=1)
    return df 


# In[61]:


dfd3=credum(dfd2,dummie_list1)
dfd3.head(10)


# In[103]:


dfd3.dtypes


# In[82]:


fin_train2=dfd3.dropna(subset=['is_goal'])
fin_train2.shape


# In[104]:


X1=fin_train2.iloc[:,1:].values
y1=fin_train2.iloc[:,0].values


# In[105]:


X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=123)


# In[106]:


log_reg.fit(X1_train,y1_train)
y1_pred=log_reg.predict(X1_test)
log_reg.score(X1_test,y1_test)
#log_reg.score(X_train,y_train)


# In[107]:


log_reg.score(X1_train,y1_train)


# In[111]:


from sklearn.metrics import confusion_matrix ,classification_report


confusion = confusion_matrix(y1_test, y1_pred)
print(confusion)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# In[112]:


print(classification_report(y1_test, y1_pred))


# In[113]:


Xf=fin_train2.iloc[:,1:].values
yf=fin_train2.iloc[:,0].values


# In[114]:


log_reg.fit(Xf,yf)
#y1_pred=log_reg.predict
log_reg.score(Xf,yf)


# In[116]:


fin_test2=dfd3[dfd3['is_goal'].isna()]
fin_test2.head()


# In[119]:


fin_test2.shape


# In[122]:


Xt=fin_test2.iloc[:,1:].values
y_prbn=log_reg.predict_proba(Xt)[:,1]


# In[123]:


y_prbn


# In[135]:


sub3=pd.DataFrame()
sub3


# In[126]:





# In[132]:


sub1=pd.read_csv('C:\\Users\\SAI SREEKAR KAJA\\Desktop\\python\\hackethon\\logregsub1.csv')
sub1.head()


# In[136]:


sub3['shot_id_number']=sub1['shot_id_number']
sub3.head()


# In[137]:


sub3['is_goal']=y_prbn
sub3.head()


# In[138]:


export_csv=sub3.to_csv(r'C:\Users\SAI SREEKAR KAJA\Desktop\python\hackethon\sub3.csv', index=None,header=True)
print(sub3)


# In[ ]:




