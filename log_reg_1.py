#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
#** second part of this code can be found in project files log_reg_2 
# which is continued using dfd dataframe for further improvisation


# In[2]:


df=pd.read_csv('C:\\Users\\SAI SREEKAR KAJA\\Desktop\\python\\hackethon\\data.csv')
df.head()


# In[3]:


df.shape


# In[4]:


list=df.columns
list


# In[5]:


data=df.ix[:,(4,5,6,7,8,9,10,11,12,13,16,17,19,20)]
data.head()


# In[6]:


data.shape


# In[7]:


pd.crosstab(data.knockout_match,data.is_goal)


# In[8]:


data.knockout_match.isnull().sum()


# In[9]:


data.knockout_match.describe()


# In[10]:


data['knockout_match'].fillna(data['knockout_match'].mode()[0],inplace=True)


# In[11]:


data.knockout_match.isnull().sum()


# In[12]:


pd.crosstab(data.knockout_match,data.is_goal)


# In[13]:


data.dtypes


# In[14]:


data['knockout_match'].values.astype(int)
data.knockout_match.describe()


# In[15]:


data.knockout_match.value_counts()


# In[16]:


labelencode=LabelEncoder()
data.iloc[:,2]=labelencode.fit_transform(data.iloc[:,2])
data.head()


# In[18]:


data['distance_of_shot'].describe()


# In[19]:


data['distance_of_shot'].isnull().sum()


# In[20]:


data['distance_of_shot'].median()


# In[21]:


data['distance_of_shot'].fillna(data['distance_of_shot'].median(),inplace=True)
data.head(20)


# In[22]:


data['distance_of_shot'].isnull().sum()


# In[23]:


pd.crosstab(data.area_of_shot,data.is_goal)


# In[24]:


data.area_of_shot.mode()


# In[25]:


data['area_of_shot'].isnull().sum()


# In[26]:


data['area_of_shot'].describe()


# In[30]:


data['area_of_shot'].fillna(data['area_of_shot'].mode()[0],inplace=True)
data.head(5)


# In[31]:



data.range_of_shot.describe()


# In[32]:


pd.crosstab(data.range_of_shot,data.is_goal)


# In[33]:


data['range_of_shot'].isnull().sum()


# In[34]:


data['range_of_shot'].fillna(data['range_of_shot'].mode()[0],inplace=True)
data.head(20)


# In[35]:


data.shot_basics.describe()


# In[36]:


pd.crosstab(data.shot_basics,data.is_goal)#.plot(kind='bar')


# In[37]:


data['shot_basics'].isnull().sum()


# In[38]:


data['shot_basics'].fillna(data['shot_basics'].mode()[0],inplace=True)
data.head(20)


# In[39]:


bkpdata0=data
bkpdata0.head()


# In[40]:


for i in range(1, len(data.index)) :
     if pd.isna(data.loc[i,'game_season']):
        data.loc[i,'game_season']=data.loc[i-1,'game_season']
     
data.head(10)


# In[41]:


pd.crosstab(data.game_season,data.is_goal)#.plot(kind='bar')


# In[42]:


bkpdata=data.iloc[:,:]
bkpdata.shape


# In[43]:


data["type_of_combined_shot"]=data["type_of_combined_shot"].str.replace("shot","Shot",case=False)
for i in data.type_of_combined_shot :
    print(i)


# In[44]:


data['type_of_shot'].fillna(0,inplace=True)
data.head(10)


# In[45]:


for i in range(0,len(data.index)) :
    if data.loc[i,'type_of_shot']==0 :
        data.loc[i,'type_of_shot']=data.loc[i,'type_of_combined_shot']
df2=data.drop('type_of_combined_shot',axis=1)   
df2.head(10)


# In[46]:


data.drop('type_of_combined_shot',axis=1,inplace=True)   
data.head(10)


# In[47]:


data.power_of_shot.describe()


# In[142]:


pd.crosstab(data.power_of_shot,data.is_goal).plot(kind='bar')


# In[49]:


data['power_of_shot'].isnull().sum()


# In[50]:


data['power_of_shot'].fillna(data['power_of_shot'].mode()[0],inplace=True)
data.head(20)


# In[51]:


for i in range(1, len(data.index)) :
     if pd.isna(data.loc[i,'home/away']):
        data.loc[i,'home/away']=data.loc[i-1,'home/away']
     
data.head(10)


# In[52]:


bkpdata2=data.iloc[:,:]
bkpdata2.shape


# In[53]:


data.remaining_min.describe()


# In[54]:



data.remaining_min.median()


# In[55]:


pd.crosstab(data.remaining_min,data.is_goal)#.plot(kind='bar')


# In[57]:


data['remaining_min'].fillna(data['remaining_min'].median(),inplace=True)
data.head(20)


# In[58]:


data.remaining_sec.describe()


# In[59]:


data['remaining_sec'].isnull().sum()


# In[63]:


data.remaining_sec.median()


# In[64]:


data['remaining_sec'].fillna(data['remaining_sec'].median(),inplace=True)
data.head(20)


# In[65]:


for i in range(1, len(data.index)) :
     if pd.isna(data.loc[i,'shot_id_number']):
        data.loc[i,'shot_id_number']=i+1
     
data.head(20) 


# In[66]:


bkpdata3=data.iloc[:,:]
bkpdata3.head()


# In[67]:


for i in range (0,len(data.index)) :
    data.iloc[i,0]=data.iloc[i,0]+(data.iloc[i,4]/60)
data['remaining_min'].head()


# In[68]:


data.head(20)


# In[69]:


data=data.drop('remaining_sec',axis=1)  
data.head()


# In[70]:


data.shape


# In[71]:


df3=data.round({'remaining_min':2})
df3.head()


# In[77]:


cols=df3.columns.tolist()
cols


# In[79]:


cols=['shot_id_number','is_goal','remaining_min',
 'power_of_shot',
 'knockout_match',
 'distance_of_shot',
 'game_season',
 'area_of_shot',
 'shot_basics',
 'range_of_shot',
 'home/away',
 'type_of_shot']
cols


# In[94]:


dfd=df3[cols]

print(dfd.shape)
dfd.head()


# In[95]:


df4=dfd


# In[96]:


for i in range (0, len(dfd.index)):
    st=dfd.loc[i,'home/away']
    dfd.loc[i,'home/away']=st.split(' ')[1]
print(dfd['home/away'].value_counts())


# In[98]:


dfd['home/away']=labelencode.fit_transform(dfd['home/away'])
dfd.head()


# In[100]:


dummie_list1=['game_season','area_of_shot','shot_basics','range_of_shot','type_of_shot']
def credum(df,dummie_list):
    for i in dummie_list:
        dummies=pd.get_dummies( df[i], prefix=i, dummy_na=False )
        df=df.drop(i,1)
        df=pd.concat([df,dummies],axis=1)
    return df 


# In[101]:


new_df2=credum(dfd,dummie_list1)
new_df2.head(10)


# In[102]:


fin_test1=new_df2[new_df2['is_goal'].isna()]
fin_test1.head()


# In[103]:


fin_train1=new_df2.dropna(subset=['is_goal'])
fin_train1.shape


# In[104]:


fin_train1.head()


# In[105]:


X=fin_train1.iloc[:,2:].values
y=fin_train1.iloc[:,1].values


# In[109]:


from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()


# In[110]:


log_reg.fit(X,y)
log_reg.score(X,y)


# In[116]:


Xf=fin_test1.iloc[:,2:].values
yf=fin_test1.iloc[:,1].values


# In[117]:


y_prb=log_reg.predict_proba(Xf)[:,1]
y_prb


# In[121]:


type(y_prb)


# In[129]:


sub1=pd.DataFrame()
sub1


# In[130]:


sub1['shot_id_number']=fin_test1['shot_id_number']
sub1.head()


# In[131]:


sub1['is_goal']=y_prb
sub1.head()


# In[134]:


export_csv=sub1.to_csv(r'C:\Users\SAI SREEKAR KAJA\Desktop\python\hackethon\sub1.csv', index=None,header=True)
print(sub1)


# In[137]:


export_csv2=fin_train1.to_csv(r'C:\Users\SAI SREEKAR KAJA\Desktop\python\hackethon\fin_train1.csv', index=None,header=True)


# In[145]:


dfd['game_season'].describe()


# In[149]:


pd.crosstab(dfd.game_season,data.is_goal).plot(kind='bar')


# In[156]:


pd.crosstab(dfd['home/away'],dfd.is_goal).plot(kind='bar')


# In[155]:


df['home/away'].describe()


# In[161]:


dfd.dtypes


# In[163]:


pd.crosstab(dfd['knockout_match'],dfd.is_goal)#.plot(kind='bar')


# In[164]:


export_csv2=dfd.to_csv(r'C:\Users\SAI SREEKAR KAJA\Desktop\python\hackethon\dfd.csv', index=None,header=True)


# In[166]:


export_csv2=fin_test1.to_csv(r'C:\Users\SAI SREEKAR KAJA\Desktop\python\hackethon\fin_test1.csv', index=None,header=True)


# In[167]:


pd.crosstab(dfd['shot_basics'],dfd.is_goal).plot(kind='bar')


# In[168]:


pd.crosstab(dfd['range_of_shot'],dfd.is_goal).plot(kind='bar')


# In[169]:


pd.crosstab(dfd['area_of_shot'],dfd.is_goal).plot(kind='bar')


# In[ ]:




