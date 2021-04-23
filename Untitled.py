#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas


# In[4]:


Raw_Housing_data=pandas.read_csv("1. Regression - Module - (Housing Prices).csv")


# In[9]:


Raw_Housing_data


# In[10]:


Raw_Housing_data.dtypes


# In[11]:


Raw_Housing_data.head(10)


# In[12]:


Raw_Housing_data.info()


# In[13]:


Raw_Housing_data['Sale Price'].std()


# In[14]:


Raw_Housing_data


# In[15]:


Raw_Housing_data['Condition of the House'].unique()


# In[ ]:





# In[16]:


import numpy as np


# In[ ]:





# In[17]:


np.std(Raw_Housing_data['Sale Price'],ddof=1)


# In[18]:


Raw_Housing_data['Sale Price'].std()


# In[ ]:





# In[19]:


dir(np)


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


plt.plot(Raw_Housing_data['Sale Price'])


# In[22]:


plt.plot(Raw_Housing_data['Sale Price'],color='green')
plt.xlabel("Record Number")
plt.ylabel("Sale Price")
plt.title(" First graph")
plt.show()


# In[23]:


plt.plot(Raw_Housing_data['Sale Price'],marker='o',markerfacecolor='Blue',markersize='5')


# In[24]:


Raw_Housing_data.groupby('Condition of the House')['ID'].count()


# In[25]:


values=(30,1701,14031,5679,172)


# In[26]:


labels=('bad','Excellent','Fair','Good','okay')


# In[27]:


plt.pie(values,labels=labels)


# In[28]:


plt.scatter(x= Raw_Housing_data['Flat Area (in Sqft)'],y=Raw_Housing_data['Sale Price'],color='red')
plt.xlabel("area")
plt.ylabel("Selling Price")
plt.title("Selling Price vs area")
plt.show()


# In[29]:


plt.scatter(x=Raw_Housing_data['No of Bathrooms'],y=Raw_Housing_data['No of Floors'])


# In[30]:


Raw_Housing_data['Condition of the House']


# In[31]:


plt.boxplot(Raw_Housing_data['Age of House (in Years)'])


# In[32]:


Raw_Housing_data


# In[33]:


Raw_Housing_data['condition_sale']=0
Raw_Housing_data
 
for i in Raw_Housing_data['Condition of the House'].unique():
        Raw_Housing_data['condition_sale'][Raw_Housing_data['Condition of the House']== str(i)]= Raw_Housing_data['Sale Price'][Raw_Housing_data['Condition of the House']==str(i)].mean()
Raw_Housing_data  
plt.figure(dpi=100)
plt.bar(Raw_Housing_data['Condition of the House'].unique(),Raw_Housing_data['condition_sale'].unique())
plt.xlabel("Condition of the House")
plt.ylabel("MEan Sale Price")
plt.show()


# In[34]:


Raw_Housing_data


# In[35]:


zip_condition=Raw_Housing_data.groupby(['Condition of the House','Zipcode'])['Sale Price'].mean()
zip_condition


# In[36]:


zipcon=pd.pivot_table(Raw_Housing_data,index[''])


# In[37]:


plt.bar(labels,values)


# In[8]:


for  i in range (0,100,20):
    print(i*2)


# In[ ]:





# In[ ]:





# In[11]:


fact=1
for i in range(1,12):
    fact=fact*i
print(fact)


# In[ ]:


def newfunction(n,r)
for i in range()
    fact_n=fact_n*i


# In[1]:


Raw_Housing_data['total_area']=Raw_Housing_data['Flat Area(in Sqft)']+Raw_Housing_data['Lot Area (in Sqft)']
Raw_Housing_data['total_area']


# In[2]:


Raw_Housing_data


# In[4]:


Raw_Housing_data=pandas.read_csv("1. Regression - Module - (Housing Prices).csv")


# In[38]:


import pandas


# In[39]:


Raw_Housing_data=pandas.read_csv("1. Regression - Module - (Housing Prices).csv")


# In[40]:


Raw_Housing_data


# In[42]:


Raw_Housing_data['total_area']=Raw_Housing_data['Flat Area (in Sqft)']+Raw_Housing_data['Lot Area (in Sqft)']
Raw_Housing_data['total_area']


# In[45]:


Raw_Housing_data['Condition of the House'][Raw_Housing_data['Condition of the House']=='Fair']=1
Raw_Housing_data['Condition of the House'][Raw_Housing_data['Condition of the House']=='Okay']=0
Raw_Housing_data['Condition of the House'][Raw_Housing_data['Condition of the House']=='Bad']=0
Raw_Housing_data['Condition of the House'][Raw_Housing_data['Condition of the House']=='Good']=3
Raw_Housing_data['Condition of the House'][Raw_Housing_data['Condition of the House']=='Excellent']=3


# In[46]:


Raw_Housing_data['Condition of the House'].unique()


# In[47]:


Raw_Housing_data


# In[49]:


def year(value):
    return value.split()[-1]
Raw_Housing_data['year_sold']=Raw_Housing_data['Date House was Sold'].map(year)
Raw_Housing_data['year_sold'].head()


# In[50]:


Raw_Housing_data['Sale Price'].describe()


# In[51]:


import matplotlib.pyplot as plt


# In[52]:


plt.scatter(x=Raw_Housing_data['ID'],y=Raw_Housing_data['Sale Price'])


# In[53]:


import seaborn as sns


# In[54]:


sns.boxplot(x=Raw_Housing_data['Sale Price'])


# In[55]:


q1=Raw_Housing_data['Sale Price'].quantile(0.25)


# In[56]:


q3=Raw_Housing_data['Sale Price'].quantile(0.75)


# In[57]:


iqr=q3-q1


# In[58]:


iqr


# In[59]:


upper_limit=q3+(1.5*iqr)
lower_limit=q1-(1.5*iqr)


# In[60]:


upper_limit,lower_limit


# In[73]:


def limit_imputer(value):
     if value > upper_limit:
        return upper_limit
     if value < lower_limit:
        return lower_limit
     else:
        return value
    


# In[74]:


Raw_Housing_data['Sale Price']=Raw_Housing_data['Sale Price'].apply(limit_imputer)


# In[75]:


Raw_Housing_data['Sale Price']


# In[76]:


Raw_Housing_data['Sale Price'].max()


# In[77]:


Raw_Housing_data['Sale Price'].describe()


# In[79]:


Raw_Housing_data.dropna(inplace=True,
                       axis=0,
                       subset=['Sale Price'])


# In[80]:


Raw_Housing_data['Sale Price']


# In[83]:


Raw_Housing_data.info() 


# In[85]:


plt.hist(Raw_Housing_data['Sale Price'],bins=10,color='green')
plt.xlabel("Intervals")
plt.ylabel("Selling Price")
plt.title("Histograaaam")
plt.show()


# In[88]:


Raw_Housing_data.isnull().sum()


# In[90]:


Raw_Housing_data[numerical_columns]


# In[103]:


numerical_columns=['No of Bathrooms','Flat Area (in Sqft)','Lot Area (in Sqft)','Area of the House from Basement (in Sqft)',
                  'Latitude','Longitude','Living Area after Renovation (in Sqft)']


# In[104]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='median')
Raw_Housing_data[numerical_columns]=imputer.fit_transform(Raw_Housing_data[numerical_columns])


# In[105]:


Raw_Housing_data.info()


# In[106]:


Raw_Housing_data['Zipcode'].shape


# In[109]:


column=Raw_Housing_data["Zipcode"].values.reshape(-1,1)
column.shape


# In[111]:


column=Raw_Housing_data["Zipcode"].values.reshape(-1,1)
imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
Raw_Housing_data['Zipcode']=imputer.fit_transform(column)


# In[113]:


Raw_Housing_data['No of Times Visited'].unique()


# In[114]:





# In[121]:


mapping = {'None' : "0",
         'Once' : "1",
         'Twice' : "2",
         'Thrice' : "3",
         'Four' : "4"
    
         }
Raw_Housing_data['No of Times Visited']=Raw_Housing_data['No of Times Visited'].map(mapping)


# In[122]:


Raw_Housing_data['No of Times Visited'].unique()


# In[124]:


Raw_Housing_data['Ever Renovate']=np.where(Raw_Housing_data['Renovated Year']==0,'no','yes')


# In[125]:


Raw_Housing_data.head()


# In[127]:


import pandas as pd


# In[130]:


Raw_Housing_data['Purchase Year']=pd.DatetimeIndex(Raw_Housing_data['Date House was Sold']).year


# In[135]:


Raw_Housing_data['Years Since Rennovation']=np.where(Raw_Housing_data['Ever Renovate']=='yes',
                                                    abs(Raw_Housing_data['Purchase Year']-
                                                       Raw_Housing_data['Renovated Year']),0)


# In[136]:


Raw_Housing_data


# In[134]:


Raw_Housing_data.head()


# In[149]:


Raw_Housing_data.drop( columns = ['Purchase Year','Date House was Sold','Renovated Year'],inplace=True)


# In[150]:


Raw_Housing_data.head()


# In[151]:


Raw_Housing_data.drop( columns = ['total_area'],inplace=True)


# In[152]:


Raw_Housing_data.head()


# In[6]:


import pandas 


# In[7]:


Transformed_Housing_Data=pandas.read_csv('Raw_Housing_Prices3.csv')


# In[8]:


Transformed_Housing_Data


# In[9]:


Transformed_Housing_Data['Sale Price'].corr(Transformed_Housing_Data['Flat Area (in Sqft)'])


# In[12]:


import numpy as np


# In[13]:


np.corrcoef(Transformed_Housing_Data['Sale Price'],Transformed_Housing_Data['Flat Area (in Sqft)'])


# In[14]:


Transformed_Housing_Data.drop(columns=['ID']).corr()


# In[15]:


Transformed_Housing_Data['Flat Area (in Sqft)'].corr(Transformed_Housing_Data['Sale Price'])


# In[16]:


Transformed_Housing_Data.info()


# In[21]:


Transformed_Housing_Data.drop(columns='ID',inplace=True)


# In[ ]:


Transformed_Housing_Data['condition']

