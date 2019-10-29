
# coding: utf-8

# In[1]:

import pandas as pd# import the library pandas
import numpy as np
from collections import Counter


# In[108]:

df=pd.read_csv("C:/Users/singh/Desktop/Imarticus Projects/telecom_project.csv") # read a csv file in python


# In[109]:

df.head()


# In[110]:

df.info() # to see basic data types and missing values for all columns
df.describe()
# note that describe is not giving stats of TotalCharges. That means some thg is fishy


# In[111]:

df['gender'].value_counts() # to get the counts of the categories just like table in R
#or
Counter(df['gender'])


# In[112]:

sum((df['gender']).isnull())# to find the total missing values in a column


# In[113]:

# find the # of missing values in the column Totl charges
sum(df['TotalCharges'].isnull())


# In[114]:

df.columns # to get a lsit of all column names


# In[115]:

# count of churn vs No churn
df["Churn"].value_counts()
from collections import Counter
#or
Counter(df["Churn"])


# In[116]:

# get the mean of TotalCharges
#df['TotalCharges'].


# In[117]:

Counter(df['gender'])


# In[118]:

df[(df['gender'].isnull())]


# In[119]:

# fill the missing gender as male
df['gender'].fillna(value="Male",inplace=True)


# In[120]:

#fill the missing values in TotalCharges with their mean
#there are some blanks (not missing value) in total charges. 
#Lets replace them with the mean
#
type(df['TotalCharges'][0])# this is a string. Thats y we didnot get the stats of Total
#charges earlier
temp=df[df['TotalCharges']!=" "]# subset out all non blank rows
temp.reset_index(inplace=True,drop=True)
# now convert the total charges in them to float
temp['TotalCharges']=[float(temp['TotalCharges'][i]) for i in range(len(temp))]
#ignore the warning
np.mean(temp['TotalCharges']) # now we have the mean
blank_rows=df[df['TotalCharges']==" "].index.tolist()
df['TotalCharges'].iloc[blank_rows]=str(np.mean(temp['TotalCharges']))
df['TotalCharges']=[float(df['TotalCharges'][i]) for i in range(len(df))]


# In[101]:

# the above code if run twice will not work. It will work only once


# In[121]:

#fill the missing values in MonthlyCharges with their mean
df['MonthlyCharges'].fillna(value=df['MonthlyCharges'].mean(),inplace=True)


# In[122]:

# hoe to delete rows and columns with msisng value
#lets read the csv again
df_test=pd.read_csv("C:/Users/singh/Desktop/Imarticus Projects/telecom_project.csv")


# In[123]:

df_test.isnull().sum() # to see total missing value per column


# In[124]:

df_test.dropna(axis=0,inplace=True)


# In[125]:

df_test.shape


# In[126]:

# create a new column called BILL 
#which is the sum of TotalCharges and MonthlyCharges
df['BILL']=df['TotalCharges']+df['MonthlyCharges']


# In[127]:

df.shape


# In[128]:

df.columns


# In[129]:

# create a data frame df1 with columns CustomerID and 
#TotalCharges & only first 4 rows
#customerID=['7590-VHVEG','5575-GNVDE']


# In[130]:

df1=df.iloc[0:4,[0,19]]
df1
#Subset by Column
df2=df.loc[:,["customerID","Churn"]]
df2.head()


# In[131]:

# create a df called df2 with columns CustomerID and Churn and row #'s 2 to 7


# In[132]:

df2=df.iloc[2:7,[0,20]]
df2


# In[133]:

print df1
print df2


# In[134]:

# merge df1 and df2 


# In[135]:

# left merge/join
leftjoin=pd.merge(df1,df2,how="left",on=['customerID'])
leftjoin


# In[136]:

# inner merge/join


# In[137]:

innerjoin=pd.merge(df1,df2,how="inner",on=['customerID'])
innerjoin


# In[138]:

df2.reset_index(drop=True,inplace=True)# reset index in a dataframe


# In[139]:

# concat 2 dataframes


# In[140]:

df1


# In[141]:

#create a new dataframe called df3 with row #'s 5 to 8 and columns
# CustomerID and TotalCharges
df3=df.iloc[5:8,[0,19]]
df3


# In[142]:

total_df=pd.concat([df1,df3])


# In[143]:

total_df


# In[144]:

total_df.reset_index(inplace=True,drop=True)
total_df


# In[145]:

df.head()


# In[146]:

# Who talks more 
# male or female 
# find the mean TotalCharges of all female & male
df.groupby(['gender'])['TotalCharges'].mean()


# In[147]:

# who had a higher BILL , churned or no Churned
df.groupby(['Churn'])['BILL'].mean()


# In[148]:

# find the max bill amount out of all Male and female
# who Churned and Didnot churn
df.groupby(['gender','Churn'])['BILL'].max()


# In[149]:

#get me all customerID's who are female and have churned
df[(df['gender']=="Female")&(df['Churn']=="Yes")]['customerID']


# In[ ]:

import seaborn as sns


# In[ ]:




# In[ ]:




# In[ ]:



