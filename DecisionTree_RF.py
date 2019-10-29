
# coding: utf-8

# # Decision Trees and Random Forests in Python

# In[193]:

import pandas as pd #for data munging
import numpy as np # for mathematicla operations
from collections import Counter # like table function in R
import matplotlib.pyplot as plt # for visualization
import seaborn as sns# for visualization
get_ipython().magic(u'matplotlib inline')


# In[194]:

#read the train data
train=pd.read_csv("C:\\Users\\singh\\Desktop\\Imarticus Projects\\Intrusion Detection Using Ensemble Learning Decision Tree\\Network_Intrusion_Train_data.csv")
#read the test data
test=pd.read_csv("C:\\Users\\singh\\Desktop\\Imarticus Projects\\Intrusion Detection Using Ensemble Learning Decision Tree\\Network_Intrusion_Validate_data.csv")


# In[195]:

#merge the 2 dataframes
all=pd.concat([train,test],axis=0)


# In[196]:

all.shape


# In[75]:

all.head()


# In[76]:

all.info()


# In[77]:

all.describe()


# In[78]:

all.columns


# In[ ]:

#identify the categorical & the continuous variables


# In[79]:

all.select_dtypes(include=['object']).head()# to see all categorical columns
#df.select_dtypes(include=[np.number]) # to see all numerical/ continuous columns


# In[ ]:

# lets study barplots of each categorical columns


# In[81]:

print all['class'].value_counts()
print("\n") # print a eppty line
print Counter(all['class'])


# In[82]:

sns.countplot(all['class'])


# In[83]:

sns.countplot("protocol_type",hue="class",data=all)
# what odes this tell us?
# is Protocol type important?
#yes esp udp and icmp as most of them are normal & anamaly repectively


# In[84]:

sns.countplot("service",hue="class",data=all)


# In[85]:

all['service'].value_counts()


# In[86]:

sns.countplot("flag",hue="class",data=all)


# In[87]:

all['flag'].value_counts()


# In[ ]:

# let us run a normal decision tree using all variables


# In[42]:

from sklearn.tree import DecisionTreeClassifier


# In[43]:

dtree = DecisionTreeClassifier()


# In[88]:

# convert the categorical variables to dummy variables
dum1=pd.get_dummies(all['flag'],drop_first=True)
dum2=pd.get_dummies(all['service'],drop_first=True)
dum3=pd.get_dummies(all['protocol_type'],drop_first=True)
#add all the dummy variable to the original data frame
df_new=pd.concat([all,dum1,dum2,dum3],axis=1)


# In[90]:

#split data back into the same train and test


# In[97]:

train_new=df_new.head(len(train))#split back into original train and test
test_new=df_new.tail(len(test))


# In[108]:

X_train=train_new.drop(['class','service','flag','protocol_type'],axis=1)
y_train=train_new['class']
X_test=test_new.drop(['class','service','flag','protocol_type'],axis=1)
y_test=test_new['class']


# In[109]:

dtree.fit(X_train,y_train)


# # Prediction and Evaluation

# In[110]:

predictions = dtree.predict(X_test)


# In[111]:

from sklearn.metrics import classification_report,confusion_matrix


# In[112]:

print(classification_report(y_test,predictions))


# In[113]:

print(confusion_matrix(y_test,predictions))


# # Tree Visualization¶

# In[124]:

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 


# In[118]:

features = list(train.columns[0:-1])
#features--> all but the dependent variable


# In[156]:

#tree visualiztion did not work as i dont have the software graphviz
#dot_data = StringIO()  
#export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

#graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#Image(graph[0].create_png()) 


# In[132]:

from sklearn import tree
tree.export_graphviz(dtree,out_file='tree.dot')   


# # Random Forests

# In[150]:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500,max_features=10)
rfc.fit(X_train, y_train)


# In[151]:

rfc_pred = rfc.predict(X_test)


# In[152]:

print(confusion_matrix(y_test,rfc_pred))


# In[153]:

print(classification_report(y_test,rfc_pred))


# In[ ]:




# In[165]:

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# In[ ]:




# In[177]:

feature_names = train_new.columns # e.g. ['A', 'B', 'C', 'D', 'E']

f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X_train.shape[1]), importances[indices],
    color="b", 
    align="center")
plt.xticks(range(X_train.shape[1]), feature_names)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("Name of the feature", fontsize = 18)


# In[178]:

# let us try to use only important features instead of using everything


# In[183]:

train.columns


# In[197]:

pd.crosstab(train['land'],train['class'], margins=False)#land is not important


# In[191]:

pd.crosstab(train['duration']<50,train['class'], margins=False)
#duration not important


# In[192]:

pd.crosstab(train['src_bytes']<50,train['class'], margins=False)
#u'src_bytes' is important


# In[ ]:

#homework, 
#do this for all remining variables
#identify the important variables
#and run a model with only the important variables


# In[ ]:



