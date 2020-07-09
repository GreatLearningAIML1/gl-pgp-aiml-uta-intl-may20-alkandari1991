#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Question number 1


# In[35]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[36]:


data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")


# In[37]:


type(data)


# In[38]:


data.info()


# In[39]:


data.describe()


# In[40]:


data.shape


# In[41]:


data.isnull().values.any()


# In[42]:


data.isnull().sum()


# In[88]:


data["Personal Loan"].nunique()


# In[89]:


data.groupby(["Personal Loan"]).mean()


# In[39]:


data.head()


# In[40]:


#Question Number 2


# In[41]:


print(data.nunique())


# In[42]:


columns = list(data)[0:-1]
data[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 


# In[12]:


data['Mortgage'].value_counts() 


# In[13]:


data["Mortgage"].nunique()


# In[11]:


data['CCAvg'].value_counts()[0]


# In[45]:


data.apply(pd.Series.value_counts)


# In[46]:


data['Age'].value_counts()


# In[47]:


plt.figure(figsize=(10,5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', center = 1 )  # heatmap
plt.show()


# In[48]:


sns.scatterplot(x=data['Experience'],y='Age', data=data)


# In[49]:


sns.pairplot(data)


# In[ ]:


sns.distplot(data)


# In[50]:


data.groupby(by=['Family'])['CCAvg'].sum().reset_index().sort_values(['CCAvg']).tail(10).plot(x='Family',
                                                                                                           y='CCAvg',
                                                                                                           kind='bar',
                                                                                                           figsize=(15,5))
plt.show()


# In[51]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x='Age', y='Personal Loan', data=data, palette='muted') 


# In[52]:


#Question 3


# In[43]:


from sklearn.model_selection import train_test_split

X = data.drop('Personal Loan',axis=1)
Y = data['Personal Loan']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[62]:


X = pd.get_dummies(X, columns=['Family'])
X = pd.get_dummies(X, columns=['Education'])


# In[ ]:





# In[68]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# In[91]:


import statsmodels.api as sm

logit = sm.Logit(y_train, sm.add_constant(x_train))
lg = logit.fit()


# In[69]:


logreg.classes_


# In[70]:


logreg.predict_proba(x_test)[:,0]


# In[71]:


logreg.predict(x_test)


# In[72]:


logreg.predict_proba(x_test)[0,:]


# In[73]:


y_predict = logreg.predict(x_test)


# In[74]:


y_predict


# In[75]:


Y_predict = logreg.predict_proba(x_test)[:,1] 
print("Traing Acc.:", logreg.score(x_train,y_train))
print("Testing Acc.:", logreg.score(x_test,y_test))
print()
print("Recall:", recall_score(y_test, y_predict))
print("Precisiopn:", precision_score(y_test, y_predict))
print()
print("F1 Score:", f1_score(y_test, y_predict))
print()
print("ROC AUC Score:", roc_auc_score(y_test, y_predict))


# In[78]:


for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    Y_predict = logreg.predict_proba(x_test)[:,1] > i
    print(i)
    print("Traing Acc.:", logreg.score(x_train,y_train))
    print("Testing Acc.:", logreg.score(x_test,y_test))
    print()
    print("Recall:", recall_score(y_test, y_predict))
    print("Precisiopn:", precision_score(y_test, y_predict))
    print()
    print("F1 Score:", f1_score(y_test, y_predict))
    print()
    print("ROC AUC Score:", roc_auc_score(y_test, y_predict))
    print("=================================================")


# In[82]:


print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(data.loc[data['Personal Loan'] == 1]), (len(data.loc[data['Personal Loan'] == 1])/len(data.index)) * 100))
print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(data.loc[data['Personal Loan'] == 0]), (len(data.loc[data['Personal Loan'] == 0])/len(data.index)) * 100))
print("")
print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))
print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))
print("")
print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))
print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
print("")


# In[83]:


from sklearn import metrics

from sklearn.linear_model import LogisticRegression

# Fit the model on train
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
#predict on test
y_predict = model.predict(x_test)


coef_df = pd.DataFrame(model.coef_)
coef_df['intercept'] = model.intercept_
print(coef_df)


# In[84]:


model_score = model.score(x_test, y_test)
print(model_score)


# In[85]:


cm=metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)


# In[ ]:


#Question 5


# In[86]:


model.get_params()


# In[87]:


#Question 6


# To decide, if this is a real-life situation, we need to calculate our risks and where do we gain money and where we would lose money
# Let us take the possible cases as an example. 
# 1- If we gave a personal loan to individuals and they return the money to the bank we will gain profits. This is good for business.
# 2 -If we did not give loans, we will neither loss nor gain This is bad. 
# 3- If we gave loans and people did not return the money, we will lose money. This is the worst case.
# 4- If we did not give a loan for people who will not return the moeny. This is good for busniess.
# 
# Situation 2 is type I error, or as they call it False Positive (FP). Where people are going to return the money, but we decided not to give them the loan. This is for sure bad. But the worse than this is the type II error False Negative (FN) where we give a person a loan and he/she will not return it. This will cost the business much more money.
# 
# In this case we want to get a model that have the highest rate of precision to save the business money.
# 

# In[ ]:




