#!/usr/bin/env python
# coding: utf-8

# Import all needed libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn import tree
from os import system


# reading the csv file/import data and apply Univariate analysis

# In[2]:


data = pd.read_csv("bank-full.csv")


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.dtypes


# In[6]:


data.describe()


# In[7]:


#any missing values? 
data.isnull().sum()


# In[8]:


columns = list(data)[0:-1] # Excluding Outcome column which has only 
data[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 
# Histogram of first 8 columns


# detecting and removeing outlirs:

# In[9]:



Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print (IQR)


# In[10]:


print(data.skew())


# In[11]:


low1 = data['age'].quantile(0.10)
high1 = data['age'].quantile(0.90)

low2 = data['balance'].quantile(0.10)
high2 = data['balance'].quantile(0.90)

low3 = data['day'].quantile(0.10)
high3 = data['day'].quantile(0.90)

low4 = data['duration'].quantile(0.10)
high4 = data['duration'].quantile(0.90)

low5 = data['campaign'].quantile(0.10)
high5 = data['campaign'].quantile(0.90)

low6 = data['pdays'].quantile(0.10)
high6 = data['pdays'].quantile(0.90)

low7 = data['previous'].quantile(0.10)
high7 = data['age'].quantile(0.90)


# In[12]:


data['age'] = np.where(data['age'] <low1, low1, data['age'])
data['age'] = np.where(data['age'] >high1, high1, data['age'])

data['balance'] = np.where(data['balance'] <low2, low2, data['balance'])
data['balance'] = np.where(data['balance'] >high2, high2, data['balance'])

data['day'] = np.where(data['day'] <low3, low3, data['day'])
data['day'] = np.where(data['day'] >high3, high3, data['day'])

data['duration'] = np.where(data['duration'] <low4, low4, data['duration'])
data['duration'] = np.where(data['duration'] >high4, high4, data['duration'])

data['campaign'] = np.where(data['campaign'] <low5, low5, data['campaign'])
data['campaign'] = np.where(data['campaign'] >high5, high5, data['campaign'])

data['pdays'] = np.where(data['pdays'] <low6, low6, data['pdays'])
data['pdays'] = np.where(data['pdays'] >high6, high6, data['pdays'])

data['previous'] = np.where(data['previous'] <low7, low7, data['previous'])
data['previous'] = np.where(data['previous'] >high7, high7, data['previous'])



# In[13]:


print(data.skew())


# Bi-Variate Analysis

# In[14]:


sns.pairplot(data)


# In[ ]:





# In[15]:


sns.heatmap(data.corr(), annot=True)

#there is a positive corralation between pdays and number of days that passed by after the client was last contacted from a previous campaign and the number of contacts performed before this campaign and for this client


# Normlization and Scalling

# In[16]:


from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale


# In[17]:


data['balance'] = std_scale.fit_transform(data[['balance']]) # returns z-scores of the values of the attribute


# In[18]:


data['balance'].head()


# In[19]:


data['balance'].min(), data['balance'].max() 


# In[20]:


data['balance'].describe()  


# Replace strings with an integer

# In[21]:


for feature in data.columns:
    if data[feature].dtype == 'object':
        data[feature] = pd.Categorical(data[feature])
data.head(10)


# In[22]:


data.describe()


# In[23]:


data.head(10)


# In[24]:


print(data.job.value_counts())
print("--------------")
print(data.marital.value_counts())
print("--------------")
print(data.education.value_counts())
print("--------------")
print(data.default.value_counts())
print("--------------")
print(data.housing.value_counts())
print("--------------")
print(data.loan.value_counts())
print("--------------")
print(data.contact.value_counts())
print("--------------")
print(data.month.value_counts())
print("--------------")
print(data.poutcome.value_counts())


# In[25]:


replaceStruct = {
                "job": {"blue-collar": 1, "management": 2, "technician": 3 ,"admin.": 4 ,"services" :5 ,"retired" :6 ,"self-employed" :7 ,"entrepreneur": 8 ,"unemployed":9 ,"housemaid" :10 ,"student" :11 ,"unknown" :-1},
                "marital": {"married": 1, "single":2 , "divorced": 3},
                 "education": {"secondary": 1, "tertiary":2, "primary": 3, "unknown": -1},
                #"default": {"no": 1, "yes": 2},
                #"housing":     {"yes": 1, "no": 2 },
                #"loan": {"no": 1, "yes": 2},
                "contact":     {"cellular": 1, "telephone": 2, "unknown": -1},
                "month":     {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "oct": 9, "sep": 10, "nov": 11, "dec": 12},
                "poutcome":     {"failure": 1, "success": 2, "other": 3, "unknown": -1}
                    }

oneHotCols=["default","housing","loan"]


# In[26]:


data=data.replace(replaceStruct)
data=pd.get_dummies(data, columns=oneHotCols)
data.head()


# splitting data into training and test set 70:30 

# In[27]:


from sklearn.model_selection import train_test_split

X = data.drop('Target',axis=1)     # Predictor feature columns (8 X m)
Y = data['Target']   # Predicted class (1=True, 0=False) (1 X m)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# 1 is just any random seed number


# In[28]:


print("{0:0.2f}% data is in training set".format((len(x_train)/len(data.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(x_test)/len(data.index)) * 100))


# In[29]:


print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(data.loc[data['Target'] == "no"]), (len(data.loc[data['Target'] == "no"])/len(data.index)) * 100))
print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(data.loc[data['Target'] == "yes"]), (len(data.loc[data['Target'] == "yes"])/len(data.index)) * 100))
print("")
print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == "no"]), (len(y_train[y_train[:] == "no"])/len(y_train)) * 100))
print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == "yes"]), (len(y_train[y_train[:] == "yes"])/len(y_train)) * 100))
print("")
print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == "no"]), (len(y_test[y_test[:] == "no"])/len(y_test)) * 100))
print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == "yes"]), (len(y_test[y_test[:] == "yes"])/len(y_test)) * 100))
print("")


# In[30]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Fit the model
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
#predict on test
y_predict = model.predict(x_test)

coef_df = pd.DataFrame(model.coef_)
coef_df['intercept'] = model.intercept_
print(coef_df)


# In[31]:


model_score = model.score(x_test, y_test)
print(model_score)


# In[32]:


cm = metrics.confusion_matrix(y_test, y_predict, labels=["no", "yes"])

df_cm = pd.DataFrame(cm, index = [i for i in ["no","yes"]], columns = [i for i in ["Predict no","Predict yes"]])
plt.figure(figsize = (5,5))
sns.heatmap(df_cm, annot=True)


# In[33]:


print(cm)


# In[34]:


TP = cm[0,0]
FN = cm[1,0]
FP = cm[0,1]
TN = cm[1,1]


# In[35]:


acc_1 = (TP+TN)/(TP+TN+FN+FP)
pre_1 = TP/(TP+FP)
rec_1 = TP/(TP+FN)
f1_1 = 2*((pre_1*rec_1)/(pre_1+rec_1))


# In[36]:


print("Accuracy:", acc_1)
print("Precision:", pre_1)
print("Recall:", rec_1)
print("F1:", f1_1)


# 

# In[37]:


dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(x_train, y_train)


# In[38]:


print(dTree.score(x_train, y_train))
print(dTree.score(x_test, y_test))


# In[39]:


dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=1)
dTreeR.fit(x_train, y_train)
print(dTreeR.score(x_train, y_train))
print(dTreeR.score(x_test, y_test))


#Better when we reduce the overfitting!


# In[ ]:





# In[40]:


print(dTreeR.score(x_test , y_test))
y_predict = dTreeR.predict(x_test)

cm=metrics.confusion_matrix(y_test, y_predict, labels=["no", "yes"])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# In[41]:


TP = cm[0,0]
FN = cm[1,0]
FP = cm[0,1]
TN = cm[1,1]


# In[42]:


acc_2 = (TP+TN)/(TP+TN+FN+FP)
pre_2 = TP/(TP+FP)
rec_2 = TP/(TP+FN)
f1_2 = 2*((pre_2*rec_2)/(pre_2+rec_2))


# In[43]:


print("Accuracy:", acc_2)
print("Precision:", pre_2)
print("Recall:", rec_2)
print("F1:", f1_2)


# Bagging

# In[44]:


from sklearn.ensemble import BaggingClassifier

bgcl = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(x_train, y_train)


# In[45]:


y_predict = bgcl.predict(x_test)

print(bgcl.score(x_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=['no', 'yes'])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# In[46]:


TP = cm[0,0]
FN = cm[1,0]
FP = cm[0,1]
TN = cm[1,1]


# In[47]:


acc_3 = (TP+TN)/(TP+TN+FN+FP)
pre_3 = TP/(TP+FP)
rec_3 = TP/(TP+FN)
f1_3 = 2*((pre_3*rec_3)/(pre_3+rec_3))


# In[48]:


print("Accuracy:", acc_3)
print("Precision:", pre_3)
print("Recall:", rec_3)
print("F1:", f1_3)


# AdaBoosting

# In[49]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators=10, random_state=1)
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
abcl = abcl.fit(x_train, y_train)


# In[50]:


y_predict = abcl.predict(x_test)
print(abcl.score(x_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=['no', 'yes'])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# In[51]:


TP = cm[0,0]
FN = cm[1,0]
FP = cm[0,1]
TN = cm[1,1]


# In[52]:


acc_4 = (TP+TN)/(TP+TN+FN+FP)
pre_4 = TP/(TP+FP)
rec_4 = TP/(TP+FN)
f1_4 = 2*((pre_4*rec_4)/(pre_4+rec_4))


# In[53]:


print("Accuracy:", acc_4)
print("Precision:", pre_4)
print("Recall:", rec_4)
print("F1:", f1_4)


# GradinatBoost

# In[54]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 50,random_state=1)
gbcl = gbcl.fit(x_train, y_train)


# In[55]:


y_predict = gbcl.predict(x_test)
print(gbcl.score(x_test, y_test))
cm=metrics.confusion_matrix(y_test, y_predict,labels=['no', 'yes'])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# In[56]:


TP = cm[0,0]
FN = cm[1,0]
FP = cm[0,1]
TN = cm[1,1]


# In[57]:


acc_5 = (TP+TN)/(TP+TN+FN+FP)
pre_5 = TP/(TP+FP)
rec_5 = TP/(TP+FN)
f1_5 = 2*((pre_5*rec_5)/(pre_5+rec_5))


# In[58]:


print("Accuracy:", acc_5)
print("Precision:", pre_5)
print("Recall:", rec_5)
print("F1:", f1_5)


# In[ ]:





# In[ ]:





# In[59]:


final = {'Logistic Regression':[acc_1, pre_1, rec_1, f1_1],
         'Decision Tree':[acc_2, pre_2, rec_2, f1_2],
        'Bagging':[acc_3, pre_3, rec_3, f1_3],
        'AdaBoost':[acc_4, pre_4, rec_4, f1_4],
        'Gradient Boosting':[acc_5, pre_5, rec_5, f1_5]} 


# In[60]:


df = pd.DataFrame(final) 
df = pd.DataFrame(final, index =['Accuracy', 'Precision', 'Recall', 'F1'])


# In[61]:


df


# In[62]:


# From what I got in this df we can notice that, all of the models are give us a good values in testing. if that is the case choseing less expencive model will be the better option(LR or DT)
# Also we to take a look at the loss and gain that we are going to have if someone will subscribe but he/she was not reached, to compare it with if someone will not subcribe and he/she was reached. 
#based of these cost we can deside which model to go with. 


# In[ ]:




