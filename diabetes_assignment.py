#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import shapiro 
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix


# In[69]:


raw_data = pd.read_csv('diabetes.csv')

raw_data.sample(5)


# ### The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.
# 
# It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 768 observations with 8 input variables and 1 output variable. Missing values are believed to be encoded with zero values. The variable names are as follows:
# 
# 1. Number of times pregnant.
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
# 3. Diastolic blood pressure (mm Hg).
# 4. Triceps skinfold thickness (mm).
# 5. 2-Hour serum insulin (mu U/ml).
# 6. Body mass index (weight in kg/(height in m)^2).
# 7. Diabetes pedigree function.
# 8. Age (years).
# 9. Class variable (0 or 1).

# In[70]:


for x in raw_data.drop('Outcome',axis=1).columns:
    print('Distribution chart for ',format(x))
#    print(shapiro(raw_data[(raw_data[x] != 0 )][x]))
    s_stats, s_p_value = shapiro(raw_data[(raw_data[x] != 0 )][x])
    sns.displot( data = raw_data, x=x)
    plt.show()
    if s_p_value >= 0.05:
        print('Based on Shapiro-Wilk test data for',x,'is normally distributed.\n\n')
    else:
        print('Based on Shapiro-Wilk test data for',x,'is NOT normally distributed.\n\n')


# In[71]:


corr = raw_data.drop('Outcome',axis=1).corr(method='pearson')

fig, ax =  plt.subplots(figsize = (20,10))
ax = sns.heatmap(corr,cmap = 'Blues',fmt='.2g',annot=True)


# In[72]:


for x in raw_data.drop('Outcome',axis=1).columns:
    sns.violinplot(data = raw_data, x='Outcome', y=x)
    plt.show()


# In[73]:


raw_data.columns


# In[74]:


df1 = raw_data.loc[raw_data.Outcome == 1]
df0 = raw_data.loc[raw_data.Outcome == 0]

df1 = df1.replace({'Glucose':0},np.median(df1['Glucose']))
df0 = df0.replace({'Glucose':0},np.median(df0['Glucose']))

union = [df1,df0]
df = pd.concat(union)

df1 = df.loc[df.Outcome == 1]
df0 = df.loc[df.Outcome == 0]

df1 = df1.replace({'BloodPressure':0},np.median(df1['BloodPressure']))
df0 = df0.replace({'BloodPressure':0},np.median(df0['BloodPressure']))

union = [df1,df0]
df = pd.concat(union)

df1 = df.loc[df.Outcome == 1]
df0 = df.loc[df.Outcome == 0]

df1 = df1.replace({'BMI':0},np.median(df1['BMI']))
df0 = df0.replace({'BMI':0},np.median(df0['BMI']))

union = [df1,df0]
df = pd.concat(union)

df1 = df.loc[df.Outcome == 1]
df0 = df.loc[df.Outcome == 0]

df1 = df1.replace({'DiabetesPedigreeFunction':0},np.median(df1['DiabetesPedigreeFunction']))
df0 = df0.replace({'DiabetesPedigreeFunction':0},np.median(df0['DiabetesPedigreeFunction']))

union = [df1,df0]
df = pd.concat(union)


# In[75]:


for x in ['SkinThickness', 'Insulin']:
    df[x] = df[x].replace(0,np.nan)
    
df.sample(10)


# In[76]:


imputer = KNNImputer(n_neighbors= 30)

df_knn30 = pd.DataFrame(imputer.fit_transform(df),columns =  df.columns)


# In[77]:


imputer = KNNImputer(n_neighbors= 5)

df_knn5 = pd.DataFrame(imputer.fit_transform(df),columns =  df.columns)


# In[78]:


df1 = df.loc[df.Outcome == 1]
df0 = df.loc[df.Outcome == 0]

imputer = KNNImputer(n_neighbors= 30)

df1_knn30 = pd.DataFrame(imputer.fit_transform(df1),columns =  df.columns)
df0_knn30 = pd.DataFrame(imputer.fit_transform(df0),columns =  df.columns)

union = [df1_knn30,df0_knn30]
dfs_knn30 = pd.concat(union)


# In[79]:


imputer = KNNImputer(n_neighbors= 5)

df1_knn5 = pd.DataFrame(imputer.fit_transform(df1),columns =  df.columns)
df0_knn5 = pd.DataFrame(imputer.fit_transform(df0),columns =  df.columns)

union = [df1_knn5,df0_knn5]
dfs_knn5 = pd.concat(union)


# In[80]:


df_knn30


# df1 = df.loc[df.Outcome == 1]
# df0 = df.loc[df.Outcome == 0]
# 
# df1 = df1.replace({'Insulin':0},np.mean(df1['Insulin']))
# df0 = df0.replace({'Insulin':0},np.mean(df0['Insulin']))
# 
# union = [df1,df0]
# df = pd.concat(union)
# 
# df1 = df.loc[df.Outcome == 1]
# df0 = df.loc[df.Outcome == 0]
# 
# df1 = df1.replace({'SkinThickness':0},np.mean(df1['SkinThickness']))
# df0 = df0.replace({'SkinThickness':0},np.mean(df0['SkinThickness']))
# 
# union = [df1,df0]
# df = pd.concat(union)

# In[81]:


sns.violinplot(data = raw_data, x='Outcome', y = 'Insulin')
plt.show()
sns.violinplot(data = df_knn30, x='Outcome', y = 'Insulin')
plt.show()


# In[82]:


sns.violinplot(data = raw_data, x='Outcome', y = 'Insulin')
plt.show()
sns.violinplot(data = df_knn5, x='Outcome', y = 'Insulin')
plt.show()


# In[83]:


sns.violinplot(data = raw_data, x='Outcome', y = 'Insulin')
plt.show()
sns.violinplot(data = dfs_knn30, x='Outcome', y = 'Insulin')
plt.show()


# In[84]:


sns.violinplot(data = raw_data, x='Outcome', y = 'Insulin')
plt.show()
sns.violinplot(data = dfs_knn5, x='Outcome', y = 'Insulin')
plt.show()


# In[85]:


sns.displot(data=df_knn30, x = 'Insulin')
plt.show()

sns.displot(data=dfs_knn30, x = 'Insulin')
plt.show()


# In[86]:


sns.displot(data=df_knn5, x = 'Insulin')
plt.show()

sns.displot(data=dfs_knn5, x = 'Insulin')
plt.show()


# In[87]:


shapiro(dfs_knn5.Insulin)


# In[88]:


shapiro(df_knn5.Insulin)


# In[89]:


sns.displot(data=raw_data, x = 'Insulin')
plt.show()

sns.displot(data=df_knn5, x = 'Insulin')
plt.show()


# In[90]:


df_val = pd.DataFrame()

for x in df.drop('Outcome',axis=1).columns:
    df_val[x]=raw_data[x].replace(0,np.nan)


# In[91]:


(df_val.isnull().sum()/768)


# In[92]:


sns.displot( data = df, x='Insulin')


# In[93]:


df1 = df.loc[df.Outcome == 1]
df0 = df.loc[df.Outcome == 0]


# In[94]:


sns.displot( data = df, x='Insulin')


# In[95]:


sns.displot( data = df, x='SkinThickness')


# In[96]:


X = dfs_knn5.drop({'Outcome'}, axis=1)

y = dfs_knn5.pop('Outcome')


# In[97]:


X_train, X_test, train_labels, test_labels = train_test_split(X,y,test_size=0.20, random_state=1)
reg_dt_model_gini = DecisionTreeClassifier(criterion = 'gini',max_depth=7,min_samples_split=80,random_state=0)
reg_dt_model_gini.fit(X_train, train_labels)

probs = reg_dt_model_gini.predict_proba(X_train)[:,1]
auc_dev = roc_auc_score(train_labels, probs)
print(auc_dev)
fpr, tpr, thresholds = roc_curve(train_labels, probs)
plt.plot([0,1],[0,1], linestyle='--')
plt.plot(fpr,tpr,marker='o')
plt.show()

probt = reg_dt_model_gini.predict_proba(X_test)[:,1]
auc_test = roc_auc_score(test_labels, probt)
print(auc_test)
fpr, tpr, thresholds = roc_curve(test_labels, probt)
plt.plot([0,1],[0,1], linestyle='--')
plt.plot(fpr,tpr,marker='o')
plt.show()


# In[98]:


pd.DataFrame(reg_dt_model_gini.feature_importances_, columns = ['Imp'], index = X_train.columns).sort_values(by='Imp',ascending = False)


# In[99]:


ytrain_predict = reg_dt_model_gini.predict(X_train)
ytest_predict = reg_dt_model_gini.predict(X_test)

#Train data Confusion Matrix
print(confusion_matrix(train_labels, ytrain_predict))
#Train Data Accuracy
print(reg_dt_model_gini.score(X_train,train_labels) )
print(classification_report(train_labels, ytrain_predict))

#Test data Confusion Matrix
print(confusion_matrix(test_labels, ytest_predict))
#Test Data Accuracy
print(reg_dt_model_gini.score(X_test,test_labels) )
print(classification_report(test_labels, ytest_predict))


# In[ ]:




