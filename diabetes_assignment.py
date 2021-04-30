#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import shapiro 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
#from sklearn.preprocessing import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix


# In[69]:


raw_data = pd.read_csv(r'C:\Users\saran\Desktop\up-skill\PGP - DSBA\Data Mining\diabetes.csv')

raw_data.sample(5)


# ### The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.
# 
# It is a binary (2-class) classification problem. The number of observations 
# for each class is not balanced. There are 768 observations with 8 input variables 
# and 1 output variable. Missing values are believed to be encoded with zero values. 
# The variable names are as follows:
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
    
#Univariate analysis checking the normal distribution


# for x in raw_data.drop('Outcome',axis=1).columns:
#     print('Distribution chart for ',format(x))
#     s_stats, s_p_value = shapiro(raw_data[(raw_data[x] != 0 )][x])
#     sns.displot( data = raw_data, x=x)
#     plt.show()
#     if s_p_value >= 0.05:
#         print('Based on Shapiro-Wilk test data for',x,'is normally distributed.\n\n')
#     else:
#         print('Based on Shapiro-Wilk test data for',x,'is NOT normally distributed.\n\n')

# In[71]:

corr = raw_data.corr(method='pearson')

fig, ax =  plt.subplots(figsize = (20,10))
ax = sns.heatmap(corr,cmap = 'Blues',fmt='.2g',annot=True)
plt.show()

# In[72]:

#sns.pairplot(data = raw_data)
#plt.show()

# In[73]:

print(raw_data.columns)

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

# In[76]:

corr = raw_data.loc[raw_data.Insulin != 0].corr(method='pearson')

fig, ax =  plt.subplots(figsize = (20,10))
ax = sns.heatmap(corr,cmap = 'Blues',fmt='.2g',annot=True)
plt.show()


# In[77]:

corr = raw_data.loc[raw_data.SkinThickness != 0].corr(method='pearson')

fig, ax =  plt.subplots(figsize = (20,10))
ax = sns.heatmap(corr,cmap = 'Blues',fmt='.2g',annot=True)
plt.show()

#%%

def model_fit(dataset):
    values = dataset.values
    X = values[:,1:8]
    Y = values[:,8]
    lda = LinearDiscriminantAnalysis()
    kfold = KFold(n_splits=3)
    result = cross_val_score(lda, X, Y, cv = kfold, scoring="accuracy")
    print("Result of LDA:", result.mean())


# In[78]:

df['SkinThickness'] = df['SkinThickness'].replace(0,np.nan)

imputer = IterativeImputer(max_iter=10, random_state=0)

dfSTImputed = pd.DataFrame(imputer.fit_transform(df.drop('Insulin',axis=1)),
                           columns=df.drop('Insulin',axis=1).columns)

print(dfSTImputed)

corr = dfSTImputed.corr(method='pearson')

fig, ax =  plt.subplots(figsize = (20,10))
ax = sns.heatmap(corr,cmap = 'Blues',fmt='.2g',annot=True)


#%%

df['SkinThickness'] = dfSTImputed['SkinThickness']

df['Insulin'] = df['Insulin'].replace(0,np.nan)

dfInsulinImputed = pd.DataFrame(imputer.fit_transform(df),
                           columns=df.columns)

print(dfInsulinImputed.head())

corr = dfInsulinImputed.corr(method='pearson')

fig, ax =  plt.subplots(figsize = (20,10))
ax = sns.heatmap(corr,cmap = 'Blues',fmt='.2g',annot=True)
plt.show()

model_fit(raw_data)

model_fit(dfInsulinImputed)

# In[96]:


X = dfInsulinImputed.drop({'Outcome'}, axis=1)

y = dfInsulinImputed.pop('Outcome')


# In[97]:

splt= float(input('Train/Test split : ')) # 0.2
max_dpth = int(input('Max Depth of Decision Tree : ')) # 7
min_smpl_splt = int(input('Min Samples before split : ')) # 80

X_train, X_test, train_labels, test_labels = train_test_split(X,y,test_size=splt, random_state=1)
reg_dt_model_gini = DecisionTreeClassifier(criterion = 'gini',max_depth=max_dpth,min_samples_split=min_smpl_splt,random_state=0)
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


pd.DataFrame(reg_dt_model_gini.feature_importances_, columns = ['Imp'], index = X_train.columns).sort_values(by='Imp',ascending = False)


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




