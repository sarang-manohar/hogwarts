import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

import warnings
warnings.filterwarnings('ignore')

#%%

raw_data = pd.read_csv(r'C:\Users\saran\Desktop\up-skill\PGP - DSBA\Data Mining\US_Heart_Patients.csv')

raw_data.head()
# %%
raw_data.info()

# %%

raw_data.describe().T
raw_data.columns
# %%
cont_col = ['age','cigsPerDay','tot cholesterol','Systolic BP', 'Diastolic BP', 'BMI', 'heartRate', 'glucose']

raw_data[cont_col].skew()
# %%
raw_data['Heart-Att'].value_counts(normalize = True)
# %%
raw_data.info()
# %%
# Initiate data imputation for missing values
raw_data.Gender.value_counts()
# %%
raw_data.Gender.fillna('Female', inplace=True)
# %%
raw_data.age.fillna(raw_data.age.median(),inplace =True)
# %%
raw_data.education.value_counts()
# %%
raw_data.education.fillna(1.0, inplace=True)
# %%
raw_data[(raw_data['currentSmoker'] == 1)].info()
# %%
df_cig = raw_data[(raw_data['currentSmoker'] == 1)]
# %%
df_nocig = raw_data[(raw_data['currentSmoker'] == 0)]
# %%
df_cig.cigsPerDay.median()
df_cig.cigsPerDay.fillna(20.0, inplace=True)
# %%
df_cig.info()
df_nocig.info()

df_nocig.cigsPerDay.unique()
# %%
df = pd.DataFrame.append(df_cig,df_nocig)
# %%
df.info()
# %%
df[(df['prevalentHyp'] == 1)]['BP Meds'].unique()

df[(df['prevalentHyp'] == 1)]['BP Meds'].value_counts()

# %%
