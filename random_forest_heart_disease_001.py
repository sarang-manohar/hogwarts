import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

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
