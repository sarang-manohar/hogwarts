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
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#%%

raw_data = pd.read_csv(r'C:\Users\saran\Desktop\up-skill\PGP - DSBA\Data Mining\Banking Dataset.csv')

raw_data.head()
# %%
### Predict whether the customer will respond to marketing campaign or not

#1. Cust_ID : Customer Identifier
#2. Target
#3. Age of the customer
#4. Balance : Balance in the bank account
#5. Occupation of the customer
#6. Number of Credit transactions
#7. Age bracket
#8. Credit Score
#9. Holding period

#%%

raw_data.info()
# %%

for feature in raw_data.columns:
    if raw_data[feature].dtype == 'object':
        raw_data[feature] = pd.Categorical(raw_data[feature]).codes
# %%
raw_data.info()
# %%
X = raw_data.drop({'Cust_ID','Target'}, axis=1)

y = raw_data.pop('Target')
# %%

splt= float(input('Train/Test split : ')) # 0.3
num_trees = int(input('Max number of Trees : '))  # 501
max_dpth = int(input('Max Depth of Trees : ')) # 10
max_ft = int(input('Max Features of Trees : ')) # 5
min_smpl_lf = int(input('Min leafs  : ')) # 50 generally taken to be 1-3% of sample size
min_smpl_splt = int(input('Min samples in node before split  : ')) # 110

X_train, X_test, train_labels, test_labels = train_test_split(X,y,test_size=splt, random_state=1)

rfcl = RandomForestClassifier(n_estimators=num_trees,
                                max_depth=max_dpth, max_features = max_ft,
                                min_samples_leaf=min_smpl_lf,
                                min_samples_split = min_smpl_splt,
                                oob_score=True)

rfcl = rfcl.fit(X_train,train_labels)

rfcl.oob_score_
# %%

param_grid = {
    'n_estimators' : [100,200,300,400,500],
    'max_depth' : [10],
    'max_features' : [7],
    'min_samples_leaf' : [50],
    'min_samples_split' : [60]
}

rfcl_test = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rfcl_test, param_grid = param_grid, cv = 3)

prop = grid_search.fit(X_train, train_labels)

#%%
prop.best_params_

# %%  https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6

prm_rng = [100,200,300,400,500]

train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = train_labels, 
                                param_name = 'n_estimators', 
                                param_range = prm_rng, cv = 2)

# %%
vc = pd.DataFrame()

vc['prm_rng'] = prm_rng

sns.set_style("darkgrid")

for i in range(2):
    vc['train_scoreNum'] = train_scoreNum[:,i]
    vc['test_scoreNum'] = test_scoreNum[:,i]
    sns.lineplot(data=vc,x=prm_rng, y= 'test_scoreNum')
    sns.lineplot(data=vc,x=prm_rng, y= 'train_scoreNum')
    plt.show()


# %%
