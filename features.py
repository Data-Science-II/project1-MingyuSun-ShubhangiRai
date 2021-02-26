# Mingyu Sun's code:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# import statsmodels as sm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import preprocessing

# store auto-mpg.csv locally and directly read this file.
cars = pd.read_csv('auto-mpg.csv',index_col='car name')

# create a DataFrame of independent variables
X = cars.drop('mpg',axis=1)
# create a series of the dependent variable
y = cars.mpg
print(X.shape)

def forward_selection(data, target, significance_level):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

def backward_elimination(data, target,significance_level):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break
        else:
            break
    return best_features

# run with
forward_selection(X,y)
backward_elimination(X,y)
stepwise_selection(X,y)

# another way to do features selection with python package
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

forwardsfs = SFS(LinearRegression(),
          k_features=5,
          forward=True,
          floating=False,
          scoring = 'r2')
backwardsfs  = SFS(LinearRegression(),
         k_features=5,
         forward=False,
         floating=False)
stepsfs = SFS(LinearRegression(),
         k_features=5,
         forward=True)

# Shubhangi Rai's code
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

estimator = SVR(kernel="linear")
#Use RFE to remove not significant features from the initial model.
min_features_to_select = 1
selector = RFECV(estimator, step=1, cv=10,min_features_to_select = min_features_to_select)
selector = selector.fit(X_train,Y_train)
#Test new model
#New features dataframe containing only selected features through RFE
X_RFE = X_train[X_train.columns[selector.support_]]

r_sq = selector.grid_scores_

### Quad Regression

x= range(1,8)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X_RFE)

poly.fit(X_RFE, Y_train)
lin4 = LinearRegression()
lin4.fit(X_RFE, Y_train)

Y_pred = lin4.predict(X_RFE)

r_sq = lin4.score(X_RFE,Y_train)

ad_r_sq = (1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1))

from sklearn.model_selection import cross_val_score
r_cv = cross_val_score(lin4, X_RFE, Y_train, cv=7)

# Visualising the Linear Regression results
plt.plot( x, r_cv, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( x, r_sq, marker='', color='', linewidth=2)
plt.plot( x, ad_r_sq, marker='', color='red', linewidth=2,label="toto")
plt.title('StepwiseRegression Quad')

plt.legend()
plt.show()
