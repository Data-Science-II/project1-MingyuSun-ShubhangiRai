#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np
data = pd.read_csv("/home/anant/Desktop/auto-mpg.csv",header = None)


# In[99]:


data.shape


# In[100]:


data = data.fillna(0)
data = data.replace('?',0)


# In[101]:


def convert_str_int(x):
    print(x)
    return int(x)
data[3] = data[3].map(lambda x: convert_str_int(x))


# In[111]:


###########Train Dataset and Test Dataset Creation########
X_train = data[data.columns[1:]]
Y_train = data[0]


# ### Stepwise regression

# In[326]:


from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline


# In[341]:


def rfecvv(lin, X, Y):
    # create pipeline
    rfe = RFE(estimator=lin, n_features_to_select=5)
    model = lin
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    cv = RepeatedKFold(n_splits=7, n_repeats=2, random_state=1)
    r_sq = cross_val_score(pipeline, X, Y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
    return rfe, r_sq


# ### Quad Regression

# In[342]:


lin4 = LinearRegression()


# In[343]:


# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import cross_val_score
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X_train) 
poly.fit(X_train, Y_train) 
lin4.fit(X_train, Y_train)


# In[344]:


X = pd.DataFrame(X_poly)


# In[345]:


X_RFE, r_sq = rfecvv(lin4, X, Y_train)


# In[346]:


ad_r_sq = (1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1))


# In[347]:


r_cv = cross_val_score(lin4, X_train, Y_train, cv=14)


# In[348]:


r_cv,r_sq,ad_r_sq


# In[385]:


x= range(1,15)
# Visualising the Linear Regression results 
plt.plot( x, r_cv, marker='o',color='skyblue', linewidth=4,label="r_cv")
plt.plot( x, r_sq, marker='o', color='blue', linewidth=4, label=" r_sq")
plt.plot( x, ad_r_sq, marker='o', color='red', linewidth=2,label="adjusted r_sq")
plt.title('StepwiseRegression Quad') 

plt.legend()
plt.show() 
plt.savefig('StepwiseRegression_Quad.png')


# ### Cubic Regression

# In[384]:


lin = LinearRegression()

# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import cross_val_score
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(X_train) 
poly.fit(X_train, Y_train) 
lin.fit(X_train, Y_train)

X = pd.DataFrame(X_poly)

X_RFE, r_sq = rfecvv(lin, X, Y_train)

ad_r_sq = (1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1))

r_cv = cross_val_score(lin, X_train, Y_train, cv=14)

r_cv,r_sq,ad_r_sq

x= range(1,15)
# Visualising the Linear Regression results 
plt.plot( x, r_cv, marker='o',color='skyblue', linewidth=4,label="r_cv")
plt.plot( x, r_sq, marker='o', color='blue', linewidth=4, label=" r_sq")
plt.plot( x, ad_r_sq, marker='o', color='red', linewidth=2,label="adjusted r_sq")
plt.title('StepwiseRegression Cubic') 

plt.legend()
plt.show() 
plt.savefig('StepwiseRegression_Cubic.png')


# ### Mutiple Linear Regression

# In[383]:


lin = LinearRegression()
from sklearn.model_selection import cross_val_score
lin.fit(X_train, Y_train)

X_RFE, r_sq = rfecvv(lin, X_train, Y_train)

ad_r_sq = (1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1))

r_cv = cross_val_score(lin, X_train, Y_train, cv=14)

r_cv,r_sq,ad_r_sq

x= range(1,15)
# Visualising the Linear Regression results 
plt.plot( x, r_cv, marker='o',color='skyblue', linewidth=4,label="r_cv")
plt.plot( x, r_sq, marker='o', color='blue', linewidth=4, label=" r_sq")
plt.plot( x, ad_r_sq, marker='o', color='red', linewidth=2,label="adjusted r_sq")
plt.title('StepwiseRegression MLR') 
plt.legend()
plt.show()
plt.savefig('StepwiseRegression_MLR.png')


# ### Lasso

# In[389]:


from sklearn.linear_model import Lasso


# In[393]:


clf = linear_model.Lasso()


# In[398]:


pipeline = Pipeline([('m',clf)])
cv = RepeatedKFold(n_splits=7, n_repeats=2, random_state=1)
r_sq = cross_val_score(pipeline, X_train, Y_train, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')


# In[400]:


ad_r_sq = (1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1))


# In[401]:


r_cv = cross_val_score(clf, X_train, Y_train, cv=14)


# In[403]:


x= range(1,15)
# Visualising the Linear Regression results 
plt.plot( x, r_cv, marker='o',color='skyblue', linewidth=4,label="r_cv")
plt.plot( x, r_sq, marker='o', color='blue', linewidth=4, label=" r_sq")
plt.plot( x, ad_r_sq, marker='o', color='red', linewidth=2,label="adjusted r_sq")
plt.title('Lasso MLR') 
plt.legend()
plt.show()
plt.savefig('Lasso_MLR.png')


# ### Ridge
# 

# In[404]:


from sklearn.linear_model import Ridge
clf = linear_model.Ridge()

pipeline = Pipeline([('m',clf)])
cv = RepeatedKFold(n_splits=7, n_repeats=2, random_state=1)
r_sq = cross_val_score(pipeline, X_train, Y_train, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')

ad_r_sq = (1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1))

r_cv = cross_val_score(clf, X_train, Y_train, cv=14)

x= range(1,15)
# Visualising the Linear Regression results 
plt.plot( x, r_cv, marker='o',color='skyblue', linewidth=4,label="r_cv")
plt.plot( x, r_sq, marker='o', color='blue', linewidth=4, label=" r_sq")
plt.plot( x, ad_r_sq, marker='o', color='red', linewidth=2,label="adjusted r_sq")
plt.title('Ridge MLR') 
plt.legend()
plt.show()
plt.savefig('Ridge_MLR.png')


# In[ ]:




