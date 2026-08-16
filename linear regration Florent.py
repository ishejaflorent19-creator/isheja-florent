#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[7]:


import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize,poly)


# In[8]:


dir()


# In[9]:


A = np.array([3,5,11])
dir(A)


# In[10]:


A.sum()


# In[11]:


Boston = load_data("Boston")
Boston.columns


# In[12]:


X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),
'lstat': Boston['lstat']})
X[:4]


# In[13]:


y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()


# In[14]:


summarize(results)


# In[15]:


design = MS(['lstat'])
design = design.fit(Boston)
X = design.transform(Boston)
X[:4]


# In[16]:


design = MS(['lstat'])
X = design.fit_transform(Boston)
X[:4]


# In[17]:


results.summary()


# In[18]:


results.params


# In[19]:


new_df = pd.DataFrame({'lstat':[5, 10, 15]})
newX = design.transform(new_df)
newX


# In[20]:


new_predictions = results.get_prediction(newX);
new_predictions.predicted_mean


# In[21]:


new_predictions.conf_int(alpha=0.05)


# In[22]:


new_predictions.conf_int(obs=True, alpha=0.05)


# In[24]:


def abline(ax, b, m):
    """Add a line with slope m and intercept b to ax."""
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim)


# In[27]:


ax = subplots(figsize=(8, 8))[1]
ax.scatter(Boston['lstat'], Boston['medv'])
abline(ax, results.params['intercept'], results.params['lstat'])


# In[46]:


vals = []

for i in range(1, X.values.shape[1]):
    vals.append(VIF(X.values, i))

print(vals)


# In[29]:


ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues , results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');


# In[30]:


infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)


# In[31]:


X = MS(['lstat', 'age']).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)


# In[32]:


terms = Boston.columns.drop('medv')
terms


# In[33]:


X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()
summarize(results)


# In[34]:


minus_age = Boston.columns.drop(['medv', 'age'])
Xma = MS(minus_age).fit_transform(Boston)
model1 = sm.OLS(y, Xma)
summarize(model1.fit())


# In[35]:


vals = [VIF(X, i)
for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif':vals},
index=X.columns[1:])
vif


# In[38]:


vals = []

for i in range(1, X.values.shape[1]):
    vals.append(VIF(X.values, i))

pd.DataFrame({'VIF': vals}, index=X.columns[1:])


# In[39]:


X = MS(['lstat',
'age',
('lstat', 'age')]).fit_transform(Boston)
model2 = sm.OLS(y, X)
summarize(model2.fit())


# In[40]:


X = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
summarize(results3)


# In[41]:


anova_lm(results1, results3)


# In[42]:


ax = subplots(figsize=(8,8))[1]
ax.scatter(results3.fittedvalues , results3.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')


# In[43]:


Carseats = load_data('Carseats')
Carseats.columns


# In[44]:


allvars = list(Carseats.columns.drop('Sales'))
y = Carseats['Sales']
final = allvars + [('Income', 'Advertising'),
('Price', 'Age')]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X)
summarize(model.fit())


# In[ ]:




