"""
ISLP - Chapter 3, Exercise 11
t-statistic for simple linear regression with and without an intercept
Author: ISheja Munyakazi Florent (ID: 101485)
AUCA - Big Data Analytics Masters
"""

import numpy as np
import statsmodels.api as sm

# ---------------------------------------------------------------
# Generate the simulated data set (matches the book's set.seed(1) setup)
# ---------------------------------------------------------------
np.random.seed(1)
x = np.random.normal(size=100)
y = 2 * x + np.random.normal(size=100)

# ---------------------------------------------------------------
# (a) Regress y onto x, WITHOUT an intercept
# ---------------------------------------------------------------
model_a = sm.OLS(y, x).fit()
print("(a) y ~ x, no intercept")
print("    coef  =", model_a.params[0])
print("    SE    =", model_a.bse[0])
print("    t     =", model_a.tvalues[0])
print("    p     =", model_a.pvalues[0])

# ---------------------------------------------------------------
# (b) Regress x onto y, WITHOUT an intercept
# ---------------------------------------------------------------
model_b = sm.OLS(x, y).fit()
print("\n(b) x ~ y, no intercept")
print("    coef  =", model_b.params[0])
print("    SE    =", model_b.bse[0])
print("    t     =", model_b.tvalues[0])
print("    p     =", model_b.pvalues[0])

# ---------------------------------------------------------------
# (d) Verify the algebraic t-statistic formula matches statsmodels
# t = sqrt(n-1) * sum(x*y) / sqrt( sum(x^2)*sum(y^2) - sum(x*y)^2 )
# ---------------------------------------------------------------
n = len(x)
t_manual = (np.sqrt(n - 1) * np.sum(x * y)) / \
           np.sqrt(np.sum(x**2) * np.sum(y**2) - (np.sum(x * y))**2)
print("\n(d) manual t-statistic (formula) =", t_manual)
print("    matches statsmodels t from (a)?", np.isclose(t_manual, model_a.tvalues[0]))

# ---------------------------------------------------------------
# (f) Same comparison, but WITH an intercept in both regressions
# ---------------------------------------------------------------
X_const = sm.add_constant(x)
model_c = sm.OLS(y, X_const).fit()
print("\n(f) y ~ x, WITH intercept: slope t =", model_c.tvalues[1],
      " p =", model_c.pvalues[1])

Y_const = sm.add_constant(y)
model_d = sm.OLS(x, Y_const).fit()
print("(f) x ~ y, WITH intercept: slope t =", model_d.tvalues[1],
      " p =", model_d.pvalues[1])

print("\nt-statistics equal (with intercept)?",
      np.isclose(model_c.tvalues[1], model_d.tvalues[1]))
