"""
ISLP - Chapter 2, Exercise 10
Boston Housing Data Set Analysis
Author: Isheja Munyakazi Florent (ID: 101485)
AUCA - Big Data Analytics Masters
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from ISLP import load_data

pd.set_option('display.width', 120)

# ---------------------------------------------------------------
# (a) Load the data set
# ---------------------------------------------------------------
Boston = load_data('Boston')
print("Shape of Boston data set:", Boston.shape)
print("Columns:", Boston.columns.tolist())
# 506 rows (each row = one Boston-area suburb/town)
# 13 columns (each column = one predictor/response variable)

# ---------------------------------------------------------------
# (b) Pairwise scatterplots
# ---------------------------------------------------------------
cols = ['crim', 'zn', 'indus', 'nox', 'rm', 'age',
        'dis', 'tax', 'ptratio', 'lstat', 'medv']
scatter_matrix(Boston[cols], diagonal='hist', alpha=0.4, s=8, figsize=(14, 14))
plt.suptitle("Pairwise Scatterplot Matrix - Boston Housing Data")
plt.tight_layout()
plt.savefig('scatter_matrix.png', dpi=120)
plt.close()

# ---------------------------------------------------------------
# (c) Predictors associated with per-capita crime rate
# ---------------------------------------------------------------
print("\nCorrelation of every predictor with crim:")
print(Boston.corr()['crim'].sort_values(ascending=False))

# ---------------------------------------------------------------
# (d) Suburbs with particularly high crime, tax, ptratio
# ---------------------------------------------------------------
print("\ncrim summary:\n", Boston['crim'].describe())
print("\ntax summary:\n", Boston['tax'].describe())
print("\nptratio summary:\n", Boston['ptratio'].describe())

print("\nSuburbs with crim > 20:", (Boston['crim'] > 20).sum())
print("Suburbs with tax > 600:", (Boston['tax'] > 600).sum())
print("Suburbs with ptratio > 20:", (Boston['ptratio'] > 20).sum())

# ---------------------------------------------------------------
# (e) Suburbs bounding the Charles river (chas == 1)
# ---------------------------------------------------------------
print("\nSuburbs bounding the Charles river:", (Boston['chas'] == 1).sum())

# ---------------------------------------------------------------
# (f) Median pupil-teacher ratio
# ---------------------------------------------------------------
print("\nMedian pupil-teacher ratio:", Boston['ptratio'].median())

# ---------------------------------------------------------------
# (g) Suburb with lowest median home value
# ---------------------------------------------------------------
idx = Boston['medv'].idxmin()
print("\nSuburb with lowest medv (row index {}):".format(idx))
print(Boston.loc[idx])

print("\nHow that suburb's predictors compare to the overall range:")
row = Boston.loc[idx]
for col in Boston.columns:
    pct = (Boston[col] < row[col]).mean() * 100
    print(f"{col:8s}: value={row[col]:>8.3f} | percentile={pct:5.1f}% | "
          f"range=[{Boston[col].min():.3f}, {Boston[col].max():.3f}]")

# ---------------------------------------------------------------
# (h) Suburbs averaging more than 7 / 8 rooms per dwelling
# ---------------------------------------------------------------
print("\nSuburbs with rm > 7:", (Boston['rm'] > 7).sum())
print("Suburbs with rm > 8:", (Boston['rm'] > 8).sum())

sub8 = Boston[Boston['rm'] > 8]
print("\nProfile of suburbs with rm > 8:")
print(sub8.describe())
