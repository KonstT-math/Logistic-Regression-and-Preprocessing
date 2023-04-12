import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, shapiro, kruskal


def chi_square_test(df, nx, ny, a):
	x = df.iloc[:, nx]
	y = df.iloc[:, ny]

	contingency = pd.crosstab(x,y,margins = False)
	#print(contingency)

	stat, p, dof, expected = chi2_contingency(contingency)

	# H_0: independent vs H_1: dependent
	if p<a:
		print("dependent (reject H_0). p-value = ", p)
	else:
		print("independent (not reject H_0). p-value = ", p)

# splits the test variable x into groups (in terms of the factor variable y)
def data_by_classes(data, ny, nx):
	
	yName = data.columns[ny]
	y = data[yName]
	yy = y.unique()

	# extract the categories from y, along with 
	# companion values for all other variables:
	Y=[]
	for i in range(len(yy)):
		Y.append(data[data[yName] == yy[i]])

	xName = data.columns[nx]

	# extract the test variable values from each category
	X=[]
	for i in range(len(yy)):
		X.append(Y[i][xName])

	return X

# Hypothesis testing:
def hypo_test(data, nx, X, a):

	xName = data.columns[nx]
	x = data[xName]

	# Shapiro-Wilk for normality testing
	print("Shapiro-Wilk normality test:")
	s, pSW = shapiro(x)
	print("P-value:",pSW)

	if pSW < 0.05:
		print("Kruskal Wallis H-test test:")
		H, pval = kruskal(*X)
		print("H-statistic:", H)
		print("P-Value:", pval)

		if pval < a:
			print("Reject NULL hypothesis - Significant differences exist between groups.")
		else:
			print("Accept NULL hypothesis - No significant difference between groups.")
	else:
		print("One-way ANOVA test:")
		F, pval = f_oneway(*X)
		print("F-statistic:", F)
		print("P-Value:", pval)

		if pval < a:
			print("Reject NULL hypothesis - Significant differences exist between groups.")
		else:
			print("Accept NULL hypothesis - No significant difference between groups.")

	
# ----------------------------------------------------------------------------

data = pd.read_csv('framingham.csv')
df = data.dropna(axis=0)

for i in range(15):
	print("----------------------------------")
	print(i, " , ", df.columns[i], " : ")
	print("For categorical variable: ")
	chi_square_test(df, i, 15, 0.05)
	print("----------------------------------")
	print("For continuous variable: ")
	X = data_by_classes(df,15,i)
	hypo_test(df, i, X, 0.05)
	print("----------------------------------")


# ----------------------------------------------------------------------------
# RESULTS:

# from categorical (0,2,3,5,6,7,8) dependent with target:
# 0, 2, 5, 6, 7, 8

# from continuous (1,4,9,10,11,12,13,14) with significant differences in terms of target: 
# 1, 4, 9, 10, 11, 12, 14

