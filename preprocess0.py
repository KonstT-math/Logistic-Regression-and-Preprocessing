
# imputation of missing values with various methods (mean, median, mode, linear regression)

import numpy as np
import pandas as pd
from scipy import stats as st
	

# fill missing values with mean/median/mode
def fill_missing(data, column, method):
	Y = data.iloc[:,column]
	if method=='mean':
		m = np.mean(Y)
		for i in range(len(Y)):
			if pd.isna(Y[i]):
				Y[i]=m
		return Y
	elif method=='median':
		m = np.nanmedian(Y)
		for i in range(len(Y)):
			if pd.isna(Y[i]):
				Y[i]=m
		return Y
	elif method=='mode':
		m = Y.mode()[0]
		for i in range(len(Y)):
			if pd.isna(Y[i]):
				Y[i]=m
		return Y
	else:
		print("wrong method")

# fill missing values via linear regression if correlated
def linear_pred_missing(data, col1 , col2):

	# NaN free data for computations:
	data1 = data.dropna(axis=0)	
	
	X_ = data1.iloc[:,col1]
	Y_ = data1.iloc[:,col2]
	
	r ,p_value = st.pearsonr(X_,Y_)

	a = r*(np.std(Y_)/np.std(X_))
	b = np.mean(Y_) - a*np.mean(X_)

	X = data.iloc[:,col1]
	Y = data.iloc[:,col2]

	for i in range(len(Y)):
		if pd.isna(Y[i]):
			Y[i] = a*X[i] + b
	
	return Y

	


df = pd.read_csv('framingham_1.csv')


# see which features have missing values
print(df.info())

# male is fine

# age is fine

# education (categorical-ordinal) has missing values - utilize: mode
y2 = fill_missing(df, 2, 'mode')

# currentSmoker is fine

# cigsPerDay (continuous) has missing values - utilize: median
y4 = fill_missing(df, 4, 'median')

# BPMeds (categorical-nominal) has missing values - utilize: mode
y5 = fill_missing(df, 5, 'mode')

# prevalentStroke is fine

# prevalentHyp is fine

# diabetes is fine

# totChol (continuous) has missing values - utilize: mean
y9 = fill_missing(df, 9, 'mean')

# sysBP is fine

# diaBP (continuous) has missing values - utilize: linear regression, since sysBP and diaBP are correlated 
y11 = linear_pred_missing(df, 10 , 11)

# BMI (continuous) has missing values - utilize: mean
y12 = fill_missing(df, 12, 'mean')

# heartRate (continuous) has one missing value - utilize: mean
y13 = fill_missing(df, 13, 'mean')

# glucose (continuous) has missing values - utilize : mean
y14 = fill_missing(df, 14, 'mean')

# TenYearCHD (target) is fine

# new dataframe
result = df
# create new csv file with new dataframe
result.to_csv(r'fr_nan_free.csv', index = False, header=True)

