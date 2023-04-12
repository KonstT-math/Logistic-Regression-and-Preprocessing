# Logistic-Regression-and-Preprocessing
We prepare the data for Logistic Regression in pure python


Logistic Regression with Preprocessing of Dataset

Remarks:

the dataset 'framingham.csv' has missing values and thus, logistic regression does not work

we follow the preprocessing steps below:

1) fill missing values with various methods and see accuracy of logistic regression (preprocess0.py)

2) or delete examples with missing values and see accuracy (preprocess2.py)

3) In any case, we need to standardize the dataset - rescale with mean 0 and std 1 (preprocess1.py)

4) some features are not in relation with the target value. Need to exclude them, and then keep the ones that provide real info about target value. Run logistic regression with these. We decide which will participate in logistic regression by (significance.py)


Contents:

csv files:

framingham.csv : the dataset is publically available on the Kaggle website, and it is from a cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes.

framingham_1.csv : the same dataset with two extra missing values in diaBP; this is in order to work on the imputation of missing data via linear regression (the sysBP and diaBP are correlated - one can compute the Pearson correlation)

fr_nan_free.csv : the dataset with all missing values restored (via mean, mode, median, linear regression)

fr_nan_free_rows.csv : the dataset with rows containing missing values removed

fr_std.csv : the dataset rescaled with mean 0 and standard deviation 1

Preprocessing files:
preprocess0.py, preprocess1.py, preprocess2.py, significance.py

Logistic regression in pure python:
lreg.py, LR_train_sig.py (or LR_train.py)

Logistic regression with sklearn (in order to compare it with my code above):
lr_sklearn.py

